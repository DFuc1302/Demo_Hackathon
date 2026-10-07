from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from app.translation_artifact import (LANGUAGES, MODEL_ID, MODEL_REVISION,
                                      file_sha256, validate_translation_artifact)

FILES = ("config.json", "generation_config.json", "pytorch_model.bin",
         "sentencepiece.bpe.model", "special_tokens_map.json", "tokenizer_config.json", "vocab.json")
WORKER = '''import errno, json, socket, sys
from pathlib import Path
sys.path.insert(0, "/work")
from translation_artifact import validate_language_codes
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
assert not Path("/mnt/d/Demo_Hackathon").exists()
assert not Path("/home/dtp").exists()
try:
    socket.create_connection(("1.1.1.1", 443), timeout=1)
except OSError as exc:
    assert exc.errno in (errno.ENETUNREACH, errno.EHOSTUNREACH)
else:
    raise RuntimeError("conversion sandbox has network access")
tokenizer = M2M100Tokenizer.from_pretrained("/work/input", local_files_only=True)
validate_language_codes(tokenizer)
model = M2M100ForConditionalGeneration.from_pretrained("/work/input", local_files_only=True, use_safetensors=False, weights_only=True)
model.save_pretrained("/work/output", safe_serialization=True)
tokenizer.save_pretrained("/work/output")
print("ISOLATED_SAFE_CONVERSION_OK", flush=True)
'''
SANDBOX = '''set -eu
root="$1"
work="$2"
venv="$3"
mount --make-rprivate /
for entry in usr lib lib64; do
    mount --rbind "/$entry" "$root/$entry"
    mount -o remount,bind,ro,nosuid,nodev "$root/$entry"
done
mount --bind "$venv" "$root/runtime"
mount -o remount,bind,ro,nosuid,nodev "$root/runtime"
mount --bind "$work" "$root/work"
for entry in null urandom random; do
    mount --bind "/dev/$entry" "$root/dev/$entry"
done
mount -t proc proc "$root/proc"
exec chroot "$root" /usr/bin/setpriv --bounding-set=-all --no-new-privs /runtime/bin/python -I /work/convert.py
'''


def prepare(output: Path) -> dict:
    if output.exists():
        metadata = validate_translation_artifact(output)
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
        from app.translation_artifact import validate_language_codes
        validate_language_codes(M2M100Tokenizer.from_pretrained(output, local_files_only=True))
        M2M100ForConditionalGeneration.from_pretrained(output, local_files_only=True, use_safetensors=True)
        return metadata
    for binary in ("unshare", "mount", "chroot", "setpriv"):
        if shutil.which(binary) is None:
            raise RuntimeError(f"isolated conversion requires {binary}; refusing unsafe fallback")
    api_url = f"https://huggingface.co/api/models/{MODEL_ID}/revision/{MODEL_REVISION}?blobs=true"
    with urllib.request.urlopen(api_url, timeout=30) as response:
        info = json.load(response)
    if info.get("id") != MODEL_ID or info.get("sha") != MODEL_REVISION:
        raise ValueError("translator model identity or exact revision mismatch")
    with urllib.request.urlopen(f"https://huggingface.co/{MODEL_ID}/raw/{MODEL_REVISION}/README.md", timeout=30) as response:
        card = response.read().decode("utf-8")
    front_matter = yaml.safe_load(card.split("---", 2)[1])
    if front_matter.get("license") != "mit":
        raise ValueError("translator model card must declare exact MIT license")
    siblings = {item["rfilename"]: item for item in info["siblings"]}
    with tempfile.TemporaryDirectory(prefix="m2m100-conversion-") as temporary:
        temporary = Path(temporary)
        root, work = temporary / "root", temporary / "work"
        for name in ("usr", "lib", "lib64", "runtime", "work", "dev", "proc", "tmp"):
            (root / name).mkdir(parents=True, exist_ok=True)
        for name in ("null", "urandom", "random"):
            (root / "dev" / name).touch()
        source = work / "input"
        source.mkdir(parents=True)
        for name in FILES:
            if name not in siblings:
                raise ValueError(f"pinned translator is missing {name}")
            url = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}"
            print(f"Downloading pinned {name}", flush=True)
            with urllib.request.urlopen(url, timeout=120) as response, (source / name).open("wb") as target:
                shutil.copyfileobj(response, target, length=1024 * 1024)
            expected_hash = siblings[name].get("lfs", {}).get("sha256")
            if expected_hash and file_sha256(source / name) != expected_hash:
                raise ValueError(f"pinned download hash mismatch: {name}")
        (work / "convert.py").write_text(WORKER, encoding="utf-8")
        shutil.copyfile(PROJECT_ROOT / "app" / "translation_artifact.py", work / "translation_artifact.py")
        environment = {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "HOME": "/tmp", "TMPDIR": "/tmp",
                       "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": ""}
        subprocess.run(["unshare", "--user", "--map-root-user", "--mount", "--net", "--pid", "--fork",
                        "bash", "-c", SANDBOX, "sandbox", str(root), str(work), sys.prefix],
                       env=environment, cwd=temporary, check=True)
        generated = work / "output"
        metadata = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "license": "mit",
                    "languages": list(LANGUAGES),
                    "sha256": {item.name: file_sha256(item) for item in sorted(generated.iterdir()) if item.is_file()}}
        (generated / "artifact.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        validate_translation_artifact(generated)
        output.parent.mkdir(parents=True, exist_ok=True)
        # Copy beside the destination, then rename: readers never see a partially copied artifact.
        with tempfile.TemporaryDirectory(prefix=".translator-staging-", dir=output.parent) as staging:
            staged = Path(staging) / "artifact"
            shutil.copytree(generated, staged)
            validate_translation_artifact(staged)
            if output.exists():
                raise FileExistsError("translator destination appeared during preparation; refusing overwrite")
            staged.rename(output)
        return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare only the pinned M2M100 safe local artifact")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models" / "translator" / "m2m100_418M")
    args = parser.parse_args()
    print(json.dumps(prepare(args.model_dir.resolve()), indent=2))


if __name__ == "__main__":
    main()
