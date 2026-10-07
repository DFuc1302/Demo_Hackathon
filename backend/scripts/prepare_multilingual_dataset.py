from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.dataset_contract import (
    MULTILINGUAL_LANGUAGES,
    check_multilingual_cross_split_leakage,
    decontaminate_multilingual_splits,
    normalize_text_key,
    serialize_jsonl,
    sha256_bytes,
    validate_multilingual_split_rows,
)
from app.multilingual_manifest import (
    CLASSIFIER_MODEL_ID,
    CLASSIFIER_MODEL_REVISION,
    SPLITS,
    validate_multilingual_manifest,
)
from app.translation_artifact import (
    LANGUAGES,
    file_sha256,
    validate_language_codes,
    validate_translation_artifact,
)
from app.translation import TranslationEngine

DEFAULT_V2_MANIFEST = PROJECT_ROOT / "data" / "v2_manifest.json"
DEFAULT_V2_DIR = PROJECT_ROOT / "data" / "v2"
DEFAULT_MULTILINGUAL_MANIFEST = PROJECT_ROOT / "data" / "multilingual_manifest.json"
DEFAULT_MULTILINGUAL_DIR = PROJECT_ROOT / "data" / "multilingual"
DEFAULT_TRANSLATION_MODEL = PROJECT_ROOT / "models" / "translator" / "m2m100_418M"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"could not read jsonl file {path}: {exc}") from exc
    rows = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        clean = line.strip()
        if not clean:
            continue
        try:
            row = json.loads(clean)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
        rows.append(row)
    return rows


def _write_atomic(target_path: Path, content: bytes) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=target_path.parent,
        prefix=f".{target_path.name}.",
        delete=False,
    ) as temp_file:
        temp_file.write(content)
        temp_path = Path(temp_file.name)
    temp_path.replace(target_path)


def _load_v2_source_splits(
    v2_manifest_path: Path,
    v2_dir: Path,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    if not v2_manifest_path.is_file():
        raise FileNotFoundError(f"source V2 manifest not found: {v2_manifest_path}")
    try:
        manifest = json.loads(v2_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid source V2 manifest at {v2_manifest_path}: {exc}") from exc

    splits: dict[str, list[dict[str, Any]]] = {}
    for split_name in SPLITS:
        split_path = v2_dir / f"{split_name}.jsonl"
        if not split_path.is_file():
            raise FileNotFoundError(f"source V2 split file missing: {split_path}")
        content = split_path.read_bytes()
        expected_sha = manifest["splits"][split_name]["prepared_sha256"]
        if sha256_bytes(content) != expected_sha:
            raise ValueError(
                f"source V2 {split_name} split hash mismatch against {v2_manifest_path}"
            )
        rows = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
        splits[split_name] = rows
    return manifest, splits


# Uses app.translation.TranslationEngine directly

def bootstrap_multilingual_candidate(
    *,
    candidate_dir: str | Path,
    v2_manifest_path: str | Path = DEFAULT_V2_MANIFEST,
    v2_dir: str | Path = DEFAULT_V2_DIR,
    translation_model_path: str | Path = DEFAULT_TRANSLATION_MODEL,
    translator_factory: Callable[[Path], Any] | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    candidate_path = Path(candidate_dir)
    if candidate_path.exists():
        raise FileExistsError(
            f"candidate directory already exists; bootstrap requires an absent directory: {candidate_path}"
        )
    candidate_path.mkdir(parents=True)

    v2_manifest_path = Path(v2_manifest_path)
    v2_dir = Path(v2_dir)
    translation_model_path = Path(translation_model_path)

    v2_manifest, v2_splits = _load_v2_source_splits(v2_manifest_path, v2_dir)
    source_v2_manifest_sha256 = file_sha256(v2_manifest_path)
    translation_artifact = validate_translation_artifact(translation_model_path)

    if translator_factory is not None:
        translator = translator_factory(translation_model_path)
    else:
        translator = TranslationEngine(translation_model_path)

    generated_raw_splits: dict[str, list[dict[str, Any]]] = {}
    for split_name in SPLITS:
        split_rows = v2_splits[split_name]
        if smoke:
            split_rows = split_rows[:10]
        
        english_texts = [" ".join(str(r["text"]).split()) for r in split_rows]
        labels = [int(r["label"]) for r in split_rows]
        source_ids = [
            hashlib.sha256(normalize_text_key(t).encode("utf-8")).hexdigest()
            for t in english_texts
        ]

        # Translate in batches per language
        translations_by_lang: dict[str, list[str]] = {}
        for target_lang in ("sw", "ha", "bn"):
            print(f"Translating {split_name} ({len(english_texts)} rows) to {target_lang}...", flush=True)
            if hasattr(translator, "translate_batch"):
                translations_by_lang[target_lang] = translator.translate_batch(
                    english_texts,
                    source_language="en",
                    target_language=target_lang,
                    batch_size=8,
                )
            else:
                translations_by_lang[target_lang] = [
                    translator.translate(t, source_language="en", target_language=target_lang)
                    for t in english_texts
                ]

        generated_rows: list[dict[str, Any]] = []
        for i in range(len(split_rows)):
            e_text = english_texts[i]
            lbl = labels[i]
            s_id = source_ids[i]
            generated_rows.append(
                {"text": e_text, "label": lbl, "language": "en", "source_id": s_id}
            )
            for target_lang in ("sw", "ha", "bn"):
                generated_rows.append(
                    {
                        "text": translations_by_lang[target_lang][i],
                        "label": lbl,
                        "language": target_lang,
                        "source_id": s_id,
                    }
                )
        generated_raw_splits[split_name] = generated_rows
        (candidate_path / f"raw_{split_name}.jsonl").write_bytes(serialize_jsonl(generated_rows))
    cleaned_splits, removal_counts_by_split = decontaminate_multilingual_splits(
        generated_raw_splits
    )

    manifest_splits: dict[str, Any] = {}
    for split_name in SPLITS:
        rows = cleaned_splits[split_name]
        serialized = serialize_jsonl(rows)
        split_file = candidate_path / f"{split_name}.jsonl"
        split_file.write_bytes(serialized)
        prepared_sha256 = sha256_bytes(serialized)
        source_groups = len(rows) // 4
        counts = {
            lang: {
                "0": sum(1 for r in rows if r["language"] == lang and r["label"] == 0),
                "1": sum(1 for r in rows if r["language"] == lang and r["label"] == 1),
            }
            for lang in MULTILINGUAL_LANGUAGES
        }
        manifest_splits[split_name] = {
            "prepared_sha256": prepared_sha256,
            "rows": len(rows),
            "source_groups": source_groups,
            "language_label_counts": counts,
            "decontamination": removal_counts_by_split[split_name],
        }

    candidate_manifest: dict[str, Any] = {
        "schema_version": 3,
        "source_v2_manifest_sha256": source_v2_manifest_sha256,
        "dataset_id": v2_manifest["dataset_id"],
        "dataset_revision": v2_manifest["dataset_revision"],
        "languages": list(LANGUAGES),
        "model_id": CLASSIFIER_MODEL_ID,
        "model_revision": CLASSIFIER_MODEL_REVISION,
        "license": "mit",
        "translation_artifact": translation_artifact,
        "splits": manifest_splits,
    }

    validate_multilingual_manifest(
        candidate_manifest,
        source_v2_manifest_path=v2_manifest_path,
        translation_model_path=translation_model_path,
    )

    manifest_bytes = (json.dumps(candidate_manifest, indent=2) + "\n").encode("utf-8")
    (candidate_path / "multilingual_manifest.json").write_bytes(manifest_bytes)
    return candidate_manifest


def promote_multilingual_candidate(
    candidate_manifest_path: str | Path,
    *,
    target_manifest_path: str | Path = DEFAULT_MULTILINGUAL_MANIFEST,
    source_v2_manifest_path: str | Path = DEFAULT_V2_MANIFEST,
    translation_model_path: str | Path = DEFAULT_TRANSLATION_MODEL,
    output_dir: str | Path = DEFAULT_MULTILINGUAL_DIR,
) -> dict[str, Any]:
    candidate_manifest_path = Path(candidate_manifest_path)
    target_manifest_path = Path(target_manifest_path)
    output_dir = Path(output_dir)

    if not candidate_manifest_path.is_file():
        raise FileNotFoundError(f"candidate manifest missing: {candidate_manifest_path}")

    try:
        manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not parse candidate manifest: {exc}") from exc

    validate_multilingual_manifest(
        manifest,
        source_v2_manifest_path=source_v2_manifest_path,
        translation_model_path=translation_model_path,
    )

    candidate_dir = candidate_manifest_path.parent
    staged_splits: dict[str, list[dict[str, Any]]] = {}
    for split_name in SPLITS:
        split_file = candidate_dir / f"{split_name}.jsonl"
        if not split_file.is_file():
            raise FileNotFoundError(f"candidate split file missing: {split_file}")
        content = split_file.read_bytes()
        expected_spec = manifest["splits"][split_name]
        if sha256_bytes(content) != expected_spec["prepared_sha256"]:
            raise ValueError(f"candidate {split_name} hash does not match candidate manifest")
        rows = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
        validated_rows = validate_multilingual_split_rows(rows, split_name, expected_spec)
        staged_splits[split_name] = validated_rows

    check_multilingual_cross_split_leakage(staged_splits)

    staging_dir = output_dir.with_name(".staging_multilingual_dataset")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    for split_name in SPLITS:
        source_split = candidate_dir / f"{split_name}.jsonl"
        dest_split = staging_dir / f"{split_name}.jsonl"
        dest_split.write_bytes(source_split.read_bytes())

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        previous_dir = output_dir.with_name(".previous_multilingual_dataset")
        if previous_dir.exists():
            shutil.rmtree(previous_dir)
        output_dir.rename(previous_dir)
        try:
            staging_dir.rename(output_dir)
            if previous_dir.exists():
                shutil.rmtree(previous_dir)
        except Exception:
            if previous_dir.exists():
                previous_dir.rename(output_dir)
            raise
    else:
        staging_dir.rename(output_dir)

    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    _write_atomic(target_manifest_path, manifest_bytes)
    return manifest


def verify_or_regenerate_offline_dataset(
    *,
    manifest_path: str | Path = DEFAULT_MULTILINGUAL_MANIFEST,
    data_dir: str | Path = DEFAULT_MULTILINGUAL_DIR,
    source_v2_manifest_path: str | Path = DEFAULT_V2_MANIFEST,
    translation_model_path: str | Path = DEFAULT_TRANSLATION_MODEL,
) -> dict[str, dict[str, Any]]:
    manifest_path = Path(manifest_path)
    data_dir = Path(data_dir)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"committed multilingual manifest missing: {manifest_path}")

    committed_bytes = manifest_path.read_bytes()
    try:
        manifest = json.loads(committed_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"committed multilingual manifest is invalid JSON: {exc}") from exc

    validate_multilingual_manifest(
        manifest,
        source_v2_manifest_path=source_v2_manifest_path,
        translation_model_path=translation_model_path,
    )

    results: dict[str, dict[str, Any]] = {}
    splits: dict[str, list[dict[str, Any]]] = {}
    for split_name in SPLITS:
        split_file = data_dir / f"{split_name}.jsonl"
        if not split_file.is_file():
            raise FileNotFoundError(
                f"offline prepared multilingual split is missing: {split_file}"
            )
        content = split_file.read_bytes()
        expected = manifest["splits"][split_name]
        actual_sha = sha256_bytes(content)
        if actual_sha != expected["prepared_sha256"]:
            raise ValueError(
                f"prepared {split_name} hash does not match committed manifest: {actual_sha} != {expected['prepared_sha256']}"
            )
        rows = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
        validated = validate_multilingual_split_rows(rows, split_name, expected)
        splits[split_name] = validated
        results[split_name] = {
            "rows": len(validated),
            "sha256": actual_sha,
            "path": str(split_file),
        }

    check_multilingual_cross_split_leakage(splits)
    if manifest_path.read_bytes() != committed_bytes:
        raise RuntimeError("offline mode modified the committed manifest")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap candidate, promote, or offline verify the schema-3 multilingual prompt-injection dataset."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--bootstrap-manifest",
        action="store_true",
        help="Bootstrap candidate manifest and staged jsonl into candidate dir.",
    )
    group.add_argument(
        "--promote-candidate",
        metavar="CANDIDATE_MANIFEST",
        help="Revalidate candidate and atomically promote it to committed manifest.",
    )
    group.add_argument(
        "--offline",
        action="store_true",
        help="Verify the committed manifest and offline generated jsonl splits without modifying manifest.",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        help="Target candidate directory for --bootstrap-manifest (must be absent).",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MULTILINGUAL_MANIFEST,
        help="Committed multilingual manifest path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_MULTILINGUAL_DIR,
        help="Destination directory for multilingual jsonl files.",
    )
    parser.add_argument(
        "--source-v2-manifest",
        type=Path,
        default=DEFAULT_V2_MANIFEST,
        help="Path to source V2 manifest.",
    )
    parser.add_argument(
        "--source-v2-dir",
        type=Path,
        default=DEFAULT_V2_DIR,
        help="Path to source V2 dataset directory.",
    )
    parser.add_argument(
        "--translation-model-path",
        type=Path,
        default=DEFAULT_TRANSLATION_MODEL,
        help="Path to installed translation artifact.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run on small slice of V2 splits for fast verification.",
    )
    args = parser.parse_args()

    if args.bootstrap_manifest:
        if not args.candidate_dir:
            parser.error("--bootstrap-manifest requires --candidate-dir")
        bootstrap_multilingual_candidate(
            candidate_dir=args.candidate_dir,
            v2_manifest_path=args.source_v2_manifest,
            v2_dir=args.source_v2_dir,
            translation_model_path=args.translation_model_path,
            smoke=args.smoke,
        )
    elif args.promote_candidate:
        promote_multilingual_candidate(
            args.promote_candidate,
            target_manifest_path=args.manifest_path,
            source_v2_manifest_path=args.source_v2_manifest,
            translation_model_path=args.translation_model_path,
            output_dir=args.output_dir,
        )
    elif args.offline:
        verify_or_regenerate_offline_dataset(
            manifest_path=args.manifest_path,
            data_dir=args.output_dir,
            source_v2_manifest_path=args.source_v2_manifest,
            translation_model_path=args.translation_model_path,
        )


if __name__ == "__main__":
    main()
