from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import sys
import urllib.request
from pathlib import Path
from tempfile import NamedTemporaryFile

import yaml

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.dataset_contract import (check_cross_split_leakage, normalize_text_key,
                                  serialize_jsonl, sha256_bytes, validate_split_rows)

REVISION_API = "https://huggingface.co/api/datasets/{dataset_id}/revision/{dataset_revision}"
README_URL = "https://huggingface.co/datasets/{dataset_id}/raw/{dataset_revision}/README.md"
RAW_URL = "https://huggingface.co/datasets/{dataset_id}/resolve/{revision}/{source_file}"


def _sha256(data: bytes) -> str:
    return sha256_bytes(data)
def _load_manifest(path: Path) -> dict:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read dataset manifest {path}: {exc}") from exc
    required = {"schema_version", "dataset_id", "dataset_revision", "model_id", "model_revision", "license", "language", "columns", "splits"}
    if set(manifest) != required or manifest["schema_version"] != 2:
        raise ValueError(f"dataset manifest {path} must use schema 2 with the required fields")
    if manifest["license"].lower() != "mit" or manifest["language"].lower() != "en":
        raise ValueError("the pinned dataset must declare the MIT license and English language")
    if manifest["columns"] != ["text", "label"] or set(manifest["splits"]) != {"train", "validation", "test"}:
        raise ValueError("the pinned dataset must declare train, validation, and test text/label splits")
    required_split_keys = {"source_file", "source_sha256", "prepared_sha256", "rows", "class_counts"}
    for split_name, split_spec in manifest["splits"].items():
        if not isinstance(split_spec, dict) or set(split_spec.keys()) != required_split_keys:
            raise ValueError(f"split {split_name} in manifest {path} is missing required split keys")
        if not isinstance(split_spec["prepared_sha256"], str) or len(split_spec["prepared_sha256"]) != 64:
            raise ValueError(f"split {split_name} has an invalid prepared_sha256 hash")
    return manifest


def _parse_yaml_front_matter(readme_text: str) -> dict:
    parts = readme_text.split("---", 2)
    if len(parts) < 3:
        raise RuntimeError("the pinned dataset README does not expose YAML front matter")
    try:
        data = yaml.safe_load(parts[1])
    except Exception as exc:
        raise RuntimeError(f"could not parse dataset YAML front matter: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("the pinned dataset README front matter is not a valid YAML mapping")
    return data


def _verify_license_and_revision(manifest: dict) -> None:
    api_url = REVISION_API.format(
        dataset_id=manifest["dataset_id"],
        dataset_revision=manifest["dataset_revision"],
    )
    try:
        with urllib.request.urlopen(api_url, timeout=30) as response:
            info = json.load(response)
    except Exception as exc:
        raise RuntimeError(f"could not verify Hugging Face dataset revision: {exc}") from exc
    if info.get("sha") != manifest["dataset_revision"]:
        raise RuntimeError("pinned dataset revision is not the current repository revision; update the manifest only after review")

    readme_url = README_URL.format(
        dataset_id=manifest["dataset_id"],
        dataset_revision=manifest["dataset_revision"],
    )
    try:
        with urllib.request.urlopen(readme_url, timeout=30) as response:
            readme_text = response.read().decode("utf-8")
    except Exception as exc:
        raise RuntimeError(f"could not fetch dataset README: {exc}") from exc

    front_matter = _parse_yaml_front_matter(readme_text)
    license_val = str(front_matter.get("license", "")).lower()
    if license_val != "mit":
        raise RuntimeError("the pinned dataset does not expose a verifiable MIT license")

    lang_val = front_matter.get("language")
    is_en = (isinstance(lang_val, str) and lang_val.lower() == "en") or (
        isinstance(lang_val, list) and any(str(item).lower() == "en" for item in lang_val)
    )
    if not is_en:
        raise RuntimeError("the pinned dataset does not declare English language")


def _read_split(raw: bytes, split: str, expected: dict) -> list[dict[str, int | str]]:
    if sha256_bytes(raw) != expected["source_sha256"]:
        raise ValueError(f"{split} source hash does not match the immutable manifest")
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    if not rows or set(rows[0].keys()) != {"text", "label"}:
        raise ValueError(f"{split} must contain exactly text and label columns")
    parsed_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=2):
        try:
            label_val = int(str(row.get("label", "")))
        except ValueError as exc:
            raise ValueError(f"{split} row {index} has a non-integer label") from exc
        parsed_rows.append({"text": row.get("text", ""), "label": label_val})
    return validate_split_rows(parsed_rows, split, expected)


def _validate_offline_dataset(output_dir: Path, manifest: dict) -> dict[str, dict[str, str | int]]:
    splits: dict[str, list[dict[str, int | str]]] = {}
    results = {}
    for split, expected in manifest["splits"].items():
        destination = output_dir / f"{split}.jsonl"
        if not destination.exists():
            raise FileNotFoundError(f"offline prepared dataset split is missing: {destination}; rerun without --offline to download pinned data")
        try:
            content = destination.read_bytes()
        except OSError as exc:
            raise ValueError(f"could not read offline prepared split: {destination}: {exc}") from exc

        if sha256_bytes(content) != expected["prepared_sha256"]:
            raise ValueError(f"prepared {split} hash does not match the immutable manifest: {destination}; rerun preparation")

        try:
            rows = [json.loads(line) for line in content.decode("utf-8").splitlines() if line]
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"offline cached split is invalid: {destination}") from exc

        validated_rows = validate_split_rows(rows, split, expected)
        splits[split] = validated_rows
        results[split] = {"rows": len(validated_rows), "sha256": expected["prepared_sha256"], "path": str(destination)}

    check_cross_split_leakage(splits)
    return results


def prepare_dataset(manifest_path: str | Path, output_dir: str | Path, *, offline: bool = False) -> dict[str, dict[str, str | int]]:
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    manifest = _load_manifest(manifest_path)

    if offline:
        return _validate_offline_dataset(output_dir, manifest)

    _verify_license_and_revision(manifest)

    splits: dict[str, list[dict[str, int | str]]] = {}
    serialized_splits: dict[str, bytes] = {}

    for split, expected in manifest["splits"].items():
        url = RAW_URL.format(
            dataset_id=manifest["dataset_id"],
            revision=manifest["dataset_revision"],
            source_file=expected["source_file"],
        )
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                raw = response.read()
        except Exception as exc:
            raise RuntimeError(f"could not download pinned {split} split: {exc}") from exc

        validated_rows = _read_split(raw, split, expected)
        serialized = serialize_jsonl(validated_rows)
        if sha256_bytes(serialized) != expected["prepared_sha256"]:
            raise ValueError(f"{split} prepared hash does not match the immutable manifest")

        splits[split] = validated_rows
        serialized_splits[split] = serialized

    check_cross_split_leakage(splits)

    staging_dir = output_dir.with_name(".staging_v2_dataset")
    previous_dir = output_dir.with_name(".previous_v2_dataset")

    if staging_dir.exists():
        raise RuntimeError(f"staging directory conflict exists at {staging_dir}; resolve manually before preparing dataset")
    if previous_dir.exists():
        raise RuntimeError(f"previous directory conflict exists at {previous_dir}; resolve manually before preparing dataset")

    staging_dir.mkdir(parents=True)
    for split, content in serialized_splits.items():
        (staging_dir / f"{split}.jsonl").write_bytes(content)

    if output_dir.exists():
        output_dir.rename(previous_dir)
    try:
        staging_dir.rename(output_dir)
        results = _validate_offline_dataset(output_dir, manifest)
    except Exception:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        if previous_dir.exists():
            previous_dir.rename(output_dir)
        raise

    if previous_dir.exists():
        shutil.rmtree(previous_dir)

    return results

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the pinned English V2 prompt-injection dataset")
    root = Path(__file__).parents[1]
    parser.add_argument("--manifest", type=Path, default=root / "data" / "v2_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "v2")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare_dataset(args.manifest, args.output_dir, offline=args.offline), indent=2))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"dataset preparation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
