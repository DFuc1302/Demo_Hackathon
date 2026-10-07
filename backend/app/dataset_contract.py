from __future__ import annotations

import hashlib
import json
from typing import Any


MULTILINGUAL_LANGUAGES = ("en", "sw", "ha", "bn")


def normalize_text_key(text: str) -> str:
    return " ".join(text.split()).casefold()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def serialize_jsonl(rows: list[dict[str, int | str]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")


def validate_split_rows(
    rows: list[dict[str, object]],
    split: str,
    expected: dict[str, Any] | None = None,
) -> list[dict[str, int | str]]:
    normalized: list[dict[str, int | str]] = []
    seen: set[str] = set()

    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or set(row.keys()) != {"text", "label"}:
            raise ValueError(f"{split} row {index} must contain exactly text and label columns")
        text_val = row.get("text")
        if not isinstance(text_val, str):
            raise ValueError(f"{split} row {index} has non-string text")
        text = text_val.strip()
        if not text:
            raise ValueError(f"{split} row {index} has empty text")
        label = row.get("label")
        if type(label) is not int or label not in (0, 1):
            raise ValueError(f"{split} row {index} has invalid label outside {{0, 1}}")

        norm_key = normalize_text_key(text)
        if norm_key in seen:
            raise ValueError(f"{split} contains a normalized duplicate at row {index}")
        seen.add(norm_key)
        normalized.append({"text": text, "label": label})

    if expected is not None:
        if "rows" in expected and len(normalized) != expected["rows"]:
            raise ValueError(f"{split} row/class counts do not match the immutable manifest")
        if "class_counts" in expected:
            counts = {str(lbl): sum(r["label"] == lbl for r in normalized) for lbl in (0, 1)}
            exp_counts = {str(k): v for k, v in expected["class_counts"].items()}
            if counts != exp_counts:
                raise ValueError(f"{split} row/class counts do not match the immutable manifest")

    return normalized


def check_cross_split_leakage(splits: dict[str, list[dict[str, Any]]]) -> None:
    normalized_keys = {
        split: {normalize_text_key(str(row["text"])) for row in rows}
        for split, rows in splits.items()
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if left in normalized_keys and right in normalized_keys:
            overlap = normalized_keys[left] & normalized_keys[right]
            if overlap:
                raise ValueError(f"normalized text leakage between {left} and {right}: {len(overlap)} rows")


def multilingual_split_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "source_groups": len({str(row["source_id"]) for row in rows}),
        "language_label_counts": {
            language: {
                str(label): sum(
                    row["language"] == language and row["label"] == label for row in rows
                )
                for label in (0, 1)
            }
            for language in MULTILINGUAL_LANGUAGES
        },
    }


def validate_multilingual_split_rows(
    rows: list[dict[str, object]],
    split: str,
    expected: dict[str, Any] | None = None,
) -> list[dict[str, int | str]]:
    normalized: list[dict[str, int | str]] = []
    groups: dict[str, list[dict[str, int | str]]] = {}
    seen_by_language = {language: set() for language in MULTILINGUAL_LANGUAGES}

    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or set(row) != {"text", "label", "language", "source_id"}:
            raise ValueError(
                f"{split} row {index} must contain exactly text, label, language, and source_id columns"
            )
        text_value = row["text"]
        if not isinstance(text_value, str):
            raise ValueError(f"{split} row {index} has non-string text")
        text = text_value.strip()
        if not text:
            raise ValueError(f"{split} row {index} has empty text")
        label = row["label"]
        if type(label) is not int or label not in (0, 1):
            raise ValueError(f"{split} row {index} has invalid label outside {{0, 1}}")
        language = row["language"]
        if not isinstance(language, str) or language not in MULTILINGUAL_LANGUAGES:
            raise ValueError(f"{split} row {index} has unsupported language")
        source_id = row["source_id"]
        if (
            not isinstance(source_id, str)
            or len(source_id) != 64
            or any(character not in "0123456789abcdef" for character in source_id)
        ):
            raise ValueError(f"{split} row {index} has invalid source_id")

        text_key = normalize_text_key(text)
        if text_key in seen_by_language[language]:
            raise ValueError(
                f"{split} contains a normalized duplicate for language {language} at row {index}"
            )
        seen_by_language[language].add(text_key)
        clean_row: dict[str, int | str] = {
            "text": text,
            "label": label,
            "language": language,
            "source_id": source_id,
        }
        normalized.append(clean_row)
        groups.setdefault(source_id, []).append(clean_row)

    for source_id, group in groups.items():
        languages = [row["language"] for row in group]
        if len(group) != len(MULTILINGUAL_LANGUAGES) or set(languages) != set(MULTILINGUAL_LANGUAGES):
            raise ValueError(
                f"{split} source group {source_id} must contain exactly one row for each language"
            )
        if len({row["label"] for row in group}) != 1:
            raise ValueError(f"{split} source group {source_id} has inconsistent labels")
        english_text = next(str(row["text"]) for row in group if row["language"] == "en")
        expected_source_id = hashlib.sha256(
            normalize_text_key(english_text).encode("utf-8")
        ).hexdigest()
        if source_id != expected_source_id:
            raise ValueError(
                f"{split} source group {source_id} source_id does not match normalized English text"
            )

    counts = multilingual_split_counts(normalized)
    cell_counts = {
        count
        for language_counts in counts["language_label_counts"].values()
        for count in language_counts.values()
    }
    if len(cell_counts) > 1:
        raise ValueError(f"{split} language/label cells are not balanced")

    if expected is not None:
        actual_contract = {
            key: counts[key] for key in ("rows", "source_groups", "language_label_counts")
        }
        expected_contract = {key: expected.get(key) for key in actual_contract}
        if isinstance(expected_contract["language_label_counts"], dict):
            expected_contract["language_label_counts"] = {
                language: {str(label): value for label, value in label_counts.items()}
                for language, label_counts in expected_contract["language_label_counts"].items()
            }
        if actual_contract != expected_contract:
            raise ValueError(f"{split} counts do not match the immutable manifest")

    language_order = {
        language: index for index, language in enumerate(MULTILINGUAL_LANGUAGES)
    }
    normalized.sort(
        key=lambda row: (
            str(row["source_id"]),
            language_order[str(row["language"])],
        )
    )
    return normalized


def check_multilingual_cross_split_leakage(
    splits: dict[str, list[dict[str, Any]]],
) -> None:
    split_names = list(splits)
    for left_index, left in enumerate(split_names):
        left_source_ids = {str(row["source_id"]) for row in splits[left]}
        left_text_keys = {
            (str(row["language"]), normalize_text_key(str(row["text"])))
            for row in splits[left]
        }
        for right in split_names[left_index + 1 :]:
            right_source_ids = {str(row["source_id"]) for row in splits[right]}
            source_overlap = left_source_ids & right_source_ids
            if source_overlap:
                raise ValueError(
                    f"source_id leakage between {left} and {right}: {len(source_overlap)} groups"
                )
            right_text_keys = {
                (str(row["language"]), normalize_text_key(str(row["text"])))
                for row in splits[right]
            }
            text_overlap = left_text_keys & right_text_keys
            if text_overlap:
                language = sorted(language for language, _ in text_overlap)[0]
                raise ValueError(
                    f"normalized {language} text leakage between {left} and {right}: "
                    f"{len(text_overlap)} rows"
                )


def decontaminate_multilingual_splits(
    splits: dict[str, list[dict[str, object]]],
) -> tuple[
    dict[str, list[dict[str, int | str]]],
    dict[str, dict[str, int | dict[str, int]]],
]:
    cleaned_splits: dict[str, list[dict[str, int | str]]] = {}
    removal_counts_by_split: dict[str, dict[str, int | dict[str, int]]] = {}

    for split, rows in splits.items():
        if not rows:
            raise ValueError(f"{split} split is empty")

        groups: dict[str, list[dict[str, object]]] = {}
        for index, row in enumerate(rows, start=1):
            if not isinstance(row, dict) or set(row) != {"text", "label", "language", "source_id"}:
                raise ValueError(
                    f"{split} row {index} must contain exactly text, label, language, and source_id columns"
                )
            source_id = row["source_id"]
            if not isinstance(source_id, str):
                raise ValueError(f"{split} row {index} has invalid source_id")
            text = row["text"]
            if not isinstance(text, str):
                raise ValueError(f"{split} row {index} has non-string text")
            language = row["language"]
            if not isinstance(language, str) or language not in MULTILINGUAL_LANGUAGES:
                raise ValueError(f"{split} row {index} has unsupported language")
            label = row["label"]
            if type(label) is not int or label not in (0, 1):
                raise ValueError(f"{split} row {index} has invalid label outside {{0, 1}}")
            groups.setdefault(source_id, []).append(row)

        for source_id, group in groups.items():
            languages = [row["language"] for row in group]
            if len(group) != len(MULTILINGUAL_LANGUAGES) or set(languages) != set(MULTILINGUAL_LANGUAGES):
                raise ValueError(
                    f"{split} source group {source_id} must contain exactly one row for each language"
                )
            if len({row["label"] for row in group}) != 1:
                raise ValueError(f"{split} source group {source_id} has inconsistent labels")

        input_groups = len(groups)
        empty_group_ids = {
            source_id
            for source_id, group in groups.items()
            if any(not str(row["text"]).strip() for row in group)
        }
        retained_after_empty = {
            source_id: group
            for source_id, group in groups.items()
            if source_id not in empty_group_ids
        }

        # Cross-split duplicate text collisions
        cross_split_text_keys = set()
        for other_split, other_rows in splits.items():
            if other_split == split:
                continue
            for r in other_rows:
                cross_split_text_keys.add((str(r["language"]), normalize_text_key(str(r["text"]))))

        collision_ids_by_language: dict[str, set[str]] = {
            language: set() for language in MULTILINGUAL_LANGUAGES
        }
        for language in MULTILINGUAL_LANGUAGES:
            source_ids_by_text: dict[str, set[str]] = {}
            for source_id, group in retained_after_empty.items():
                text = next(
                    str(row["text"]) for row in group if row["language"] == language
                )
                source_ids_by_text.setdefault(normalize_text_key(text), set()).add(source_id)
            for text_key, source_ids in source_ids_by_text.items():
                if len(source_ids) > 1 or (language, text_key) in cross_split_text_keys:
                    collision_ids_by_language[language].update(source_ids)

        collision_group_ids = set().union(*collision_ids_by_language.values())
        retained_after_collisions = {
            source_id: group
            for source_id, group in retained_after_empty.items()
            if source_id not in collision_group_ids
        }

        source_ids_by_label = {
            label: sorted(
                source_id
                for source_id, group in retained_after_collisions.items()
                if group[0]["label"] == label
            )
            for label in (0, 1)
        }
        label_difference = abs(
            len(source_ids_by_label[0]) - len(source_ids_by_label[1])
        )
        if len(source_ids_by_label[0]) > len(source_ids_by_label[1]):
            larger_label = 0
        else:
            larger_label = 1
        rebalance_group_ids = set(source_ids_by_label[larger_label][:label_difference])

        removed_group_ids = empty_group_ids | collision_group_ids | rebalance_group_ids
        removed_groups = len(removed_group_ids)
        if removed_groups * 10 > input_groups:
            raise ValueError(
                f"{split} decontamination removed more than 10% of source groups "
                f"({removed_groups}/{input_groups})"
            )

        cleaned_rows = [
            row
            for source_id, group in retained_after_collisions.items()
            if source_id not in rebalance_group_ids
            for row in group
        ]
        cleaned_splits[split] = validate_multilingual_split_rows(cleaned_rows, split)
        removal_counts_by_split[split] = {
            "input_groups": input_groups,
            "empty_groups": len(empty_group_ids),
            "collision_groups": len(collision_group_ids),
            "collision_groups_by_language": {
                language: len(collision_ids_by_language[language])
                for language in MULTILINGUAL_LANGUAGES
            },
            "rebalance_groups": len(rebalance_group_ids),
            "removed_groups": removed_groups,
        }

    check_multilingual_cross_split_leakage(cleaned_splits)
    return cleaned_splits, removal_counts_by_split
