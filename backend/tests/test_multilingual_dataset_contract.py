from __future__ import annotations

import hashlib

import pytest

from app.dataset_contract import (check_multilingual_cross_split_leakage, decontaminate_multilingual_splits, multilingual_split_counts, normalize_text_key, validate_multilingual_split_rows)

LANGUAGES = ("en", "sw", "ha", "bn")


def _source_id(english_text: str) -> str:
    return hashlib.sha256(normalize_text_key(english_text).encode("utf-8")).hexdigest()


def _group(english_text: str, label: int, suffix: str = "") -> list[dict[str, object]]:
    source_id = _source_id(english_text)
    translations = {
        "en": english_text,
        "sw": f"Kiswahili {english_text}{suffix}",
        "ha": f"Hausa {english_text}{suffix}",
        "bn": f"বাংলা {english_text}{suffix}",
    }
    return [
        {"text": translations[language], "label": label, "language": language, "source_id": source_id}
        for language in LANGUAGES
    ]


def _balanced_rows(prefix: str = "") -> list[dict[str, object]]:
    return _group(f"{prefix} benign request", 0) + _group(f"{prefix} attack request", 1)


def test_multilingual_validation_preserves_bengali_and_returns_canonical_order():
    rows = list(reversed(_balanced_rows()))

    validated = validate_multilingual_split_rows(rows, "train")

    expected_order = [
        (source_id, language)
        for source_id in sorted({_source_id(" benign request"), _source_id(" attack request")})
        for language in LANGUAGES
    ]
    assert [(row["source_id"], row["language"]) for row in validated] == expected_order
    assert {row["text"] for row in validated if row["language"] == "bn"} == {
        "বাংলা  benign request",
        "বাংলা  attack request",
    }


def test_multilingual_validation_requires_exact_four_fields():
    rows = _balanced_rows()
    rows[0]["extra"] = "not allowed"

    with pytest.raises(ValueError, match="exactly text, label, language, and source_id"):
        validate_multilingual_split_rows(rows, "train")


def test_multilingual_validation_requires_complete_four_language_groups():
    rows = _balanced_rows()
    rows.pop()

    with pytest.raises(ValueError, match="exactly one row for each language"):
        validate_multilingual_split_rows(rows, "train")


def test_multilingual_validation_requires_same_label_within_group():
    rows = _balanced_rows()
    rows[1]["label"] = 1

    with pytest.raises(ValueError, match="inconsistent labels"):
        validate_multilingual_split_rows(rows, "train")


def test_multilingual_validation_requires_source_id_from_normalized_english():
    rows = _balanced_rows()
    for row in rows[:4]:
        row["source_id"] = "0" * 64

    with pytest.raises(ValueError, match="source_id does not match normalized English text"):
        validate_multilingual_split_rows(rows, "train")


def test_multilingual_validation_rejects_same_language_normalized_duplicate():
    rows = _balanced_rows()
    swahili = [row for row in rows if row["language"] == "sw"]
    swahili[1]["text"] = "  KISWAHILI   BENIGN REQUEST  "

    with pytest.raises(ValueError, match="normalized duplicate for language sw"):
        validate_multilingual_split_rows(rows, "train")


def test_multilingual_validation_requires_balanced_language_label_cells():
    rows = _balanced_rows() + _group("second benign request", 0)

    with pytest.raises(ValueError, match="language/label cells are not balanced"):
        validate_multilingual_split_rows(rows, "train")


def test_multilingual_counts_and_expected_manifest_contract():
    rows = _balanced_rows()
    expected_counts = {
        "rows": 8,
        "source_groups": 2,
        "language_label_counts": {
            language: {"0": 1, "1": 1} for language in LANGUAGES
        },
    }

    assert multilingual_split_counts(rows) == expected_counts
    assert validate_multilingual_split_rows(rows, "validation", expected_counts)

    wrong = {**expected_counts, "source_groups": 3}
    with pytest.raises(ValueError, match="counts do not match the immutable manifest"):
        validate_multilingual_split_rows(rows, "validation", wrong)


def test_multilingual_cross_split_rejects_source_id_overlap():
    train = _balanced_rows("train")
    validation = _balanced_rows("validation")
    validation.extend(_group("train benign request", 0))

    with pytest.raises(ValueError, match="source_id leakage between train and validation"):
        check_multilingual_cross_split_leakage({"train": train, "validation": validation})


def test_multilingual_cross_split_rejects_language_text_overlap():
    train = _balanced_rows("train")
    validation = _balanced_rows("validation")
    train_sw = next(row for row in train if row["language"] == "sw")
    validation_sw = next(row for row in validation if row["language"] == "sw")
    validation_sw["text"] = f"  {str(train_sw['text']).upper()}  "

    with pytest.raises(ValueError, match="normalized sw text leakage between train and validation"):
        check_multilingual_cross_split_leakage({"train": train, "validation": validation})


def test_multilingual_cross_split_accepts_disjoint_splits():
    check_multilingual_cross_split_leakage(
        {
            "train": _balanced_rows("train"),
            "validation": _balanced_rows("validation"),
            "test": _balanced_rows("test"),
        }
    )


def _many_balanced_groups(groups_per_label: int) -> list[dict[str, object]]:
    return [
        row
        for label in (0, 1)
        for index in range(groups_per_label)
        for row in _group(f"source {label} {index}", label)
    ]


def test_decontamination_drops_empty_group_rebalances_and_accounts_exactly():
    rows = _many_balanced_groups(20)
    empty_source_id = str(rows[0]["source_id"])
    next(row for row in rows if row["source_id"] == empty_source_id and row["language"] == "bn")["text"] = "  "

    cleaned, removals = decontaminate_multilingual_splits({"train": rows})

    counts = removals["train"]
    assert counts == {
        "input_groups": 40,
        "empty_groups": 1,
        "collision_groups": 0,
        "collision_groups_by_language": {language: 0 for language in LANGUAGES},
        "rebalance_groups": 1,
        "removed_groups": 2,
    }
    assert empty_source_id not in {row["source_id"] for row in cleaned["train"]}
    assert multilingual_split_counts(cleaned["train"])["language_label_counts"] == {
        language: {"0": 19, "1": 19} for language in LANGUAGES
    }


def test_decontamination_drops_all_groups_in_collisions_and_tracks_language_overlap():
    rows = _many_balanced_groups(20)
    label_zero_ids = sorted(
        {str(row["source_id"]) for row in rows if row["label"] == 0}
    )
    label_one_ids = sorted(
        {str(row["source_id"]) for row in rows if row["label"] == 1}
    )
    colliding_ids = {label_zero_ids[0], label_one_ids[0]}
    for language in ("sw", "bn"):
        for row in rows:
            if row["source_id"] in colliding_ids and row["language"] == language:
                row["text"] = f"shared {language} collision"

    cleaned, removals = decontaminate_multilingual_splits({"train": rows})

    assert removals["train"] == {
        "input_groups": 40,
        "empty_groups": 0,
        "collision_groups": 2,
        "collision_groups_by_language": {"en": 0, "sw": 2, "ha": 0, "bn": 2},
        "rebalance_groups": 0,
        "removed_groups": 2,
    }
    assert not colliding_ids & {str(row["source_id"]) for row in cleaned["train"]}


def test_decontamination_rebalances_larger_label_by_ascending_source_id():
    rows = _many_balanced_groups(20) + _group("one extra benign group", 0)
    label_zero_ids = sorted(
        {str(row["source_id"]) for row in rows if row["label"] == 0}
    )

    cleaned, removals = decontaminate_multilingual_splits({"train": rows})

    assert removals["train"]["rebalance_groups"] == 1
    assert label_zero_ids[0] not in {str(row["source_id"]) for row in cleaned["train"]}
    assert multilingual_split_counts(cleaned["train"])["source_groups"] == 40


def test_decontamination_rejects_more_than_ten_percent_removed_groups():
    rows = _many_balanced_groups(5)
    empty_source_id = str(rows[0]["source_id"])
    next(row for row in rows if row["source_id"] == empty_source_id and row["language"] == "sw")["text"] = ""

    with pytest.raises(ValueError, match="more than 10%"):
        decontaminate_multilingual_splits({"train": rows})


def test_decontamination_rejects_completely_empty_split():
    with pytest.raises(ValueError, match="train split is empty"):
        decontaminate_multilingual_splits({"train": []})


def test_decontamination_is_deterministic_across_input_order():
    rows = _many_balanced_groups(20) + _group("one extra attack group", 1)

    forward = decontaminate_multilingual_splits({"train": rows})
    reverse = decontaminate_multilingual_splits({"train": list(reversed(rows))})

    assert forward == reverse
