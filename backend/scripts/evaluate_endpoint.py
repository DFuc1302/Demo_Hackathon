#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).parents[1]


def evaluate_sample(endpoint: str, text: str, timeout: float = 10.0) -> str:
    data = json.dumps({"prompt": text}).encode("utf-8")
    req = Request(endpoint, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return str(payload.get("label", "error"))
    except Exception:
        return "error"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark an inference endpoint against dataset splits.")
    parser.add_argument("--endpoint", required=True, help="URL of the analyze endpoint (e.g. http://127.0.0.1:8000/api/analyze)")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test", help="Dataset split to evaluate")
    parser.add_argument("--limit", type=int, default=None, help="Optional max rows to evaluate")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of concurrent workers")
    args = parser.parse_args()

    split_file = PROJECT_ROOT / "data" / "v2" / f"{args.split}.jsonl"
    if not split_file.is_file():
        print(f"Error: dataset file not found: {split_file}", file=sys.stderr)
        sys.exit(1)

    rows = []
    with split_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if args.limit:
        rows = rows[:args.limit]

    total = len(rows)
    print(f"Loaded {total} samples from {split_file.name}. Running evaluation against {args.endpoint} (concurrency={args.concurrency})...")

    start_time = time.monotonic()
    predictions = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(evaluate_sample, args.endpoint, r["text"]) for r in rows]
        for f in futures:
            predictions.append(f.result())
    elapsed = time.monotonic() - start_time

    # Calculate metrics
    # label mapping: 0 -> benign, 1 -> jailbreak
    tp = tn = fp = fn = errors = 0
    for r, pred in zip(rows, predictions):
        true_label = "jailbreak" if r["label"] == 1 else "benign"
        if pred == "error":
            errors += 1
        elif true_label == "jailbreak" and pred == "jailbreak":
            tp += 1
        elif true_label == "benign" and pred == "benign":
            tn += 1
        elif true_label == "benign" and pred == "jailbreak":
            fp += 1
        elif true_label == "jailbreak" and pred == "benign":
            fn += 1

    valid_total = tp + tn + fp + fn
    accuracy = (tp + tn) / valid_total if valid_total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print("\n" + "=" * 40)
    print(f"Evaluation Results for '{args.split}' split")
    print("=" * 40)
    print(f"Total Rows:     {total}")
    print(f"Elapsed Time:   {elapsed:.2f}s ({total / elapsed:.1f} req/s)")
    print(f"Errors:         {errors}")
    print(f"Accuracy:       {accuracy:.4f}")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1 Score:       {f1:.4f}")
    print(f"Confusion Matrix: [[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]]")
    print("=" * 40)


if __name__ == "__main__":
    main()
