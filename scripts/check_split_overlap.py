"""Check for overlap between the splits of the Igbo 1.0 dataset.

Downloads the train, validation and test files of
coded-by-49/langbridge_ai_v3_igbo_datasets from Hugging Face and counts how
many rows of each evaluation split also appear in the train split.

A row "overlaps" when its exact (English, Igbo) pair, after stripping
leading and trailing whitespace, is present in the train split.

Writes results/v3_split_overlap.csv and prints a short summary.

Usage:
    python scripts/check_split_overlap.py
"""

from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path

REPO = "coded-by-49/langbridge_ai_v3_igbo_datasets"
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
FILES = {
    "train": "igbo_eng_lv1_train.jsonl",
    "validation (full)": "igbo_eng_lv1_validation.jsonl",
    "validation (256-token filtered, used in training)": "igbo_eng_lv1_validation_256_filtered.jsonl",
    "test": "igbo_eng_lv1_test_.jsonl",
}
OUT = Path(__file__).resolve().parents[1] / "results" / "v3_split_overlap.csv"


def load_pairs(filename: str) -> list[tuple[str, str]]:
    with urllib.request.urlopen(BASE_URL + filename) as response:
        lines = response.read().decode("utf-8").splitlines()
    pairs = []
    for line in lines:
        if not line.strip():
            continue
        t = json.loads(line)["translation"]
        pairs.append((t["eng_Latn"].strip(), t["ibo_Latn"].strip()))
    return pairs


def main() -> None:
    splits = {name: load_pairs(fname) for name, fname in FILES.items()}
    train_pairs = set(splits["train"])
    train_english = {en for en, _ in splits["train"]}

    rows = []
    for name, pairs in splits.items():
        if name == "train":
            continue
        pair_hits = sum(p in train_pairs for p in pairs)
        english_hits = sum(en in train_english for en, _ in pairs)
        rows.append(
            {
                "split": name,
                "file": FILES[name],
                "rows": len(pairs),
                "rows_with_exact_pair_in_train": pair_hits,
                "rows_with_same_english_in_train": english_hits,
                "rows_not_in_train": len(pairs) - pair_hits,
                "pct_exact_pair_in_train": f"{100 * pair_hits / len(pairs):.2f}",
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"train rows: {len(splits['train'])}")
    for r in rows:
        print(
            f"{r['split']}: {r['rows_with_exact_pair_in_train']} of {r['rows']} rows "
            f"({r['pct_exact_pair_in_train']}%) have their exact pair in train"
        )
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
