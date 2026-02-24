#!/usr/bin/env python3
"""
Targeted labeling session: crops where leading-zero confusion is likely.

Fields included:
  hv_current_a       -- values like 0.09, 0.37 (small amperes)
  hv_res_tap1_1u1v   -- values like 014, 025, 0.91 (small ohm taps)
  hv_res_tap1_1v1w
  hv_res_tap1_1w1u
  lv_current_a       -- values like 0.32, 0.33 (small amperes)

Run:
  source .venv/Scripts/activate
  python label_leading_zeros.py
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.labeling_tool import run_labeling, LABELS_CSV

CROPS_DIR = Path(__file__).resolve().parent / "data" / "crops"

TARGET_FIELDS = [
    "hv_current_a",
    "hv_res_tap1_1u1v",
    "hv_res_tap1_1v1w",
    "hv_res_tap1_1w1u",
    "lv_current_a",
]


def main():
    labeled = set()
    if LABELS_CSV.exists():
        with open(LABELS_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                labeled.add(Path(row["image_path"]).name)

    file_list = []
    for field in TARGET_FIELDS:
        for p in sorted(CROPS_DIR.glob(f"*_{field}.png")):
            if p.name not in labeled:
                file_list.append(p)

    if not file_list:
        print("All targeted crops are already labeled.")
        return

    print(f"Targeted labeling: {len(file_list)} unlabeled crops")
    print(f"Fields: {', '.join(TARGET_FIELDS)}\n")

    run_labeling(file_list=file_list, labels_csv=LABELS_CSV)


if __name__ == "__main__":
    main()
