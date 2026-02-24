#!/usr/bin/env python3
"""
Evaluate how many inference errors the postprocessor can recover.

For each test case: run validate_field() on the prediction, then check
whether the corrected value matches ground truth.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from postprocess.rules import validate_field, load_validation_rules, try_parse_numeric

TEST_CASES = [
    ("22854-01_no_load_vars.png",           "139,9",  "1,33"),
    ("300049-05_ratio_lv1_1u2u_col2.png",   "24.98",  "77,97"),
    ("300050-02_hv_res_tap5_1w1u.png",      "8118",   ".811"),
    ("300049-01_hv_res_tap1_1w1u.png",      "964",    ",4"),
    ("300049-01_hv_res_tap7_1u1v.png",      "1,7645", "1,7764"),
    ("300049-20_hv_winding_temp_c.png",     "27",     "23"),
    ("300049-01_no_load_100_io_a.png",      "350",    "400"),
    ("300050-01_hv_res_tap1_1w1u.png",      "014",    ".4"),
    ("300050-02_hv_current_a.png",          ".37",    ".33"),
    ("300049-01_load_loss_lv1_uk_v.png",    "405",    "400"),
]


def field_name_from_filename(fname: str) -> str:
    """Strip the report-number prefix (e.g. '300049-01_') and .png suffix."""
    stem = Path(fname).stem
    parts = stem.split("_", 1)
    # prefix is digits+dash e.g. "300049-01"
    if len(parts) == 2 and parts[0].replace("-", "").isdigit():
        return parts[1]
    return stem


def numeric_match(a: str, b: str) -> bool:
    """True if both parse to the same float (comma-normalised)."""
    na = try_parse_numeric(a)
    nb = try_parse_numeric(b)
    if na is None or nb is None:
        return False
    return abs(na - nb) < 1e-9


def main():
    rules = load_validation_rules()

    raw_correct = 0
    post_correct = 0
    flagged_count = 0

    col = dict(fname=45, gt=10, pred=10, corr=12, flagged=9, match=11)
    header = (
        f"{'Image':<{col['fname']}} {'GT':>{col['gt']}} {'Pred':>{col['pred']}} "
        f"{'Corrected':>{col['corr']}} {'Flagged':>{col['flagged']}} {'PostMatch':>{col['match']}}"
    )
    sep = "-" * (sum(col.values()) + len(col) - 1)

    print(header)
    print(sep)

    for fname, gt, pred in TEST_CASES:
        field = field_name_from_filename(fname)
        result = validate_field(field, pred, rules)

        corrected = result["corrected"]
        was_flagged = result["was_corrected"] or not result["in_range"]
        flagged_str = "YES" if was_flagged else "-"

        raw_ok = numeric_match(pred, gt)
        post_ok = numeric_match(corrected, gt)
        match_str = "PASS" if post_ok else "FAIL"

        if raw_ok:
            raw_correct += 1
        if post_ok:
            post_correct += 1
        if was_flagged:
            flagged_count += 1

        reason = f"  <- {result['correction_reason']}" if result["was_corrected"] else ""

        print(
            f"{fname:<{col['fname']}} {gt:>{col['gt']}} {pred:>{col['pred']}} "
            f"{corrected:>{col['corr']}} {flagged_str:>{col['flagged']}} {match_str:>{col['match']}}"
            f"{reason}"
        )

    n = len(TEST_CASES)
    print(sep)
    print(f"\nRaw accuracy (no postprocess):   {raw_correct}/{n}  ({100*raw_correct/n:.0f}%)")
    print(f"Post-processed accuracy:         {post_correct}/{n}  ({100*post_correct/n:.0f}%)")
    print(f"Cases flagged by range check:    {flagged_count}/{n}")

    print("\n--- Per-case details (field / range / predicted numeric value) ---")
    for fname, gt, pred in TEST_CASES:
        field = field_name_from_filename(fname)
        field_cfg = rules.get(field, {})
        v = field_cfg.get("validation", {})
        lo, hi = v.get("min", "?"), v.get("max", "?")
        pred_num = try_parse_numeric(pred)
        in_range = "in range" if (
            pred_num is not None
            and isinstance(lo, (int, float))
            and isinstance(hi, (int, float))
            and lo <= pred_num <= hi
        ) else "OUT OF RANGE"
        print(f"  {field:<35} range [{lo}, {hi}]  pred={pred_num}  -> {in_range}")


if __name__ == "__main__":
    main()
