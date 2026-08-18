#!/usr/bin/env python3
"""Summarize everything under results/ into one table.

`script/test_all.sh` calls this automatically when it finishes; it can also be run on
its own:

    python utils/summarize.py                # everything
    python utils/summarize.py level2 level5  # only these levels

Models use different metric-key prefixes (`ufno/rmse`, `fno_det/rmse`, `test/rmse`),
so lookups match on the key **suffix** instead of a hardcoded prefix.
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

# The six reported metrics: (header, key suffix, higher-is-better)
COLS = [
    ("RMSE v", "rmse", False),
    ("MAE v", "mean_absolute_error", False),
    ("R2 ^", "r2", True),
    ("MaxAE v", "max_absolute_error", False),
    ("T_max err v", "max_temperature_error", False),
    ("Top-MAE v", "topk50_temperature_difference", False),
]
ORDER = ["FNO", "UFNO", "SAUFNO", "UNet", "DeepONet",
         "ThermFM-T", "ThermFM-B", "ThermFM-L"]


def pick(d, suffix):
    """Look up by key suffix -- prefixes differ per model, and `r2` must not match
    `r2_per_sample`."""
    for k, v in d.items():
        if k.rsplit("/", 1)[-1] == suffix and isinstance(v, (int, float)):
            return float(v)
    return None


def main():
    wanted = sys.argv[1:]
    rows = {}
    for p in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
        name = os.path.basename(p)[:-5]
        if "_" not in name:
            continue
        level, model = name.split("_", 1)
        if wanted and level not in wanted:
            continue
        with open(p, encoding="utf-8") as f:
            rows.setdefault(level, {})[model] = json.load(f)

    if not rows:
        sys.exit(f"no results under {RESULTS}. Run script/test_all.sh first")

    for level in sorted(rows):
        print(f"\n## {level}")
        w = max(len(m) for m in rows[level]) + 2
        print(f"{'model':<{w}}" + "".join(f"{c:>13}" for c, _, _ in COLS))
        print("-" * (w + 13 * len(COLS)))
        models = [m for m in ORDER if m in rows[level]]
        models += [m for m in sorted(rows[level]) if m not in ORDER]
        best = {}
        for _, key, hi in COLS:
            vals = [pick(rows[level][m], key) for m in models]
            vals = [v for v in vals if v is not None]
            if vals:
                best[key] = max(vals) if hi else min(vals)
        for m in models:
            line = f"{m:<{w}}"
            for _, key, _ in COLS:
                v = pick(rows[level][m], key)
                if v is None:
                    line += f"{'-':>13}"
                else:
                    mark = "*" if v == best.get(key) else " "
                    line += f"{v:>12.4f}{mark}"
            print(line)
    print("\n* = best in column")


if __name__ == "__main__":
    main()
