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
import math
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


def model_family(model):
    """Treat Therm-FM scale variants as one model family for leaderboard rank."""
    return "ThermFM" if model.startswith("ThermFM-") else model


def rank_marks(model_values, higher_is_better):
    """Mark the leading two method families, using each family's best variant."""
    family_best = {}
    for model, value in model_values:
        if value is None or not math.isfinite(value):
            continue
        family = model_family(model)
        incumbent = family_best.get(family)
        if incumbent is None or ((value > incumbent) if higher_is_better
                                 else (value < incumbent)):
            family_best[family] = value

    ranked = sorted(set(family_best.values()), reverse=higher_is_better)
    marks = {}
    for model, value in model_values:
        if value is None or not math.isfinite(value):
            continue
        if value != family_best[model_family(model)]:
            continue
        if ranked and value == ranked[0]:
            marks[model] = "*"
        elif len(ranked) > 1 and value == ranked[1]:
            marks[model] = "+"
    return marks


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
        marks = {}
        for _, key, hi in COLS:
            model_values = [(m, pick(rows[level][m], key)) for m in models]
            marks[key] = rank_marks(model_values, hi)
        for m in models:
            line = f"{m:<{w}}"
            for _, key, _ in COLS:
                v = pick(rows[level][m], key)
                if v is None:
                    line += f"{'-':>13}"
                else:
                    mark = marks.get(key, {}).get(m, " ")
                    line += f"{v:>12.4f}{mark}"
            print(line)
    print("\n* = best method family; + = second-best method family "
          "(ThermFM-T/B/L count once)")


if __name__ == "__main__":
    main()
