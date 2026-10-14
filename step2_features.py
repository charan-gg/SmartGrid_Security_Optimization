"""
STEP 2 - align + build modality views.

Purpose: turn the 128 flat columns into the four layer-aligned views the thesis
argues about, and record exactly which column went where. That mapping is a
claim about the architecture, so it gets written to disk and cited, not left
implicit in a regex.

Run:  python step2_features.py
"""
import json

import numpy as np
import pandas as pd

from src import config as C
from src.loading import load_dataset
from src.modalities import classify_column, STATUS_FLAG_IS_CONTROL_LAYER


def main():
    print("=" * 78)
    print("STEP 2 - MODALITY VIEWS")
    print("=" * 78)

    df, feature_cols, views = load_dataset()

    # ------------------------------------------------ the mapping, written out
    mapping = pd.DataFrame({
        "column": feature_cols,
        "modality": [classify_column(c) for c in feature_cols],
    })
    mapping["purdue_level"] = mapping["modality"].map({
        "pmu": "L1 physical",
        "relay": "L2 control",
        "control": "L2/L3 operator",
        "snort": "L3.5 network",
    })
    mapping.to_csv(C.OUTPUT_DIR / "step2_modality_map.csv", index=False)

    unknown = mapping.loc[mapping["modality"] == "unknown", "column"].tolist()
    assert not unknown, f"unmapped columns, fix modalities.py: {unknown}"

    print("\n[1] VIEW SIZES")
    for name, cols in views.items():
        print(f"  {name:<12} {len(cols):>4} columns")
    print(f"\n  R*:S status flag treated as "
          f"{'control layer' if STATUS_FLAG_IS_CONTROL_LAYER else 'PMU'} "
          "(modalities.STATUS_FLAG_IS_CONTROL_LAYER)")

    # ------------------------------------- can each view even separate classes?
    # Before spending a single model fit on E1, check whether a view varies at
    # all with the label. A view whose features are identical across classes
    # cannot produce a classifier, and knowing that now saves misreading E1.
    print("\n[2] PER-VIEW SEPARABILITY PRE-CHECK")
    print("  For each view: does ANY column differ in mean across the three")
    print("  classes by more than its own pooled standard deviation?")
    rows = []
    for name, cols in views.items():
        if not cols:
            continue
        block = df[cols].to_numpy(dtype="float64")
        block = np.nan_to_num(block, nan=0.0)
        best = 0.0
        for j in range(block.shape[1]):
            col = block[:, j]
            means = [col[(df[C.TARGET] == c).to_numpy()].mean() for c in C.CLASS_ORDER]
            spread = max(means) - min(means)
            sd = col.std()
            if sd > 0:
                best = max(best, spread / sd)
        rows.append({"view": name, "n_columns": len(cols), "max_effect_size": best})
        verdict = "usable" if best > 0.2 else "WEAK - expect near-chance in E1"
        print(f"  {name:<12} max standardised class gap = {best:7.3f}   {verdict}")
    pd.DataFrame(rows).to_csv(C.OUTPUT_DIR / "step2_separability.csv", index=False)

    # -------------------------------------------- engineered feature inventory
    print("\n[3] ENGINEERED FEATURES")
    eng = [c for c in feature_cols if c.endswith("__was_inf")]
    print(f"  open-circuit indicators: {len(eng)}  {eng}")
    print("  rationale: Z = V/I -> Inf means no current, i.e. the line is open.")
    print("  That is a physical event, not a parsing error, so it is preserved")
    print("  as an explicit boolean before the Inf itself is turned into NaN.")

    summary = {
        "view_sizes": {k: len(v) for k, v in views.items()},
        "engineered": eng,
        "unknown_columns": unknown,
    }
    with open(C.OUTPUT_DIR / "step2_views.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {C.OUTPUT_DIR/'step2_modality_map.csv'}")
    print(f"wrote {C.OUTPUT_DIR/'step2_separability.csv'}")
    print(f"wrote {C.OUTPUT_DIR/'step2_views.json'}")


if __name__ == "__main__":
    main()
