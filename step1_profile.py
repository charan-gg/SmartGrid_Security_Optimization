"""
STEP 1 - acquire + profile (Phase 0 verification).

Purpose: replace every "recalled from literature" claim in the project brief
with a measurement taken from the actual files, and fail loudly where the data
does not match what the plan assumed. The output of this script IS Appendix A.

Run:  python step1_profile.py
"""
import json

import numpy as np
import pandas as pd

from src import config as C
from src.loading import load_dataset, META_COLUMNS
from src.modalities import classify_column
from src.splits import describe


def main():
    print("=" * 78)
    print("STEP 1 - DATASET PROFILE (Phase 0 verification)")
    print("=" * 78)

    # load_dataset() asserts the three structural facts F1/F2/F3 on the way in;
    # if any of them stopped holding, this call raises rather than returning
    # quietly wrong data.
    df, feature_cols, views = load_dataset(use_cache=False)
    report = {}

    # ---------------------------------------------------------------- shape
    print("\n[1] SHAPE")
    report["n_rows"] = len(df)
    report["n_features"] = len(feature_cols)
    print(f"  rows          : {len(df):,}")
    print(f"  feature cols  : {len(feature_cols)}  (128 source + engineered)")
    print(f"  meta cols     : {META_COLUMNS}")

    # ------------------------------------------------------------- labelling
    print("\n[2] LABELS")
    triple_counts = df["y_triple"].value_counts().to_dict()
    report["triple_counts"] = {str(k): int(v) for k, v in triple_counts.items()}
    for k, v in triple_counts.items():
        print(f"  {k:<10} {v:>7,}  ({v / len(df):6.2%})")
    print(f"  scenarios     : {df['scenario_id'].nunique()} distinct ids")
    # Majority-class baseline. Every accuracy figure must be compared to this,
    # which is exactly why the project reports macro-F1 instead.
    majority = max(triple_counts.values()) / len(df)
    report["majority_class_accuracy"] = float(majority)
    print(f"  ** always-predict-majority accuracy = {majority:.4f} **")
    print("     -> any 'accuracy' below this is worse than a constant. "
          "Report macro-F1 and per-class recall only.")

    # ------------------------------------------------- event/group structure
    print("\n[3] EVENT STRUCTURE  (the basis of every honest split)")
    struct = describe(df)
    report["structure"] = struct
    print(f"  event instances (file x scenario) : {struct['n_events']}")
    print(f"  rows per event  min/median/max    : "
          f"{struct['rows_per_event_min']} / {struct['rows_per_event_median']} / "
          f"{struct['rows_per_event_max']}")
    print(f"  distinct scenarios                : {struct['n_scenarios']}")
    print("  -> row order inside an event is TIME order; there is no timestamp")
    print("     column, so E2 uses position-in-event as relative time.")

    # ----------------------------------------------------- per-column health
    print("\n[4] PER-COLUMN HEALTH  (cardinality, missingness, constancy)")
    rows = []
    for c in feature_cols:
        s = df[c]
        rows.append({
            "column": c,
            "modality": classify_column(c),
            "n_unique": int(s.nunique(dropna=True)),
            "n_nan": int(s.isna().sum()),
            "pct_nan": float(s.isna().mean()),
            "std": float(np.nanstd(s.to_numpy(dtype="float64"))),
            # For the binary flag columns, this is the one number that matters:
            # how many rows in the whole dataset ever see this flag set.
            "n_nonzero": int((s.fillna(0) != 0).sum()),
        })
    health = pd.DataFrame(rows)
    health.to_csv(C.OUTPUT_DIR / "step1_column_health.csv", index=False)

    constant = health.loc[health["n_unique"] <= 1, "column"].tolist()
    report["constant_columns"] = constant
    print(f"  constant columns (zero information): {len(constant)}  {constant}")

    # ----------------------------------- THE FINDING THAT RESHAPES E1
    print("\n[5] MODALITY SIGNAL DENSITY  <-- read this before trusting E1")
    print("  The thesis assumes four comparable evidence streams. Measure how")
    print("  often each non-PMU flag is actually set across the whole dataset:")
    dense = []
    for mod in ("pmu", "relay", "snort", "control"):
        cols = views[mod]
        if not cols:
            continue
        nz = health.loc[health["modality"] == mod, "n_nonzero"].sum()
        ever_set = int((df[cols].fillna(0) != 0).any(axis=1).sum())
        frac = ever_set / len(df)
        dense.append({"modality": mod, "n_columns": len(cols),
                      "rows_with_any_signal": ever_set, "fraction": frac})
        flag = ""
        if mod != "pmu" and frac < 0.01:
            flag = "  <-- DEGENERATE: cannot support a row-level classifier"
        print(f"  {mod:<9} cols={len(cols):>4}  rows with any non-zero: "
              f"{ever_set:>7,} ({frac:7.4%}){flag}")
    report["modality_density"] = dense
    pd.DataFrame(dense).to_csv(C.OUTPUT_DIR / "step1_modality_density.csv", index=False)

    # -------------------------------------------------------- duplicate rows
    print("\n[6] DUPLICATES")
    dup = int(df[feature_cols].duplicated().sum())
    report["duplicate_feature_rows"] = dup
    print(f"  fully duplicated feature rows: {dup}")
    print("  -> too few to matter, but they must not straddle a split; the")
    print("     event-grouped splitter already prevents that.")

    # ------------------------------------------------------------- config echo
    print("\n[7] CHOICES IN EFFECT  (copy into Appendix A verbatim)")
    choices = {
        "target": C.TARGET,
        "inf_policy": C.INF_POLICY,
        "add_inf_indicators": C.ADD_INF_INDICATORS,
        "group_key": C.GROUP_KEY,
        "n_splits": C.N_SPLITS,
        "random_seed": C.RANDOM_SEED,
        "sample_rate_hz_CONFIRM": C.SAMPLE_RATE_HZ,
        "benign_fraction_CONFIRM": C.BENIGN_FRACTION_REAL,
        "model_params": C.MODEL_PARAMS,
    }
    report["choices"] = choices
    for k, v in choices.items():
        print(f"  {k:<28} {v}")

    with open(C.OUTPUT_DIR / "step1_profile.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\nwrote {C.OUTPUT_DIR/'step1_profile.json'}")
    print(f"wrote {C.OUTPUT_DIR/'step1_column_health.csv'}")
    print(f"wrote {C.OUTPUT_DIR/'step1_modality_density.csv'}")


if __name__ == "__main__":
    main()
