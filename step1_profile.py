"""
STEP 1 - profile the data and audit how independent Train.csv and Test.csv are.

Replaces every "recalled" claim about the dataset with a measurement, and fails loudly
(src/data.py asserts T1-T4) if a different download does not match what the pipeline
assumes. Writes outputs/step1_profile.json, which step 5 reads (merging units per row,
message rate), and outputs/step1_column_health.csv.

Run:  python step1_profile.py
"""
import json

import numpy as np
import pandas as pd

from src import config as C
from src.data import META_COLUMNS, audit, load_published, rates
from src.modalities import classify_column, n_units
from src.splits import describe


def main():
    print("=" * 78)
    print("STEP 1 - DATASET PROFILE")
    print("=" * 78)
    train, test, feature_cols = load_published()
    report = {}

    print("\n[1] SHAPE")
    report["n_rows_train"], report["n_rows_test"] = len(train), len(test)
    report["n_features"] = len(feature_cols)
    print(f"  Train {len(train):,} rows, Test {len(test):,} rows, {len(feature_cols)} features "
          f"(+ label and the `time` clock, which is metadata, never a feature)")

    print("\n[2] LABELS (Train.csv)")
    counts = train[C.TARGET].value_counts().reindex(C.CLASS_ORDER)
    report["class_counts"] = {k: int(v) for k, v in counts.items()}
    for k, v in counts.items():
        print(f"  {k:<11} {v:>5,}  ({v / len(train):6.2%})   family: {C.FAMILY[k]}")
    maj = counts.max() / len(train)
    report["majority_class_accuracy"] = float(maj)
    print(f"  ** always-predict-Normal accuracy = {maj:.4f} **  -> nothing here reports bare accuracy;")
    print("     macro-F1 and per-class recall are used (the three attacks together are "
          f"{counts[C.ATTACK_CLASSES].sum() / len(train):.0%} of the rows).")

    print("\n[3] THE CAPTURE")
    r = rates(train)
    report["rates"] = r
    report["n_units"] = n_units(feature_cols)
    print(f"  duration {r['duration_s']:.0f} s, {r['row_rate_hz']:.2f} messages/s on average "
          f"(median gap {r['median_gap_ms']:.0f} ms, p95 gap {r['p95_gap_ms']:.0f} ms)")
    print(f"  non-attack (Normal + Fault) rate {r['negative_rate_hz']:.2f}/s")
    print(f"  one row spans {report['n_units']} merging units -> per-bay figures divide by "
          f"{report['n_units']} (config.MEASURED_UNIT_IEDS overrides)")
    d = describe(train)
    report["blocks"] = d
    print(f"  {d['n_blocks']} time blocks of {d['rows_per_block_min']}-{d['rows_per_block_max']} rows "
          f"(median {d['rows_per_block_median']}); {d['blocks_containing_all_classes']} contain all "
          f"{len(C.CLASS_ORDER)} classes")

    print("\n[4] FEATURE HEALTH")
    rows = []
    for c in feature_cols:
        s = train[c]
        rows.append(dict(column=c, layer=classify_column(c), n_unique=int(s.nunique()),
                         nonzero_rows=int((s != 0).sum()), min=float(s.min()), max=float(s.max()),
                         constant=bool(s.nunique() == 1)))
    health = pd.DataFrame(rows)
    health.to_csv(C.OUTPUT_DIR / "step1_column_health.csv", index=False)
    const = health.loc[health["constant"], "column"].tolist()
    unknown = health.loc[health["layer"] == "unknown", "column"].tolist()
    report["constant_columns"], report["unclassified_columns"] = const, unknown
    print(f"  constant columns: {const or 'none'}")
    print(f"  columns in no layer: {unknown or 'none'}")
    print(health.groupby("layer").agg(columns=("column", "size"),
                                      mean_unique=("n_unique", "mean")).round(1).to_string())

    print("\n[5] HOW INDEPENDENT ARE Train.csv AND Test.csv?")
    a = audit()
    report["audit"] = a
    print(f"  Test messages that are also in Train (same time and class): {a['test_messages_present_in_train']:.1%}")
    print(f"  Test rows identical to a Train row in EVERY column          : {a['exact_duplicate_test_rows']:.1%}")
    print(f"  class counts equal in both files                            : {a['class_counts_equal']}")
    diff = a["columns_that_differ"]
    same = a["columns_identical_in_both"]
    print(f"  columns identical in both files: {len(same)} of {len(same) + len(diff)}; by layer "
          f"{dict(pd.Series([classify_column(c) for c in same]).value_counts())}")
    print(f"  columns that differ (fraction of rows): {diff}")
    tr_c, te_c = a["timing_reproducible_from_rows"]["train"], a["timing_reproducible_from_rows"]["test"]
    print(f"  timing columns reproducible from the rows' own order: Train {tr_c}  Test {te_c}")
    print("\n  -> Test.csv is the SAME recording as Train.csv, not held-out data. Scoring a model")
    print("     fitted on Train against Test measures memorisation (step 3 puts a number on it).")
    print("     Every reported figure uses blocked time CV inside Train.csv.")
    if min(tr_c.values()) > 0.9 and max(te_c.values()) < 0.9:
        print("     Train's derived timing columns match its own stream; Test's do not, so they were computed on a")
        print("     different stream. Train.csv is the one used for modelling.")

    with open(C.OUTPUT_DIR / "step1_profile.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\nwrote outputs/step1_profile.json, outputs/step1_column_health.csv")


if __name__ == "__main__":
    main()
