"""
STEP 3 - split safely, then prove the split is safe, then baseline.

Three things happen here, in this order, because the third is meaningless
without the first two:

  A  Structure report: how many events, how the folds are composed.
  B  LEAKAGE CONTROL (brief trap #3). Train a model on identifiers ONLY - file
     id, scenario id, position in event. These encode nothing about grid
     physics. If such a model scores well, the pipeline is leaking provenance
     and every later number is fiction. This must FAIL to be good news.
  C  Baselines on the fused view under all three splitters, so the report can
     show the honest number beside the inflated one (brief trap #5).

Run:  python step3_split_and_baseline.py
"""
import json

import numpy as np
import pandas as pd

from src import config as C
from src.loading import load_dataset
from src.metrics import aggregate_folds, evaluate, text_report, confusion
from src.models import EncodedModel, backend_name
from src.splits import SPLITTERS, describe


def run_cv(df, feature_cols, splitter_name, label=""):
    """Cross-validate the shared model over one splitter, return fold means."""
    fold_results = []
    splitter = SPLITTERS[splitter_name]
    for fold, (tr, te) in enumerate(splitter(df)):
        Xtr, ytr = df.iloc[tr][feature_cols], df.iloc[tr][C.TARGET]
        Xte, yte = df.iloc[te][feature_cols], df.iloc[te][C.TARGET]
        model = EncodedModel(n_classes=len(C.CLASS_ORDER)).fit(Xtr, ytr)
        pred = model.predict(Xte)
        proba = model.predict_proba(Xte)
        res = evaluate(yte, pred, proba)
        fold_results.append(res)
        print(f"    fold {fold}: macro_f1={res['macro_f1']:.4f}  "
              + "  ".join(f"recall[{c}]={res[f'recall__{c}']:.3f}" for c in C.CLASS_ORDER))
    agg = aggregate_folds(fold_results)
    print(f"  {label or splitter_name}: macro_f1 = "
          f"{agg['macro_f1']:.4f} +/- {agg['macro_f1__sd']:.4f}")
    return agg


def main():
    print("=" * 78)
    print("STEP 3 - SPLITTING, LEAKAGE CONTROL, BASELINES")
    print(f"model backend: {backend_name()}")
    print("=" * 78)

    df, feature_cols, views = load_dataset()
    out = {"backend": backend_name()}

    # ------------------------------------------------------------ A structure
    print("\n[A] SPLIT STRUCTURE")
    struct = describe(df)
    out["structure"] = struct
    for k, v in struct.items():
        print(f"  {k:<24} {v}")

    # Show that the grouped splitter really does keep events whole.
    tr, te = next(SPLITTERS["by_event"](df))
    overlap = set(df.iloc[tr][C.GROUP_KEY]) & set(df.iloc[te][C.GROUP_KEY])
    print(f"\n  events appearing in BOTH train and test: {len(overlap)}  "
          f"(must be 0)")
    assert not overlap, "event leaked across the split - splits.by_event is broken"

    # ----------------------------------------------------- B leakage control
    print("\n[B] PROVENANCE LEAKAGE CONTROL  (trap #3)")
    print("  Training on identifiers only: file_id, scenario_id, t_in_event.")
    print("  A high score here means the model can recover the label from")
    print("  bookkeeping rather than from the grid. We WANT this to be poor.")

    leak_df = df.copy()
    leak_features = ["file_id", "t_in_event", "event_len"]
    # scenario_id is deliberately EXCLUDED from the honest leak test because it
    # determines the label by construction (see scenario_label_map.csv) - it
    # would score 1.00 and tell us nothing. It is reported separately as the
    # upper bound.
    leak_results = {}
    for name, cols in [("identifiers_only", leak_features),
                       ("scenario_id_only", ["scenario_id"])]:
        fold_results = []
        for tr, te in SPLITTERS["by_event"](leak_df):
            m = EncodedModel(len(C.CLASS_ORDER)).fit(
                leak_df.iloc[tr][cols], leak_df.iloc[tr][C.TARGET])
            fold_results.append(evaluate(
                leak_df.iloc[te][C.TARGET], m.predict(leak_df.iloc[te][cols])))
        agg = aggregate_folds(fold_results)
        leak_results[name] = agg
        print(f"  {name:<20} macro_f1 = {agg['macro_f1']:.4f} "
              f"+/- {agg['macro_f1__sd']:.4f}")
    out["leakage_control"] = leak_results

    ident = leak_results["identifiers_only"]["macro_f1"]
    if ident > 0.55:
        print("  ** WARNING: identifiers alone are predictive. Investigate")
        print("     before believing any experiment result. **")
    else:
        print("  -> identifiers alone are near-useless. Pipeline is clean on")
        print("     this axis; real features are doing the work.")

    # ---------------------------------------------------------- C baselines
    print("\n[C] FUSED-VIEW BASELINES UNDER EACH SPLITTER")
    print("  The gap between naive_rows and by_event is the size of the")
    print("  memorisation effect that most published numbers on this dataset")
    print("  contain. Report both, side by side (trap #5).")
    baselines = {}
    for splitter_name in ("naive_rows", "by_event", "by_scenario"):
        print(f"\n  --- {splitter_name} ---")
        baselines[splitter_name] = run_cv(df, views["all"], splitter_name)
    out["baselines"] = baselines

    honest = baselines["by_event"]["macro_f1"]
    naive = baselines["naive_rows"]["macro_f1"]
    hardest = baselines["by_scenario"]["macro_f1"]
    print("\n  SUMMARY")
    print(f"    naive shuffled rows   {naive:.4f}   <- inflated, do not report alone")
    print(f"    event-held-out        {honest:.4f}   <- the honest headline")
    print(f"    scenario-held-out     {hardest:.4f}   <- E8, new-site difficulty")
    print(f"    memorisation gap      {naive - honest:+.4f}")
    print(f"    transferability gap   {honest - hardest:+.4f}  (hypothesis H5)")

    # one full text report on the honest split, for the appendix
    tr, te = next(SPLITTERS["by_event"](df))
    m = EncodedModel(len(C.CLASS_ORDER)).fit(df.iloc[tr][views["all"]], df.iloc[tr][C.TARGET])
    pred = m.predict(df.iloc[te][views["all"]])
    print("\n  classification report, fold 0, event-held-out:")
    print(text_report(df.iloc[te][C.TARGET], pred))
    print("  confusion matrix (rows=true, cols=pred, order "
          f"{C.CLASS_ORDER}):")
    print(confusion(df.iloc[te][C.TARGET], pred))

    with open(C.OUTPUT_DIR / "step3_baselines.json", "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nwrote {C.OUTPUT_DIR/'step3_baselines.json'}")


if __name__ == "__main__":
    main()
