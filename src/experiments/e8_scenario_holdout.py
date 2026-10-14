"""
E8 - SCENARIO-HELD-OUT GENERALISATION.
Architectural parameter it fixes: central model shipping vs federated /
per-site training.

The question a utility actually asks: if we train this at site A and deploy it
at site B, which sees attack variants A never saw, does it still work? Here,
that is simulated by holding out WHOLE scenarios - the model is tested on
attack types it has never seen a single sample of.

This produces the lowest number in the whole project, and that is correct.
Brief trap #5: report it beside the naive number and explain the gap in
advance, as rigour. A small gap argues for shipping one central model. A large
gap argues that each site must train on its own data, which is a completely
different system to build.
"""
import numpy as np
import pandas as pd

from .. import config as C
from ..metrics import aggregate_folds, evaluate
from ..models import EncodedModel
from ..splits import SPLITTERS


def run(df, views, view_names=("all", "pmu", "relay")):
    view_names = [v for v in view_names if views.get(v)]
    rows = []
    for splitter_name in ("naive_rows", "by_event", "by_scenario"):
        for view_name in view_names:
            cols = views[view_name]
            folds = []
            for tr, te in SPLITTERS[splitter_name](df):
                m = EncodedModel(len(C.CLASS_ORDER)).fit(
                    df.iloc[tr][cols], df.iloc[tr][C.TARGET])
                folds.append(evaluate(df.iloc[te][C.TARGET],
                                      m.predict(df.iloc[te][cols]),
                                      m.predict_proba(df.iloc[te][cols])))
            agg = aggregate_folds(folds)
            agg.update(split=splitter_name, view=view_name)
            rows.append(agg)
            print(f"  {splitter_name:<12} {view_name:<10} "
                  f"macroF1={agg['macro_f1']:.4f}+/-{agg['macro_f1__sd']:.4f}")

    res = pd.DataFrame(rows)
    pivot = res.pivot(index="view", columns="split", values="macro_f1")
    print("\n  MACRO-F1 BY SPLIT DIFFICULTY")
    print(pivot.to_string(float_format=lambda v: f"{v:.4f}"))
    pivot["memorisation_gap"] = pivot["naive_rows"] - pivot["by_event"]
    pivot["transferability_gap"] = pivot["by_event"] - pivot["by_scenario"]
    print("\n  GAPS")
    print(pivot[["memorisation_gap", "transferability_gap"]]
          .to_string(float_format=lambda v: f"{v:+.4f}"))

    gap = float(pivot.loc["all", "transferability_gap"])
    print(f"\n  H5 transferability gap (fused view): {gap:+.4f}")
    if gap > 0.10:
        print("  -> LARGE. A centrally trained model does not transfer to unseen")
        print("     attack variants. The architecture should support per-site or")
        print("     federated training, and Tier 3 should own model updates.")
    else:
        print("  -> SMALL. One centrally trained model can be shipped to sites;")
        print("     Tier 3 can hold a single shared model.")

    # ------------------------------------- which scenarios are unlearnable?
    print("\n  PER-HELD-OUT-SCENARIO RECALL (fused view)")
    print("  Scenarios that collapse when unseen are the ones a new site would")
    print("  miss entirely - name them in the limitations section.")
    cols = views["all"]
    per_scenario = []
    for tr, te in SPLITTERS["by_scenario"](df):
        m = EncodedModel(len(C.CLASS_ORDER)).fit(df.iloc[tr][cols], df.iloc[tr][C.TARGET])
        test = df.iloc[te]
        pred = m.predict(test[cols])
        for sid, idx in test.groupby("scenario_id").groups.items():
            mask = test.index.isin(idx)
            truth = test.loc[mask, C.TARGET]
            correct = (pred[mask] == truth.to_numpy()).mean()
            per_scenario.append({"scenario_id": sid, "true_class": truth.iloc[0],
                                 "recall_when_unseen": float(correct),
                                 "n_rows": int(mask.sum())})
    ps = pd.DataFrame(per_scenario).sort_values("recall_when_unseen")
    ps.to_csv(C.OUTPUT_DIR / "e8_per_scenario.csv", index=False)
    print(ps.head(10).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    res.to_csv(C.OUTPUT_DIR / "e8_scenario_holdout.csv", index=False)
    return res, ps
