"""
E3 - MINIMAL FEATURE SET k*.
Architectural parameter it fixes: the edge -> correlator conduit spec, i.e. how
many numbers a Tier 1 sentinel has to push upstream every sample, and therefore
the bandwidth each substation consumes.

Method: rank features once on the training folds by model gain, then retrain
the identical model on the top-k for a ladder of k. k* is the smallest budget
whose macro-F1 is within KSTAR_TOLERANCE of the full-feature model.

Ranking is computed INSIDE the cross-validation loop, not once on all the data.
Ranking on the full dataset and then cross-validating the reduced set is a
classic selection-leak that inflates small-k scores.
"""
import numpy as np
import pandas as pd

from .. import config as C
from ..metrics import aggregate_folds, evaluate, upstream_bytes_per_second
from ..models import EncodedModel
from ..splits import SPLITTERS


def run(df, views, view_name="all"):
    cols = views[view_name]
    budgets = [k for k in C.FEATURE_BUDGETS if k <= len(cols)]
    if len(cols) not in budgets:
        budgets.append(len(cols))

    # accumulate per-fold results for every budget
    per_budget = {k: [] for k in budgets}
    chosen_features = {k: [] for k in budgets}

    for fold, (tr, te) in enumerate(SPLITTERS["by_event"](df)):
        Xtr, ytr = df.iloc[tr][cols], df.iloc[tr][C.TARGET]
        Xte, yte = df.iloc[te][cols], df.iloc[te][C.TARGET]

        # rank on THIS fold's training data only
        ranker = EncodedModel(len(C.CLASS_ORDER)).fit(Xtr, ytr)
        # X/y are passed because the sklearn fallback backend needs them for
        # permutation importance; XGBoost ignores them and uses split gain.
        importance = ranker.feature_importance(cols, X=Xtr, y=ytr)
        ranked = [f for f, _ in sorted(importance.items(), key=lambda kv: -kv[1])]

        for k in budgets:
            top = ranked[:k]
            chosen_features[k].append(top)
            m = EncodedModel(len(C.CLASS_ORDER)).fit(Xtr[top], ytr)
            per_budget[k].append(evaluate(yte, m.predict(Xte[top]), m.predict_proba(Xte[top])))

    rows = []
    for k in budgets:
        agg = aggregate_folds(per_budget[k])
        agg.update(k=k, upstream_bytes_per_s=upstream_bytes_per_second(k))
        rows.append(agg)
    res = pd.DataFrame(rows).sort_values("k").set_index("k")

    full = res.loc[max(budgets), "macro_f1"]
    within = res[res["macro_f1"] >= full - C.KSTAR_TOLERANCE]
    kstar = int(within.index.min())

    print(f"  full-feature macro-F1 ({max(budgets)} features): {full:.4f}")
    for k in budgets:
        mark = "  <== k*" if k == kstar else ""
        print(f"    k={k:>4}  macroF1={res.loc[k,'macro_f1']:.4f}"
              f"+/-{res.loc[k,'macro_f1__sd']:.4f}  "
              f"upstream={res.loc[k,'upstream_bytes_per_s']/1024:7.1f} KiB/s per bay{mark}")

    print(f"\n  k* = {kstar} features (within {C.KSTAR_TOLERANCE} macro-F1 of full)")
    print(f"  H3 (<20 features) is {'SUPPORTED' if kstar < 20 else 'NOT supported'}")
    saving = 1 - res.loc[kstar, "upstream_bytes_per_s"] / res.loc[max(budgets), "upstream_bytes_per_s"]
    print(f"  bandwidth saving vs shipping everything: {saving:.1%}")

    # which features survive, and how stable is that choice across folds?
    from collections import Counter
    counts = Counter(f for fold_list in chosen_features[kstar] for f in fold_list)
    stable = [f for f, n in counts.items() if n == C.N_SPLITS]
    print(f"\n  features chosen in ALL {C.N_SPLITS} folds: {len(stable)}/{kstar}")
    print("  -> only these belong in a conduit spec; fold-unstable features are")
    print("     noise-selected and would not survive a different capture.")
    for f in list(stable)[:min(kstar, 25)]:
        print(f"      {f}")
    if len(stable) > 25:
        print(f"      ... and {len(stable)-25} more (see e3_feature_stability.csv)")

    res.to_csv(C.OUTPUT_DIR / "e3_feature_budget.csv")
    pd.Series(counts).sort_values(ascending=False).to_csv(
        C.OUTPUT_DIR / "e3_feature_stability.csv")
    return res, kstar, stable
