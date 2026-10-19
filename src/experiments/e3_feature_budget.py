"""
E3 - MINIMAL FEATURE SET k*.
Architectural parameter it fixes: the edge -> correlator conduit spec, i.e. how
many numbers a Tier 1 sentinel has to push upstream per message, and therefore the
bandwidth each substation consumes.

Method: rank features on the training folds by model gain, then retrain the
identical model on the top-k for a ladder of k. k* is the smallest budget whose
macro-F1 is within KSTAR_TOLERANCE of the full-feature model.

Ranking is computed INSIDE the cross-validation loop, not once on all the data:
ranking on everything and then cross-validating the reduced set is a classic
selection leak that inflates small-k scores.
"""
from collections import Counter

import pandas as pd

from .. import config as C
from ..data import rates
from ..metrics import aggregate_folds, evaluate, upstream_bytes_per_second
from ..models import EncodedModel
from ..modalities import classify_column
from ..splits import SPLITTERS


def run(df, views, view_name="all"):
    cols = views[view_name]
    rate = rates(df)["row_rate_hz"]
    budgets = [k for k in C.FEATURE_BUDGETS if k <= len(cols)]
    if len(cols) not in budgets:
        budgets.append(len(cols))
    per_budget = {k: [] for k in budgets}
    chosen = {k: [] for k in budgets}

    for tr, te in SPLITTERS["blocked"](df):
        Xtr, ytr = df.iloc[tr][cols], df.iloc[tr][C.TARGET]
        Xte, yte = df.iloc[te][cols], df.iloc[te][C.TARGET]
        ranker = EncodedModel().fit(Xtr, ytr)
        importance = ranker.feature_importance(cols, X=Xtr, y=ytr)
        ranked = [f for f, _ in sorted(importance.items(), key=lambda kv: -kv[1])]
        for k in budgets:
            top = ranked[:k]
            chosen[k].append(top)
            m = EncodedModel().fit(Xtr[top], ytr)
            per_budget[k].append(evaluate(yte, m.predict(Xte[top]), m.predict_proba(Xte[top])))

    rows = []
    for k in budgets:
        agg = aggregate_folds(per_budget[k])
        agg.update(k=k, upstream_bytes_per_s=upstream_bytes_per_second(k, rate))
        rows.append(agg)
    res = pd.DataFrame(rows).sort_values("k").set_index("k")

    full = res.loc[max(budgets), "macro_f1"]
    kstar = int(res[res["macro_f1"] >= full - C.KSTAR_TOLERANCE].index.min())
    print(f"  full-feature macro-F1 ({max(budgets)} features): {full:.4f}   message rate {rate:.2f}/s")
    for k in budgets:
        print(f"    k={k:>3}  macroF1={res.loc[k, 'macro_f1']:.4f}+/-{res.loc[k, 'macro_f1__sd']:.4f}  "
              f"upstream={res.loc[k, 'upstream_bytes_per_s']:7.1f} B/s per unit"
              f"{'  <== k*' if k == kstar else ''}")
    saving = 1 - res.loc[kstar, "upstream_bytes_per_s"] / res.loc[max(budgets), "upstream_bytes_per_s"]
    print(f"\n  k* = {kstar} features (within {C.KSTAR_TOLERANCE} macro-F1 of full); "
          f"bandwidth saving vs shipping everything: {saving:.1%}")

    counts = Counter(f for fold_list in chosen[kstar] for f in fold_list)
    stable = [f for f, n in counts.items() if n == C.N_SPLITS]
    by_layer = Counter(classify_column(f) for f in stable)
    print(f"  features chosen in ALL {C.N_SPLITS} folds: {len(stable)}/{kstar}  by layer: {dict(by_layer)}")
    print("  -> only these belong in a conduit spec; fold-unstable features are noise-selected.")
    for f in stable:
        print(f"      {f:<22} {classify_column(f)}")

    res.to_csv(C.OUTPUT_DIR / "e3_feature_budget.csv")
    pd.Series(counts).sort_values(ascending=False).to_csv(C.OUTPUT_DIR / "e3_feature_stability.csv")
    return res, kstar, stable
