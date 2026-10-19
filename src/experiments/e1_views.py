"""
E1 - VIEW ABLATION.
Architectural parameter it fixes: which layer of the substation must feed the
detector, and whether a fusion point (one node seeing several layers) is needed.

Method: the SAME model, the SAME hyperparameter budget, the SAME blocked-time folds -
only the input columns change. Any difference between views is then attributable to
the evidence in that layer and nothing else.

Two controls guard against the model learning the CAPTURE instead of the attacks:
  control_time_only      the raw clock alone
  control_counters_only  sqNum + stnum alone (GOOSE counters that grow with the capture)
Both should do badly under blocked CV; under shuffled rows they show how much a
random split would have flattered a model.

What to read: per-class recall, not macro-F1 alone. Fusion is justified on accuracy
only if the fused view beats every single view by more than the fold-to-fold spread.
"""
import numpy as np
import pandas as pd

from .. import config as C
from ..metrics import aggregate_folds, attack_coverage, evaluate
from ..models import EncodedModel
from ..splits import SPLITTERS

CONTROLS = {"control_time_only": ["t_ms"], "control_counters_only": ["sqNum", "stnum"]}
SINGLES = ("process", "goose", "consistency", "relay")


def _cv(df, cols, splitter):
    folds = []
    for tr, te in SPLITTERS[splitter](df):
        m = EncodedModel().fit(df.iloc[tr][cols], df.iloc[tr][C.TARGET])
        folds.append(evaluate(df.iloc[te][C.TARGET], m.predict(df.iloc[te][cols]),
                              m.predict_proba(df.iloc[te][cols])))
    return folds


def run(df, views):
    rows, fold_f1 = [], {}
    for name, cols in {**{k: v for k, v in views.items() if v}, **CONTROLS}.items():
        folds = _cv(df, cols, "blocked")
        agg = aggregate_folds(folds)
        agg.update(view=name, n_features=len(cols), attack_coverage=attack_coverage(agg))
        rows.append(agg)
        fold_f1[name] = [f["macro_f1"] for f in folds]
        print(f"  {name:<22} n={len(cols):>3}  macroF1={agg['macro_f1']:.4f}+/-{agg['macro_f1__sd']:.4f}  "
              + "  ".join(f"{c[:4]}={agg[f'recall__{c}']:.2f}" for c in C.CLASS_ORDER))
    res = pd.DataFrame(rows).set_index("view")

    print("\n  SAME CONTROLS UNDER SHUFFLED ROWS (what a random split would have shown)")
    naive = {}
    for name, cols in CONTROLS.items():
        naive[name] = float(np.mean([f["macro_f1"] for f in _cv(df, cols, "naive_rows")]))
        print(f"  {name:<22} shuffled macroF1={naive[name]:.4f}   blocked macroF1="
              f"{res.loc[name, 'macro_f1']:.4f}")

    print("\n  PER-CLASS WINNER AMONG SINGLE LAYERS")
    singles = [v for v in SINGLES if v in res.index]
    for cls in C.CLASS_ORDER:
        col = f"recall__{cls}"
        best = res.loc[singles, col].idxmax()
        print(f"    {cls:<11} best single layer = {best:<12} ({res.loc[best, col]:.3f})   "
              f"all features = {res.loc['all', col]:.3f}")

    gain = res.loc["all", "macro_f1"] - res.loc[singles, "macro_f1"].max()
    print(f"\n  fusion gain over the best single layer : {gain:+.4f}  (fold SD of 'all' "
          f"{res.loc['all', 'macro_f1__sd']:.4f})")
    paired = np.array(fold_f1["all"]) - np.array(fold_f1["station"])
    print(f"  phasor gain (all - station)            : {paired.mean():+.4f}  (paired fold SD "
          f"{paired.std():.4f})")
    verdict = "within noise" if abs(paired.mean()) <= paired.std() else "larger than noise"
    print(f"    -> adding the MU phasors to the station-bus evidence is {verdict}.")
    print("\n  ATTACK COVERAGE (mean recall over Masquerade/Injection/Replay), the per-view")
    print("  detection figure step 5 uses:")
    for v in ("all", "station", "process"):
        print(f"    {v:<9} {res.loc[v, 'attack_coverage']:.3f}")

    res.to_csv(C.OUTPUT_DIR / "e1_views_blocked.csv")
    pd.Series(naive).to_csv(C.OUTPUT_DIR / "e1_controls_shuffled.csv", header=["macro_f1"])
    return res
