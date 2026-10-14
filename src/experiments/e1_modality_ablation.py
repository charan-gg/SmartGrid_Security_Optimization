"""
E1 - MODALITY ABLATION.
Architectural parameter it fixes: is a fusion point needed at all, and which
conduits must deliver which modality to the same node.

Method: the SAME model, the SAME hyperparameter budget, the SAME event-grouped
folds - only the input columns change. Any difference between views is then
attributable to the evidence in that layer and nothing else.

What to read: per-class recall, never macro-F1 alone. H1 (complementarity) is
supported only if DIFFERENT views win on DIFFERENT classes. If one view wins
everything, fusion is an accuracy trick and the thesis is weaker.
"""
import pandas as pd

from .. import config as C
from ..metrics import aggregate_folds, evaluate
from ..models import EncodedModel
from ..splits import SPLITTERS


def run(df, views, splitter_name="by_event"):
    rows = []
    for view_name, cols in views.items():
        if not cols:
            continue
        fold_results = []
        for tr, te in SPLITTERS[splitter_name](df):
            model = EncodedModel(len(C.CLASS_ORDER)).fit(
                df.iloc[tr][cols], df.iloc[tr][C.TARGET])
            fold_results.append(evaluate(
                df.iloc[te][C.TARGET],
                model.predict(df.iloc[te][cols]),
                model.predict_proba(df.iloc[te][cols]),
            ))
        agg = aggregate_folds(fold_results)
        agg.update(view=view_name, n_features=len(cols))
        rows.append(agg)
        print(f"  {view_name:<12} n={len(cols):>4}  "
              f"macroF1={agg['macro_f1']:.4f}+/-{agg['macro_f1__sd']:.4f}  "
              + "  ".join(f"{c[:3]}={agg[f'recall__{c}']:.3f}" for c in C.CLASS_ORDER))

    res = pd.DataFrame(rows).set_index("view")

    # --- the actual finding: who wins each class, and does fusion add anything
    print("\n  PER-CLASS WINNER (H1 complementarity)")
    singles = [v for v in res.index if v not in ("all", "cyber_only")]
    for cls in C.CLASS_ORDER:
        col = f"recall__{cls}"
        best = res.loc[singles, col].idxmax()
        print(f"    {cls:<10} best single modality = {best:<8} "
              f"({res.loc[best, col]:.3f})   fused = {res.loc['all', col]:.3f}")
    distinct = {res.loc[singles, f'recall__{c}'].idxmax() for c in C.CLASS_ORDER}
    print(f"    -> {len(distinct)} distinct winning modalities across "
          f"{len(C.CLASS_ORDER)} classes.")
    print("       H1 is supported only if this is > 1. If it is 1, say so "
          "plainly in the report rather than burying it.")

    gain = res.loc["all", "macro_f1"] - res.loc[singles, "macro_f1"].max()
    sd = res.loc["all", "macro_f1__sd"]
    print(f"\n  fusion gain over best single view: {gain:+.4f} "
          f"(fold SD {sd:.4f})")
    print("    -> if the gain is smaller than the SD, fusion buys nothing "
          "measurable here and must be justified on COVERAGE, not accuracy.")

    res.to_csv(C.OUTPUT_DIR / f"e1_modality_ablation_{splitter_name}.csv")
    return res
