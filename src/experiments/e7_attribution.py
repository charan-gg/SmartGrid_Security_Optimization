"""
E7 - PER-CLASS ATTRIBUTION.
Architectural parameters it fixes: which conduits must be authenticated
(IEC 62351), and what an alert has to carry as evidence (design rule R5).

The logic: if class X is detected almost entirely from relay-log evidence, then
an attacker who can forge relay logs can blind the detector for class X. So the
attribution map is also a threat model - it tells you which links are worth the
cost of authentication and which are not.

Uses SHAP if installed (exact for tree models), otherwise falls back to model
gain, which is coarser but directionally the same. The fallback is labelled in
the output so the report never silently claims SHAP it did not run.
"""
import numpy as np
import pandas as pd

from .. import config as C
from ..models import EncodedModel
from ..modalities import classify_column
from ..splits import by_event_single

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def run(df, views, view_name="all", max_background=2000):
    tr, te = by_event_single(df)
    train_df, test_df = df.iloc[tr], df.iloc[te]
    cols = views[view_name]

    model = EncodedModel(len(C.CLASS_ORDER)).fit(train_df[cols], train_df[C.TARGET])
    method = "shap" if HAS_SHAP else "gain"
    print(f"  attribution method: {method}"
          + ("" if HAS_SHAP else "   (pip install shap for per-class SHAP)"))

    if HAS_SHAP:
        # Subsample: TreeExplainer is exact but the cost scales with rows, and
        # a few thousand rows already gives a stable mean |SHAP|.
        sample = test_df.sample(min(max_background, len(test_df)), random_state=C.RANDOM_SEED)
        explainer = shap.TreeExplainer(model.model)
        values = explainer.shap_values(sample[cols])
        # normalise shape to (n_classes, n_rows, n_features)
        if isinstance(values, list):
            arr = np.stack(values, axis=0)
        else:
            arr = np.asarray(values)
            if arr.ndim == 3 and arr.shape[-1] == len(C.CLASS_ORDER):
                arr = np.transpose(arr, (2, 0, 1))
        per_class = {cls: np.abs(arr[i]).mean(axis=0) for i, cls in enumerate(C.CLASS_ORDER)}
    else:
        # Fallback: per-class permutation importance. Scored on the RECALL of
        # one class at a time, so the result really is class-specific - shuffle
        # a feature and see how much detection of THAT class collapses.
        # (Refitting the same multiclass model per class, as an earlier draft
        # did, returns the identical vector three times and is meaningless.)
        from sklearn.inspection import permutation_importance
        from sklearn.metrics import recall_score

        sample = test_df.sample(min(max_background, len(test_df)), random_state=C.RANDOM_SEED)
        Xs = sample[cols]
        ys = model.encoder.transform(sample[C.TARGET].to_numpy())

        def class_recall_scorer(code):
            def _score(estimator, X, y):
                return recall_score(y, estimator.predict(X), labels=[code],
                                    average="macro", zero_division=0)
            return _score

        per_class = {}
        for cls in C.CLASS_ORDER:
            code = int(model.encoder.transform([cls])[0])
            r = permutation_importance(
                model.model, Xs, ys, n_repeats=2, random_state=C.RANDOM_SEED,
                n_jobs=1, scoring=class_recall_scorer(code))
            per_class[cls] = r.importances_mean

    attribution = pd.DataFrame(per_class, index=cols)
    attribution["modality"] = [classify_column(c) for c in cols]
    attribution.to_csv(C.OUTPUT_DIR / "e7_feature_attribution.csv")

    # ----------------------------------------- roll up features -> conduits
    by_mod = attribution.groupby("modality")[C.CLASS_ORDER].sum()
    share = by_mod / by_mod.sum(axis=0)
    print("\n  SHARE OF EVIDENCE BY CONDUIT, PER CLASS")
    print((share * 100).round(1).to_string())
    share.to_csv(C.OUTPUT_DIR / "e7_modality_attribution.csv")

    print("\n  AUTHENTICATION PRIORITY (IEC 62351 scoping)")
    for cls in C.CLASS_ORDER:
        top = share[cls].sort_values(ascending=False)
        lead = top.index[0]
        print(f"    {cls:<10} dominated by {lead:<8} ({top.iloc[0]:.1%})  "
              f"-> forging {lead} blinds detection of {cls}")
    print("\n  -> a conduit carrying >70% of any class's evidence is a single")
    print("     point of failure for that class and must be authenticated:")
    # share is (modality x class); axis=0 maxes DOWN the modalities, so the
    # result is indexed by class and must be reported as (class, modality).
    flagged = False
    for cls in C.CLASS_ORDER:
        col = share[cls]
        mod = col.idxmax()
        if col.max() > 0.70:
            flagged = True
            print(f"       {mod:<8} carries {col.max():.1%} of class '{cls}'")
    if not flagged:
        print("       none - evidence is spread across conduits for every class")

    print("\n  ALERT CONTENT (design rule R5)")
    print("  Top 5 features per class - these are what an alert must quote so")
    print("  an operator can act on it without re-querying the whole system:")
    for cls in C.CLASS_ORDER:
        top5 = attribution[cls].sort_values(ascending=False).head(5)
        print(f"    {cls:<10} " + ", ".join(top5.index))

    return attribution, share
