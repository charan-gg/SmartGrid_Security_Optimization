"""
One model factory, used by every experiment.

E1 only means anything if each modality view gets the identical model with the
identical budget - otherwise a view could win because it got more trees, not
because it carries more evidence. So the hyperparameters live in config and
nothing here tunes them.

XGBoost is the intended model (it is also what the architecture assigns to the
Tier 2 correlator, so this code is not throwaway). If it is not installed we
fall back to sklearn's HistGradientBoostingClassifier, which is the same family
and, critically, also routes NaN natively - which the Inf policy depends on.
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import LabelEncoder

from . import config as C

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:                       # pragma: no cover
    HAS_XGB = False
    from sklearn.ensemble import HistGradientBoostingClassifier


def backend_name() -> str:
    return "xgboost" if HAS_XGB else "sklearn-histgradientboosting"


def make_model(n_classes: int, seed: int = None, **overrides):
    """A fresh, untrained classifier under the frozen shared budget."""
    seed = C.RANDOM_SEED if seed is None else seed
    params = {**C.MODEL_PARAMS, **overrides}

    if HAS_XGB:
        # num_class is pinned to the FULL label space, not to whatever a given
        # fold happens to contain. The by_scenario split that holds out scenario
        # 41 trains on two classes only (NoEvents is a single scenario), and
        # letting xgboost infer num_class there would produce a 2-column
        # probability matrix for that fold and a 3-column one for every other -
        # silently incomparable across folds.
        #
        # The cost of pinning it is that predict() then returns the raw
        # (n, num_class) softprob matrix rather than a label vector, and
        # predict_proba()'s columns become code-indexed. EncodedModel handles
        # both; see its predict/predict_proba.
        return XGBClassifier(
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            num_class=n_classes if n_classes > 2 else None,
            eval_metric="mlogloss" if n_classes > 2 else "logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
            **params,
        )

    # Map the shared budget onto the sklearn estimator's parameter names so the
    # two backends really are running the same budget, not just similar ones.
    return HistGradientBoostingClassifier(
        max_iter=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        l2_regularization=params["reg_lambda"],
        random_state=seed,
    )


class EncodedModel:
    """Thin wrapper that keeps the string labels ('Attack', ...) at the edges and
    integer labels inside, so every experiment can work with readable class
    names without each one re-implementing the encoding."""

    def __init__(self, n_classes: int, seed: int = None, **overrides):
        self.encoder = LabelEncoder().fit(C.CLASS_ORDER)
        self.model = make_model(n_classes, seed, **overrides)

    @property
    def classes_(self):
        """Report order, NOT the encoder's internal sorted order."""
        return list(C.CLASS_ORDER)

    def fit(self, X, y):
        self.model.fit(X, self.encoder.transform(y))
        return self

    def predict(self, X):
        """Label vector, whatever shape the backend handed back.

        With num_class pinned, xgboost's predict() returns the (n, num_class)
        softprob matrix instead of labels, so collapse it here rather than in
        each of the eight experiments.
        """
        raw = np.asarray(self.model.predict(X))
        if raw.ndim == 2:
            raw = raw.argmax(axis=1)
        return self.encoder.inverse_transform(raw.astype(int))

    def predict_proba(self, X):
        """Columns are in config.CLASS_ORDER, always.

        This is not cosmetic. LabelEncoder SORTS its classes, so its internal
        order is ['Attack', 'NoEvents', 'Natural'] while the project reports in
        ['NoEvents', 'Natural', 'Attack']. Returning the raw matrix would hand
        every caller silently transposed columns - PR-AUC would be computed
        against the wrong class and E2 would track the wrong probability - with
        no error anywhere. So the columns are remapped by NAME here, once.

        A fold that happens to contain no samples of some class gets a column
        of zeros for it rather than a shape mismatch.
        """
        raw = np.asarray(self.model.predict_proba(X))
        model_classes = list(self.model.classes_)
        n = len(C.CLASS_ORDER)
        out = np.zeros((len(X), n), dtype=float)
        # Two possible column conventions, depending on the backend and on
        # whether this fold saw every class:
        #   width == n  -> columns are indexed by LABEL CODE (num_class pinned)
        #   width <  n  -> columns follow model.classes_, the classes this fold
        #                  actually trained on; the absent class gets zeros
        code_indexed = raw.shape[1] == n
        for j, name in enumerate(C.CLASS_ORDER):
            code = int(self.encoder.transform([name])[0])
            if code_indexed:
                out[:, j] = raw[:, code]
            elif code in model_classes:
                out[:, j] = raw[:, model_classes.index(code)]
        return out

    def feature_importance(self, feature_names, X=None, y=None,
                           n_samples: int = 4000, n_repeats: int = 2) -> dict:
        """{feature: importance}, used by E3 to rank features and by E7 as a
        SHAP backstop.

        XGBoost exposes split gain directly. HistGradientBoostingClassifier
        exposes NO importances at all, so for that backend we must compute
        permutation importance, which needs data - hence the optional X/y.
        Silently returning zeros here would make E3's ranking arbitrary while
        still producing a plausible-looking k* curve, so this raises instead.
        """
        if HAS_XGB:
            booster = self.model.get_booster()
            score = booster.get_score(importance_type="gain")
            out = {name: 0.0 for name in feature_names}
            # Booster keys are the real feature names when the model was fitted
            # from a DataFrame (xgboost >= 2), but fall back to positional
            # f0/f1/... when it was fitted from a bare array. Handle both, and
            # never silently drop a key - an unmatched key means the ranking
            # would be built from an incomplete importance table.
            for key, val in score.items():
                if key in out:
                    out[key] = float(val)
                elif key.startswith("f") and key[1:].isdigit():
                    out[feature_names[int(key[1:])]] = float(val)
                else:
                    raise KeyError(
                        f"booster importance key {key!r} matches no feature name")
            return out

        importances = getattr(self.model, "feature_importances_", None)
        if importances is not None:
            return dict(zip(feature_names, map(float, importances)))

        if X is None or y is None:
            raise RuntimeError(
                f"backend {backend_name()} exposes no feature importances; pass "
                "X and y so permutation importance can be computed. Refusing to "
                "return zeros, which would silently make any ranking arbitrary."
            )

        from sklearn.inspection import permutation_importance
        # Subsample: permutation importance costs n_features x n_repeats full
        # predictions, which is unaffordable on the full training split and
        # unnecessary - the ranking stabilises well before 58k rows.
        idx = np.random.default_rng(C.RANDOM_SEED).choice(
            len(X), size=min(n_samples, len(X)), replace=False)
        Xs = X.iloc[idx]
        ys = self.encoder.transform(np.asarray(y)[idx])
        # n_jobs=1 deliberately: joblib's loky backend cannot un-serialise the
        # Arrow-backed pandas frames this project loads, and fails with an
        # opaque BrokenProcessPool. Single-process is slower but correct.
        result = permutation_importance(
            self.model, Xs, ys, n_repeats=n_repeats,
            random_state=C.RANDOM_SEED, n_jobs=1, scoring="f1_macro")
        return dict(zip(feature_names, map(float, result.importances_mean)))
