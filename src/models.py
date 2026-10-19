"""
One model factory, used by every experiment.

E1 only means anything if each view gets the identical model with the identical
budget - otherwise a view could win because it got more trees, not because it
carries more evidence. So the hyperparameters live in config and nothing here
tunes them.

XGBoost is the intended model. If it is not installed we fall back to sklearn's
HistGradientBoostingClassifier, which is the same family and also routes NaN
natively.
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
    return "xgboost" if HAS_XGB else "sklearn-histgradboosting"


def make_model(n_classes: int, seed: int = None, **overrides):
    """A fresh, untrained classifier under the frozen shared budget."""
    seed = C.RANDOM_SEED if seed is None else seed
    params = {**C.MODEL_PARAMS, **overrides}

    if HAS_XGB:
        # The sklearn wrapper sets num_class from the labels it is fitted on; EncodedModel
        # re-maps probability columns by name, so folds still share one output width.
        return XGBClassifier(
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            eval_metric="mlogloss" if n_classes > 2 else "logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
            **params,
        )

    return HistGradientBoostingClassifier(
        max_iter=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        l2_regularization=params["reg_lambda"],
        random_state=seed,
    )


class EncodedModel:
    """Keeps string labels ('Replay', ...) at the edges and integers inside, so every
    experiment works with readable class names. `classes` is the report order;
    default is config.CLASS_ORDER (the 5-class problem). E6/E8 pass a 2-class list."""

    def __init__(self, classes=None, seed: int = None, **overrides):
        self.classes = list(classes if classes is not None else C.CLASS_ORDER)
        self.encoder = LabelEncoder().fit(self.classes)
        self.model = make_model(len(self.classes), seed, **overrides)

    @property
    def classes_(self):
        """Report order, NOT the encoder's internal sorted order."""
        return list(self.classes)

    def _encode(self, y):
        """Report-order labels -> contiguous 0..k-1 codes over the classes this model was
        fitted on (xgboost refuses gaps such as [0, 3, 4])."""
        return np.searchsorted(self._present_codes, self.encoder.transform(np.asarray(y)))

    def fit(self, X, y):
        codes = np.unique(self.encoder.transform(np.asarray(y)))
        if len(codes) < 2:
            raise ValueError(f"need at least two classes to fit, got {list(self.encoder.inverse_transform(codes))}")
        self._present_codes = codes
        self.model.fit(X, self._encode(y))
        return self

    def predict(self, X):
        """Label vector, whatever shape the backend handed back (with num_class
        pinned, xgboost's predict() can return the raw softprob matrix)."""
        raw = np.asarray(self.model.predict(X))
        if raw.ndim == 2:
            raw = raw.argmax(axis=1)
        return self.encoder.inverse_transform(self._present_codes[raw.astype(int)])

    def predict_proba(self, X):
        """Columns are in self.classes order, always.

        LabelEncoder SORTS its classes, so its internal order differs from the report
        order, and a fold may have been fitted without some class. Columns are therefore
        remapped by NAME, once: a class the model never saw gets a column of zeros.
        """
        raw = np.asarray(self.model.predict_proba(X))
        out = np.zeros((len(X), len(self.classes)), dtype=float)
        present = list(self._present_codes)
        for j, name in enumerate(self.classes):
            code = int(self.encoder.transform([name])[0])
            if code in present:
                out[:, j] = raw[:, present.index(code)]
        return out

    def feature_importance(self, feature_names, X=None, y=None,
                           n_samples: int = 4000, n_repeats: int = 2) -> dict:
        """{feature: importance}, used by E3 to rank features.

        XGBoost exposes split gain directly. HistGradientBoostingClassifier exposes
        none, so that backend computes permutation importance and needs X/y. Silently
        returning zeros would make E3's ranking arbitrary, so this raises instead.
        """
        if HAS_XGB:
            score = self.model.get_booster().get_score(importance_type="gain")
            out = {name: 0.0 for name in feature_names}
            for key, val in score.items():
                if key in out:
                    out[key] = float(val)
                elif key.startswith("f") and key[1:].isdigit():
                    out[feature_names[int(key[1:])]] = float(val)
                else:
                    raise KeyError(f"booster importance key {key!r} matches no feature name")
            return out

        importances = getattr(self.model, "feature_importances_", None)
        if importances is not None:
            return dict(zip(feature_names, map(float, importances)))
        if X is None or y is None:
            raise RuntimeError(
                f"backend {backend_name()} exposes no feature importances; pass X and y so "
                "permutation importance can be computed. Refusing to return zeros.")
        from sklearn.inspection import permutation_importance
        idx = np.random.default_rng(C.RANDOM_SEED).choice(
            len(X), size=min(n_samples, len(X)), replace=False)
        ys = self._encode(np.asarray(y)[idx])
        result = permutation_importance(self.model, X.iloc[idx], ys, n_repeats=n_repeats,
                                        random_state=C.RANDOM_SEED, n_jobs=1, scoring="f1_macro")
        return dict(zip(feature_names, map(float, result.importances_mean)))
