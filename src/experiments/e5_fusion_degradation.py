"""
E5 - FUSION STRATEGY AND GRACEFUL DEGRADATION.
Architectural parameter it fixes: the inter-tier contract, and whether the
detection plane survives a network partition.

Two fusion strategies:
  EARLY  one model over the concatenated columns. Simple, but it needs every
         modality present at inference; a missing conduit means missing columns.
  LATE   one expert model per modality, each emitting class probabilities, and
         a small meta-learner combining them. This is what the proposed
         architecture actually does - Tier 1 sentinels emit EVIDENCE VECTORS
         (design rule R5), not verdicts, and Tier 2 combines them.

Then we break things on purpose. For each modality, and for the WAN-loss case,
we remove that evidence at inference time only (the model is NOT retrained -
that is the point, a link drops without warning) and measure what survives.

H4 is supported if late fusion loses less than early fusion under modality loss.
"""
import numpy as np
import pandas as pd

from .. import config as C
from ..metrics import evaluate
from ..models import EncodedModel
from ..splits import by_event_single
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit


class LateFusion:
    """Per-modality experts + a logistic meta-learner over their probabilities."""

    def __init__(self, views, modalities):
        self.views = views
        self.modalities = modalities
        self.experts = {}
        self.meta = None
        self.prior = None

    def fit(self, train_df):
        # Inner split: experts learn on part A, the meta-learner learns how much
        # to trust each expert on part B. Fitting the meta-learner on the same
        # rows the experts memorised would make it trust them far too much.
        inner = GroupShuffleSplit(n_splits=1, test_size=0.35, random_state=C.RANDOM_SEED)
        a, b = next(inner.split(train_df, groups=train_df[C.GROUP_KEY]))
        part_a, part_b = train_df.iloc[a], train_df.iloc[b]

        for m in self.modalities:
            self.experts[m] = EncodedModel(len(C.CLASS_ORDER)).fit(
                part_a[self.views[m]], part_a[C.TARGET])

        meta_X = self._stack(part_b, drop=())
        # multinomial is the default in modern scikit-learn; the explicit
        # multi_class argument was removed in 1.7 and errors in 1.9.
        self.meta = LogisticRegression(max_iter=2000)
        self.meta.fit(meta_X, part_b[C.TARGET])

        # class prior, used to stand in for an expert whose conduit is down
        counts = train_df[C.TARGET].value_counts(normalize=True)
        self.prior = np.array([counts.get(c, 0.0) for c in C.CLASS_ORDER])
        return self

    def _stack(self, X_df, drop=()):
        """Concatenate every expert's probability vector. A dropped modality
        contributes the class prior instead - the honest representation of
        'this conduit told us nothing', not zeros."""
        blocks = []
        for m in self.modalities:
            if m in drop:
                blocks.append(np.tile(self.prior, (len(X_df), 1)))
            else:
                blocks.append(self.experts[m].predict_proba(X_df[self.views[m]]))
        return np.hstack(blocks)

    def predict(self, X_df, drop=()):
        return self.meta.predict(self._stack(X_df, drop))

    def predict_proba(self, X_df, drop=()):
        """Columns in config.CLASS_ORDER, for the same reason EncodedModel
        remaps them: LogisticRegression.classes_ is sorted alphabetically, not
        in report order, and handing the raw matrix to evaluate() would score
        every class against the wrong column."""
        raw = self.meta.predict_proba(self._stack(X_df, drop))
        cls = list(self.meta.classes_)
        out = np.zeros((len(X_df), len(C.CLASS_ORDER)), dtype=float)
        for j, name in enumerate(C.CLASS_ORDER):
            if name in cls:
                out[:, j] = raw[:, cls.index(name)]
        return out


def run(df, views, modalities=("pmu", "relay", "snort", "control")):
    modalities = [m for m in modalities if views.get(m)]
    tr, te = by_event_single(df)
    train_df, test_df = df.iloc[tr], df.iloc[te]
    yte = test_df[C.TARGET]

    early = EncodedModel(len(C.CLASS_ORDER)).fit(train_df[views["all"]], train_df[C.TARGET])
    late = LateFusion(views, modalities).fit(train_df)

    rows = []

    def record(strategy, dropped, y_pred, y_proba):
        r = evaluate(yte, y_pred, y_proba)
        r.update(strategy=strategy, dropped=dropped or "none")
        rows.append(r)
        return r

    # ------------------------------------------------------------- intact
    base_early = record("early", None, early.predict(test_df[views["all"]]),
                        early.predict_proba(test_df[views["all"]]))
    base_late = record("late", None, late.predict(test_df), late.predict_proba(test_df))
    print(f"  intact   early macroF1={base_early['macro_f1']:.4f}   "
          f"late macroF1={base_late['macro_f1']:.4f}")

    # ------------------------------------------- one modality at a time lost
    print("\n  DEGRADATION UNDER MODALITY LOSS (model NOT retrained)")
    for m in modalities:
        # early fusion: the columns simply stop arriving -> NaN
        Xe = test_df[views["all"]].copy()
        Xe[views[m]] = np.nan
        re = record("early", m, early.predict(Xe), early.predict_proba(Xe))
        rl = record("late", m, late.predict(test_df, drop=(m,)),
                    late.predict_proba(test_df, drop=(m,)))
        de = re["macro_f1"] - base_early["macro_f1"]
        dl = rl["macro_f1"] - base_late["macro_f1"]
        print(f"    lose {m:<8} early {re['macro_f1']:.4f} ({de:+.4f})   "
              f"late {rl['macro_f1']:.4f} ({dl:+.4f})   "
              f"{'LATE degrades less' if dl >= de else 'EARLY degrades less'}")

    # ------------------------------------------------------- WAN-loss case
    # The scenario the architecture is built for: the substation is cut off from
    # the control centre, so anything that only exists upstream is gone, and
    # only what is local to the substation keeps working.
    print("\n  WAN PARTITION (only substation-local evidence survives)")
    wan_lost = [m for m in modalities if m in ("snort", "control")]
    if wan_lost:
        Xe = test_df[views["all"]].copy()
        for m in wan_lost:
            Xe[views[m]] = np.nan
        re = record("early", "+".join(wan_lost), early.predict(Xe), early.predict_proba(Xe))
        rl = record("late", "+".join(wan_lost), late.predict(test_df, drop=tuple(wan_lost)),
                    late.predict_proba(test_df, drop=tuple(wan_lost)))
        print(f"    lose {'+'.join(wan_lost):<14} early {re['macro_f1']:.4f}   "
              f"late {rl['macro_f1']:.4f}")
        print("    -> this is the number that matters for the A/B/C replay")
        print("       harness in step 6: config A and B have nothing left here.")

    res = pd.DataFrame(rows)
    res.to_csv(C.OUTPUT_DIR / "e5_fusion_degradation.csv", index=False)
    return res
