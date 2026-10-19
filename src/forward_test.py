"""
FORWARD TEST - the closest thing to a held-out test set this capture allows.

Train.csv and Test.csv are ONE recording (every message of Test is in Train), so Test.csv cannot
be a test set. Time is the honest axis instead: fit on the EARLY part of the capture, then score
the LATE part exactly once.

    blocks 0 .. 13   train          (alert thresholds are set from out-of-fold scores in here)
    block  14        purge          (dropped: its first messages are near-duplicates of block 15's last)
    blocks 15 .. 19  test           (never seen, in time or by the threshold)

The blocked CV in steps 3-4 answers "how well does it generalise across time blocks, on average";
this answers "if I deploy the model trained on the first 70% of a capture, what does it do on what
comes next". It is one split, so it is noisy: counts are printed next to every rate. Fault has
8 messages in the test part - do not read its recall as a measurement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import detectors as DT
from .metrics import confusion, evaluate
from .models import EncodedModel

N_TEST_BLOCKS = 5
PURGE_BLOCKS = 1
VIEWS = ("all", "station", "process")


def split_forward(df, n_test_blocks=N_TEST_BLOCKS, purge_blocks=PURGE_BLOCKS):
    """(train, test) frames; every test block is later than every train block, with
    `purge_blocks` dropped in between."""
    blocks = sorted(df[C.GROUP_KEY].unique())
    if len(blocks) < n_test_blocks + purge_blocks + 2:
        raise ValueError(f"{len(blocks)} blocks is too few for a {n_test_blocks}-block forward test")
    test_b = blocks[-n_test_blocks:]
    train_b = blocks[:len(blocks) - n_test_blocks - purge_blocks]
    return (df[df[C.GROUP_KEY].isin(train_b)].reset_index(drop=True),
            df[df[C.GROUP_KEY].isin(test_b)].reset_index(drop=True))


def tier_test(train, test, cols, fpr):
    """One detector tier: threshold from the TRAIN part's out-of-fold scores, applied once to
    the test part. Returns a dict of rates with their counts."""
    y_tr, y_te = train[C.TARGET].to_numpy(), test[C.TARGET].to_numpy()
    oof = DT.oof_attack_score(train, cols)
    thr = DT.threshold_at_fpr(oof, np.isin(y_tr, ["Normal", "Fault"]), fpr)
    ref = DT.point_estimate(oof, y_tr, fpr)
    model = EncodedModel().fit(train[cols], train[C.TARGET])
    score = DT.attack_score(model, test[cols])
    neg = np.isin(y_te, ["Normal", "Fault"])
    fp = int((score[neg] >= thr).sum())
    rec = DT.recalls_at(score, y_te, thr)
    out = dict(threshold=thr, train_oof_coverage=ref["coverage"],
               test_coverage=DT.coverage(rec), test_false_alerts=fp, test_negatives=int(neg.sum()),
               test_fpr=fp / int(neg.sum()))
    for c in C.ATTACK_CLASSES:
        out[f"recall_{c}"] = rec[c]
        out[f"n_{c}"] = int((y_te == c).sum())
    return out


def five_class_test(train, test, cols):
    """Classic hold-out numbers for one view: 5-class macro-F1, per-class recall, confusion."""
    m = EncodedModel().fit(train[cols], train[C.TARGET])
    pred, proba = m.predict(test[cols]), m.predict_proba(test[cols])
    res = evaluate(test[C.TARGET], pred, proba)
    cm = pd.DataFrame(confusion(test[C.TARGET], pred), index=C.CLASS_ORDER, columns=C.CLASS_ORDER)
    return res, cm


def run(df, views, stable, fpr=None):
    """(tier rows, five-class rows, {view: confusion}) for the forward split of `df`."""
    fpr = C.OPERATING_FPR if fpr is None else fpr
    train, test = split_forward(df)
    cols = DT.tier_columns(views, stable)
    tiers = [dict(tier=t, **tier_test(train, test, cols[t], fpr)) for t in DT.TIERS]
    classes, cms = [], {}
    for v in VIEWS:
        res, cms[v] = five_class_test(train, test, views[v])
        classes.append(dict(view=v, **res))
    info = dict(n_train=len(train), n_test=len(test),
                train_blocks=sorted(train[C.GROUP_KEY].unique().tolist()),
                test_blocks=sorted(test[C.GROUP_KEY].unique().tolist()),
                test_class_counts=test[C.TARGET].value_counts().to_dict())
    return pd.DataFrame(tiers), pd.DataFrame(classes), cms, info
