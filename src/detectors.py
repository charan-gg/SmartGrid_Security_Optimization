"""
The three detector tiers the grid comparison needs, and their recall WITH error bars.

The Kaggle data cannot say anything about power-grid topology. What it can say is how well a
detector sees an attack given the evidence available at a point in the grid. The comparison
(src/design/compare.py) therefore takes three recall numbers from here and lets the graph
decide, for every failure scenario, which substation gets which one:

  sentinel  the E3 fold-stable features only (what a bay-level edge model sees)
  station   goose + consistency + relay layers: a station correlator WITHOUT raw phasors
  fused     every layer, incl. merging-unit phasors: needs a PMU-class sensor on the process bus

How each number is obtained (identical for all three, so differences are differences in evidence):

  1. out-of-fold P(Masquerade)+P(Injection)+P(Replay) for every message, blocked time CV
     (a message is always scored by a model that never saw its time block)
  2. ONE alert threshold per tier, chosen so that OPERATING_FPR of the non-attack messages
     (Normal + Fault) would be flagged - not the threshold that happened to maximise F1
  3. attack coverage = mean recall over the three attack classes at that threshold (the same
     definition metrics.attack_coverage uses everywhere else)
  4. uncertainty: a BLOCK bootstrap over the 20 contiguous time blocks. The threshold is
     re-derived inside every replicate, because it is defined by the false-alert budget; a
     fixed threshold would understate the spread. 41 Fault and 100 Replay messages are few.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .experiments.e6_operating_point import oof_attack_score

TIERS = ("sentinel", "station", "fused")


# ------------------------------------------------------------------ pure functions
def stable_features(output_dir=None):
    """Features E3 selected in EVERY fold: the sentinel's input."""
    path = (output_dir or C.OUTPUT_DIR) / "e3_feature_stability.csv"
    if not path.exists():
        raise SystemExit(f"missing {path}: it comes from step 4 (E3), which this step builds on.\n"
                         "  run it first:  python step4_run_experiments.py      (or: python run_pipeline.py --only 4)")
    s = pd.read_csv(path, index_col=0).iloc[:, 0]
    return [f for f, n in s.items() if int(n) == C.N_SPLITS]


def tier_columns(views, stable):
    if not stable:
        raise ValueError("E3 produced no fold-stable features; run step4 --only e3 first")
    return {"sentinel": list(stable), "station": list(views["station"]), "fused": list(views["all"])}


def threshold_at_fpr(score, is_negative, fpr):
    """Smallest threshold whose alert rate on negatives is <= fpr (alert = score >= thr).
    Ties are resolved conservatively: the realised FPR is never above the budget."""
    neg = np.sort(np.asarray(score)[np.asarray(is_negative, dtype=bool)])
    if len(neg) == 0:
        raise ValueError("no negative messages to set a threshold on")
    allowed = int(np.floor(fpr * len(neg)))
    if allowed >= len(neg):
        return -np.inf
    return float(np.nextafter(neg[len(neg) - allowed - 1], np.inf))


def recalls_at(score, y, threshold, classes=None):
    """{class: share of that class's messages with score >= threshold}; NaN if absent."""
    classes = classes or C.ATTACK_CLASSES
    score, y = np.asarray(score), np.asarray(y)
    out = {}
    for c in classes:
        m = y == c
        out[c] = float((score[m] >= threshold).mean()) if m.any() else float("nan")
    return out


def coverage(recalls):
    """Mean recall over the attack classes present (nan-safe)."""
    vals = [v for v in recalls.values() if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def point_estimate(score, y, fpr):
    is_neg = np.isin(y, ["Normal", "Fault"])
    thr = threshold_at_fpr(score, is_neg, fpr)
    rec = recalls_at(score, y, thr)
    realised = float((np.asarray(score)[is_neg] >= thr).mean())
    return dict(threshold=thr, realised_fpr=realised, coverage=coverage(rec),
                **{f"recall_{c}": rec[c] for c in rec})


def block_bootstrap(score, y, block, fpr, B, seed):
    """B replicates, each resampling whole time blocks with replacement and re-deriving the
    threshold on the resampled negatives. Returns a DataFrame of per-replicate results."""
    score, y, block = np.asarray(score), np.asarray(y), np.asarray(block)
    ids = np.unique(block)
    rows_of = {b: np.flatnonzero(block == b) for b in ids}
    rng = np.random.default_rng(seed)
    out = []
    for b in range(B):
        pick = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([rows_of[k] for k in pick])
        try:
            res = point_estimate(score[idx], y[idx], fpr)
        except ValueError:          # a resample with no negatives at all (cannot happen with 20 blocks, but be safe)
            continue
        out.append(dict(b=b, **res))
    return pd.DataFrame(out)


def attack_score(model, X):
    """P(Masquerade) + P(Injection) + P(Replay) from a fitted 5-class EncodedModel."""
    att = [C.CLASS_ORDER.index(c) for c in C.ATTACK_CLASSES]
    return model.predict_proba(X)[:, att].sum(axis=1)


# ------------------------------------------------------------------ the data-bound part
def score_tiers(df, views, stable):
    """{tier: out-of-fold attack score} for the whole capture."""
    return {t: oof_attack_score(df, cols) for t, cols in tier_columns(views, stable).items()}


def summarise(df, scores, fpr=None, B=None, seed=None):
    """(table, draws): one row per tier with point estimate + 95% block-bootstrap interval,
    and the long table of every replicate (the compare step re-uses it, paired by replicate)."""
    fpr = C.OPERATING_FPR if fpr is None else fpr
    B = C.BOOTSTRAP_B if B is None else B
    seed = C.RANDOM_SEED if seed is None else seed
    y, block = df[C.TARGET].to_numpy(), df[C.GROUP_KEY].to_numpy()
    rows, draws = [], []
    for tier in TIERS:
        pt = point_estimate(scores[tier], y, fpr)
        bs = block_bootstrap(scores[tier], y, block, fpr, B, seed)      # same seed => same block draws per tier
        lo, hi = bs["coverage"].quantile([0.025, 0.975])
        rows.append(dict(tier=tier, **pt, coverage_lo=float(lo), coverage_hi=float(hi),
                         coverage_sd=float(bs["coverage"].std())))
        bs.insert(0, "tier", tier)
        draws.append(bs)
    return pd.DataFrame(rows), pd.concat(draws, ignore_index=True)


def load_recall_draws(output_dir=None):
    """(table, draws) written by step 6. Raises FileNotFoundError with the fix if absent."""
    d = output_dir or C.OUTPUT_DIR
    try:
        return pd.read_csv(d / "step6_tier_recall.csv"), pd.read_csv(d / "step6_recall_draws.csv")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{e.filename} missing - run `python step6_train_detectors.py` first") from e
