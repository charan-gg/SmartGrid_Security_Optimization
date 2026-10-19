"""
Load Train.csv / Test.csv once, correctly, and hand every later step the same
in-memory objects.

The project depends on structural facts about this data. They are ASSERTED here so
that a different download fails loudly instead of producing quietly wrong numbers:

  T1  Train and Test have the same columns in the same order
  T2  the label column holds exactly the five known classes in both files
  T3  every feature is numeric and there are no missing values
  T4  the capture clock (`time`) is a usable, non-degenerate timeline

What is MEASURED and reported (audit()) rather than assumed:
  * how much of Test is a copy of Train (the files turned out to be one recording)
  * whether the derived timing columns can be reproduced from the rows themselves
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .modalities import build_views

META_COLUMNS = [C.TARGET, C.FAMILY_COL, "t_ms", C.GROUP_KEY]


def _read(path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Put Train.csv and Test.csv from the Kaggle dataset in "
            f"{C.DATA_DIR}.")
    return pd.read_csv(path)


def _prepare(raw: pd.DataFrame, n_blocks: int = None) -> pd.DataFrame:
    """Raw CSV -> frame sorted by capture time, with metadata columns appended."""
    n_blocks = n_blocks or C.N_BLOCKS
    df = raw.sort_values(C.TIME_COL, kind="stable").reset_index(drop=True)
    t = df[C.TIME_COL].to_numpy(dtype="float64")
    span = max(t.max() - t.min(), 1.0)
    # blocks: equal-width contiguous slices of the capture clock
    block = np.minimum(((t - t.min()) / span * n_blocks).astype(int), n_blocks - 1)
    meta = pd.DataFrame({
        C.TARGET: df[C.LABEL_COL].to_numpy(),
        C.FAMILY_COL: df[C.LABEL_COL].map(C.FAMILY).to_numpy(),
        "t_ms": t,
        C.GROUP_KEY: block,
    }, index=df.index)
    feats = df.drop(columns=[C.LABEL_COL, C.TIME_COL])
    return pd.concat([feats, meta], axis=1)


def _check(train: pd.DataFrame, test: pd.DataFrame):
    assert list(train.columns) == list(test.columns), (
        "Train.csv and Test.csv have different columns; this project assumes the two "
        "files share one schema (T1).")
    for name, d in (("Train", train), ("Test", test)):
        got = set(d[C.LABEL_COL].unique())
        assert got == set(C.CLASS_ORDER), (
            f"{name}.csv classes are {sorted(got)}, expected {sorted(C.CLASS_ORDER)} (T2).")
        feats = d.drop(columns=[C.LABEL_COL])
        bad = [c for c in feats.columns if not pd.api.types.is_numeric_dtype(feats[c])]
        assert not bad, f"{name}.csv has non-numeric feature columns {bad} (T3)."
        assert not feats.isna().any().any(), f"{name}.csv has missing values (T3)."
        assert d[C.TIME_COL].nunique() > 100 and d[C.TIME_COL].min() >= 0, (
            f"{name}.csv `time` is not a usable timeline (T4).")


def load_published():
    """(train_df, test_df, feature_cols) exactly as shipped: two separate files."""
    tr_raw, te_raw = _read(C.TRAIN_CSV), _read(C.TEST_CSV)
    _check(tr_raw, te_raw)
    train, test = _prepare(tr_raw), _prepare(te_raw)
    feature_cols = [c for c in train.columns if c not in META_COLUMNS]
    return train, test, feature_cols


def load_dataset():
    """(df, feature_cols, views): Train.csv, time-ordered. This is the frame every
    experiment cross-validates inside (see config: blocked time CV)."""
    train, _, feature_cols = load_published()
    return train, feature_cols, build_views(feature_cols)


def rates(df: pd.DataFrame) -> dict:
    """Message rates MEASURED from the capture (they size bandwidth and alert budgets)."""
    t = df["t_ms"].to_numpy()
    dur_s = float(t.max() - t.min()) / 1000.0
    gaps = np.diff(t)
    gaps = gaps[gaps > 0]
    negatives = int(df[C.FAMILY_COL].isin(["Normal", "Natural"]).sum())
    return dict(
        duration_s=dur_s,
        row_rate_hz=len(df) / dur_s,
        negative_rate_hz=negatives / dur_s,         # Normal + Fault: what an alert would be false on
        median_gap_ms=float(np.median(gaps)),
        p95_gap_ms=float(np.percentile(gaps, 95)),
    )


def _stream_consistency(raw: pd.DataFrame) -> dict:
    """Fraction of rows whose derived timing columns equal what the rows themselves
    imply when read in time order. ~1.0 means the columns were computed on this very
    stream; much lower means they were computed on a different one."""
    s = raw.sort_values(C.TIME_COL, kind="stable")
    first = lambda x: x.iloc[0]
    gap = s[C.TIME_COL].diff().fillna(first(s[C.TIME_COL]))
    return dict(
        timeLastMsg=float((s["timeLastMsg"] == gap).mean()),
        sqDiff=float((s["sqDiff"] == s["sqNum"].diff().fillna(first(s["sqNum"]))).mean()),
        stDiff=float((s["stDiff"] == s["stnum"].diff().fillna(first(s["stnum"]))).mean()),
    )


def audit() -> dict:
    """How independent are Train.csv and Test.csv? Numbers, not assumptions."""
    tr, te = _read(C.TRAIN_CSV), _read(C.TEST_CSV)
    _check(tr, te)
    feat = [c for c in tr.columns if c not in (C.LABEL_COL, C.TIME_COL)]
    key = lambda d: set(d[feat + [C.LABEL_COL]].round(6).astype(str).agg("|".join, axis=1))
    k_tr, k_te = key(tr), key(te)
    pair = [C.TIME_COL, C.LABEL_COL]
    # (time, class) identifies a message in both files. Merging on time alone would
    # cross-pair the few messages that share a timestamp and fake small differences.
    one_to_one = not (tr.duplicated(pair).any() or te.duplicated(pair).any())
    merged = tr.merge(te, on=pair, suffixes=("_tr", "_te"),
                      validate="one_to_one" if one_to_one else "many_to_many")
    differ = pd.Series({c: float((merged[c + "_tr"] != merged[c + "_te"]).mean()) for c in feat})
    return dict(
        n_train=len(tr), n_test=len(te),
        test_messages_present_in_train=float(len(merged) / len(te)),   # same (time, class)
        exact_duplicate_test_rows=float(len(k_te & k_tr) / len(k_te)),
        class_counts_equal=bool(tr[C.LABEL_COL].value_counts().sort_index().equals(
            te[C.LABEL_COL].value_counts().sort_index())),
        columns_identical_in_both=[c for c, f in differ.items() if f == 0.0],
        columns_that_differ={c: round(f, 3) for c, f in differ[differ > 0].sort_values(ascending=False).items()},
        timing_reproducible_from_rows=dict(train=_stream_consistency(tr), test=_stream_consistency(te)),
    )


def xy(df: pd.DataFrame, columns: list[str]):
    """Feature matrix + label vector for a given view."""
    return df[columns], df[C.TARGET]
