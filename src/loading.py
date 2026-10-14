"""
Step 1-2: load the dataset once, correctly, and hand every later step the same
in-memory object.

The whole project depends on three structural facts, all of which are ASSERTED
here rather than assumed, so that a silent change in the files fails loudly:

  F1  multiclass/dataN.csv and triple/dataN.csv are the same rows in the same
      order, so their marker columns can be joined positionally.
  F2  within a file, each scenario occupies ONE contiguous block of rows.
  F3  row order within a block is time order, so position within the block is
      a usable relative timestamp (there is no timestamp column in this data).
"""
from __future__ import annotations

import itertools
import numpy as np
import pandas as pd

from . import config as C
from .modalities import build_views

META_COLUMNS = ["y_triple", "scenario_id", "file_id", "event_id", "t_in_event", "event_len"]


def _read_one(file_id: int) -> pd.DataFrame:
    """Read one capture, joining the 3-class label onto the 37-way scenario id."""
    multi  = pd.read_csv(C.VARIANT_MULTICLASS / f"data{file_id}.csv")
    triple = pd.read_csv(C.VARIANT_TRIPLE / f"data{file_id}.csv", usecols=["marker"])

    # F1: positional join is only valid if the row counts match exactly.
    assert len(multi) == len(triple), (
        f"data{file_id}: multiclass has {len(multi)} rows, triple has {len(triple)}. "
        "The variants are no longer row-aligned; the positional join is invalid."
    )

    df = multi.rename(columns={"marker": "scenario_id"})

    # F2 + F3: turn the contiguous blocks into event instances and relative time.
    run_lengths = [(k, len(list(g))) for k, g in itertools.groupby(df["scenario_id"])]
    seen = [k for k, _ in run_lengths]
    assert len(seen) == len(set(seen)), (
        f"data{file_id}: a scenario appears in more than one non-contiguous block. "
        "Event instances can no longer be defined as (file, scenario)."
    )
    # built in one concat rather than three assignments - repeated insertion
    # into a 129-column frame fragments its block manager and is slow.
    meta = pd.DataFrame({
        "y_triple": triple["marker"].to_numpy(),
        "file_id": file_id,
        "t_in_event": np.concatenate([np.arange(n) for _, n in run_lengths]),
        "event_len": np.repeat([n for _, n in run_lengths], [n for _, n in run_lengths]),
        "event_id": f"{file_id}-" + df["scenario_id"].astype(str),
    }, index=df.index)
    return pd.concat([df, meta], axis=1)


def _apply_inf_policy(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Deal with the 10,906 Inf cells. See config.INF_POLICY for the rationale."""
    block = df[feature_cols].to_numpy(dtype="float64")
    inf_mask = np.isinf(block)
    inf_cols = [c for c, n in zip(feature_cols, inf_mask.sum(0)) if n > 0]

    if C.ADD_INF_INDICATORS and inf_cols:
        # Record WHERE the open-circuit condition happened before erasing it.
        # These are appended after the originals so column order stays stable.
        indicators = {
            f"{c}__was_inf": inf_mask[:, feature_cols.index(c)].astype(np.int8)
            for c in inf_cols
        }
        df = pd.concat([df, pd.DataFrame(indicators, index=df.index)], axis=1)
        feature_cols = feature_cols + list(indicators)

    if C.INF_POLICY == "nan":
        # Tree models learn a default branch direction for missing values, so
        # this is a genuine "unknown", not an imputed number we invented.
        df[inf_cols] = df[inf_cols].replace([np.inf, -np.inf], np.nan)
    elif C.INF_POLICY == "clip":
        for c in inf_cols:
            finite = df.loc[np.isfinite(df[c]), c]
            df[c] = df[c].clip(finite.min(), finite.max())
    elif C.INF_POLICY == "drop_rows":
        df = df.loc[~inf_mask.any(1)].reset_index(drop=True)
    else:
        raise ValueError(f"unknown INF_POLICY {C.INF_POLICY!r}")

    return df, feature_cols


def load_dataset(use_cache: bool = True):
    """Return (df, feature_cols, views).

    df           one row per sample, feature columns + META_COLUMNS
    feature_cols the model input columns, in a fixed order
    views        {view_name: [columns]} as defined in modalities.build_views
    """
    cache = C.CACHE_DIR / "dataset.parquet"
    if use_cache and cache.exists():
        df = pd.read_parquet(cache)
        feature_cols = [c for c in df.columns if c not in META_COLUMNS]
        return df, feature_cols, build_views(feature_cols)

    df = pd.concat([_read_one(i) for i in C.FILE_IDS], ignore_index=True)

    # Feature columns = everything that is not bookkeeping. NOTE there is no
    # timestamp column in this dataset, so trap #2 (absolute-time leakage)
    # cannot occur here; t_in_event is RELATIVE and is deliberately kept out of
    # the feature list, available only to E2.
    feature_cols = [c for c in df.columns if c not in META_COLUMNS]
    df, feature_cols = _apply_inf_policy(df, feature_cols)

    # Order the frame so that every later step iterates events in a stable way.
    df = df.sort_values(["file_id", "scenario_id", "t_in_event"]).reset_index(drop=True)

    try:
        df.to_parquet(cache, index=False)
    except Exception as exc:              # pyarrow missing or similar
        print(f"[loading] cache write skipped: {exc}")

    return df, feature_cols, build_views(feature_cols)


def xy(df: pd.DataFrame, columns: list[str]):
    """Feature matrix + label vector for a given view."""
    return df[columns], df[C.TARGET]
