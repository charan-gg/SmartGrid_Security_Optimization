"""
Splitting, done so that the numbers are honest.

Consecutive messages in this capture are near-duplicates of each other (a heartbeat
GOOSE frame every second barely changes the MU angles), and Train.csv / Test.csv hold
the SAME messages. So a shuffled-row split, and the shipped Train -> Test split, both
put copies of every message on both sides and score a model for memorising them.

  blocked      the honest default: the capture is cut into contiguous time blocks
               and whole blocks are held out (StratifiedGroupKFold on `block`)
  naive_rows   the dishonest one, computed ONLY so the report can show the gap
"""
from __future__ import annotations

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from . import config as C


def blocked(df, n_splits: int = None):
    """Yield (train_idx, test_idx); no time block is on both sides, and every
    fold keeps all five classes (stratified on the label)."""
    n_splits = n_splits or C.N_SPLITS
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=C.RANDOM_SEED)
    return splitter.split(df, df[C.TARGET], groups=df[C.GROUP_KEY])


def blocked_single(df):
    """One (train_idx, test_idx) pair: the first blocked fold. For experiments that do
    not need full cross-validation (E4 timing)."""
    return next(blocked(df))


def naive_rows(df, n_splits: int = None):
    """The WRONG split, kept deliberately: shuffled rows, no grouping."""
    n_splits = n_splits or C.N_SPLITS
    return StratifiedKFold(n_splits=n_splits, shuffle=True,
                           random_state=C.RANDOM_SEED).split(df, df[C.TARGET])


SPLITTERS = {"blocked": blocked, "naive_rows": naive_rows}


def describe(df) -> dict:
    """Sanity numbers about the grouping structure."""
    per_block = df.groupby(C.GROUP_KEY).size()
    present = df.groupby(C.GROUP_KEY)[C.TARGET].nunique()
    return {
        "n_rows": len(df),
        "n_blocks": int(per_block.size),
        "rows_per_block_min": int(per_block.min()),
        "rows_per_block_median": int(per_block.median()),
        "rows_per_block_max": int(per_block.max()),
        "blocks_containing_all_classes": int((present == len(C.CLASS_ORDER)).sum()),
        "class_counts": df[C.TARGET].value_counts().to_dict(),
    }
