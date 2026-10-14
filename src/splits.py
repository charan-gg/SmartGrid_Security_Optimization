"""
Step 3: splitting, done so that the numbers are honest.

Trap #1 in the project brief: never random-split rows. Consecutive rows inside
one event are near-duplicates of each other (a PMU at 120 Hz barely changes in
8 ms), so a shuffled split puts almost-identical rows on both sides and the
model scores ~0.99 by memorising the event, not by detecting the attack.

Three splitters live here, and the difference between their scores IS a result:
  by_event     the honest default for E1-E7 (no event straddles the boundary)
  by_scenario  E8, the hardest test (whole scenarios unseen in training)
  naive_rows   the dishonest one, computed ONLY so the report can show the gap
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import (
    GroupKFold,
    StratifiedGroupKFold,
    StratifiedShuffleSplit,
    GroupShuffleSplit,
)

from . import config as C


def by_event(df, n_splits: int = None):
    """Yield (train_idx, test_idx) with no event instance on both sides.

    Stratified on the label so every fold keeps all three classes, grouped on
    event_id so no event straddles the split. This is the default for E1-E7.
    """
    n_splits = n_splits or C.N_SPLITS
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=C.RANDOM_SEED
    )
    return splitter.split(df, df[C.TARGET], groups=df[C.GROUP_KEY])


def by_event_single(df, test_size: float = 0.25):
    """One (train_idx, test_idx) pair grouped by event. For quick experiments
    that do not need full cross-validation (E4 timing, E7 attribution)."""
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=C.RANDOM_SEED
    )
    return next(splitter.split(df, df[C.TARGET], groups=df[C.GROUP_KEY]))


def by_scenario(df, n_splits: int = None):
    """E8: hold out WHOLE scenarios. The model is asked about an attack variant
    it has literally never seen, which is what deploying to a new site is like.

    Not stratified, because a held-out scenario belongs entirely to one class by
    construction; stratifying would defeat the point.
    """
    n_splits = n_splits or C.N_SPLITS
    return GroupKFold(n_splits=n_splits).split(
        df, df[C.TARGET], groups=df[C.SCENARIO_KEY]
    )


def naive_rows(df, n_splits: int = None):
    """The WRONG split, kept deliberately. Shuffled rows, no grouping. Its score
    is reported beside the honest score to quantify how much of the published
    benchmark literature on this dataset is event memorisation."""
    n_splits = n_splits or C.N_SPLITS
    splitter = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=1 / n_splits, random_state=C.RANDOM_SEED
    )
    return splitter.split(df, df[C.TARGET])


SPLITTERS = {"by_event": by_event, "by_scenario": by_scenario, "naive_rows": naive_rows}


def describe(df) -> dict:
    """Sanity numbers about the grouping structure, for Appendix A."""
    return {
        "n_rows": len(df),
        "n_events": df[C.GROUP_KEY].nunique(),
        "n_scenarios": df[C.SCENARIO_KEY].nunique(),
        "n_files": df["file_id"].nunique(),
        "rows_per_event_min": int(df.groupby(C.GROUP_KEY).size().min()),
        "rows_per_event_median": int(df.groupby(C.GROUP_KEY).size().median()),
        "rows_per_event_max": int(df.groupby(C.GROUP_KEY).size().max()),
        "class_counts": df[C.TARGET].value_counts().to_dict(),
    }
