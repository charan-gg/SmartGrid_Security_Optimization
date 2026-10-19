"""
STEP 4 - run E1, E3, E4, E6, E8.

Each experiment fixes one parameter of the architecture, so the output of this script
is the input to step 5 (designing the grid). Nothing tunes a model to look good: the
budget is frozen in config.MODEL_PARAMS and shared by every view, so differences are
attributable to evidence, not to effort. Everything cross-validates with blocked time
folds inside Train.csv (config.py explains why Test.csv is not used).

Run everything:      python step4_run_experiments.py
Run a subset:        python step4_run_experiments.py --only e1 e3
Re-time on a laptop: python step4_run_experiments.py --only e3 e4

Ordering note: E3 produces the stable feature set that E4 profiles as the edge model.
Running E4 alone profiles the edge model on the stable set saved by the last E3.
"""
import argparse
import json
import time

import pandas as pd

from src import config as C
from src.data import load_dataset
from src.models import backend_name
from src.experiments import (
    e1_views as e1,
    e3_feature_budget as e3,
    e4_compute_profile as e4,
    e6_operating_point as e6,
    e8_unseen_attack as e8,
)

ALL = ["e1", "e3", "e4", "e6", "e8"]
HEADINGS = {
    "e1": "E1 VIEW ABLATION                -> which layer feeds the detector, is fusion needed",
    "e3": "E3 MINIMAL FEATURE SET k*       -> conduit spec + bandwidth",
    "e4": "E4 COMPUTE PROFILE              -> tier assignment",
    "e6": "E6 OPERATING POINT              -> alarm budget",
    "e8": "E8 UNSEEN ATTACK TYPE           -> central vs per-site training",
}


def saved_stable_set():
    """The fold-stable features from the last E3 run, if any."""
    try:
        s = pd.read_csv(C.OUTPUT_DIR / "e3_feature_stability.csv", index_col=0).iloc[:, 0]
        return [f for f, n in s.items() if int(n) == C.N_SPLITS] or None
    except (OSError, ValueError, IndexError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=ALL, default=None)
    ap.add_argument("--skip", nargs="*", choices=ALL, default=[])
    args = ap.parse_args()
    selected = [k for k in (args.only or ALL) if k not in args.skip]

    print(f"model backend : {backend_name()}")
    print(f"experiments   : {', '.join(selected)}")
    df, feature_cols, views = load_dataset()
    print(f"loaded        : {len(df):,} rows, {len(feature_cols)} features, "
          f"{df[C.GROUP_KEY].nunique()} time blocks")

    timings, kstar_features = {}, None
    for key in selected:
        print("\n" + "=" * 78 + f"\n{HEADINGS[key]}\n" + "=" * 78)
        t0 = time.perf_counter()
        if key == "e1":
            e1.run(df, views)
        elif key == "e3":
            _, _, stable = e3.run(df, views)
            kstar_features = stable or None
        elif key == "e4":
            e4.run(df, views, kstar_features=kstar_features or saved_stable_set())
        elif key == "e6":
            e6.run(df, views)
        elif key == "e8":
            e8.run(df, views)
        timings[key] = round(time.perf_counter() - t0, 1)
        print(f"\n  [{key} finished in {timings[key]}s]")

    print("\n" + "=" * 78 + f"\nDONE. Results written to {C.OUTPUT_DIR}")
    for k, v in timings.items():
        print(f"  {k}: {v}s")
    with open(C.OUTPUT_DIR / "step4_timings.json", "w") as fh:
        json.dump(timings, fh, indent=2)


if __name__ == "__main__":
    main()
