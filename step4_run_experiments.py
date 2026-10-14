"""
STEP 4 - run E1..E8.

Each experiment fixes one parameter of the architecture, so the output of this
script is the input to step 5 (designing the tiers). Nothing here tunes a model
to look good; the budget is frozen in config.MODEL_PARAMS and shared by every
view so that differences are attributable to evidence, not to effort.

Run everything:      python step4_run_experiments.py
Run a subset:        python step4_run_experiments.py --only e1 e2
Skip the slow ones:  python step4_run_experiments.py --skip e3 e8

Ordering note: E3 produces k*, which E4 needs to profile the edge model, so E4
reuses E3's result when both run in the same invocation.
"""
import argparse
import json
import time

from src import config as C
from src.loading import load_dataset
from src.models import backend_name
from src.experiments import (
    e1_modality_ablation as e1,
    e2_lead_time as e2,
    e3_feature_budget as e3,
    e4_compute_profile as e4,
    e5_fusion_degradation as e5,
    e6_operating_point as e6,
    e7_attribution as e7,
    e8_scenario_holdout as e8,
)

ALL = ["e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8"]

HEADINGS = {
    "e1": "E1 MODALITY ABLATION            -> is a fusion point needed",
    "e2": "E2 OBSERVABILITY + LEAD TIME    -> which levels to instrument",
    "e3": "E3 MINIMAL FEATURE SET k*       -> conduit spec + bandwidth",
    "e4": "E4 COMPUTE PROFILE              -> tier assignment",
    "e5": "E5 FUSION + DEGRADATION         -> inter-tier contract",
    "e6": "E6 OPERATING POINT              -> alarm budget",
    "e7": "E7 PER-CLASS ATTRIBUTION        -> conduits to authenticate",
    "e8": "E8 SCENARIO-HELD-OUT            -> central vs federated",
}


def banner(key):
    print("\n" + "=" * 78)
    print(HEADINGS[key])
    print("=" * 78)


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
          f"{df[C.GROUP_KEY].nunique()} event instances")

    timings, kstar_features = {}, None

    for key in selected:
        banner(key)
        t0 = time.perf_counter()

        if key == "e1":
            e1.run(df, views)
        elif key == "e2":
            e2.run(df, views)
        elif key == "e3":
            _, kstar, stable = e3.run(df, views)
            kstar_features = stable or None
        elif key == "e4":
            e4.run(df, views, kstar_features=kstar_features)
        elif key == "e5":
            e5.run(df, views)
        elif key == "e6":
            e6.run(df, views)
        elif key == "e7":
            e7.run(df, views)
        elif key == "e8":
            e8.run(df, views)

        timings[key] = round(time.perf_counter() - t0, 1)
        print(f"\n  [{key} finished in {timings[key]}s]")

    print("\n" + "=" * 78)
    print("DONE. Results written to", C.OUTPUT_DIR)
    for k, v in timings.items():
        print(f"  {k}: {v}s")
    with open(C.OUTPUT_DIR / "step4_timings.json", "w") as fh:
        json.dump(timings, fh, indent=2)


if __name__ == "__main__":
    main()
