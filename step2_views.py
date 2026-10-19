"""
STEP 2 - the layer views: which columns belong to which layer of the substation, and
how much signal each layer carries before any model is fitted.

The views are hypotheses about the architecture (src/modalities.py); E1 tests them.
This step only shows the membership and a per-class profile of each layer's columns,
so a reader can see WHY a layer is expected to matter for a class.

Run:  python step2_views.py
"""
import pandas as pd

from src import config as C
from src.data import load_dataset
from src.modalities import classify_column


def main():
    print("=" * 78)
    print("STEP 2 - LAYER VIEWS")
    print("=" * 78)
    df, feature_cols, views = load_dataset()
    pd.DataFrame({"column": feature_cols, "layer": [classify_column(c) for c in feature_cols]}) \
        .to_csv(C.OUTPUT_DIR / "step2_view_map.csv", index=False)

    print("\n[1] VIEW MEMBERSHIP")
    for name, cols in views.items():
        print(f"  {name:<12} {len(cols):>3} columns  e.g. {', '.join(cols[:4])}{' ...' if len(cols) > 4 else ''}")

    print("\n[2] PER-CLASS PROFILE: share of rows where the column is non-zero / flagged")
    flags = ["state_cb", "recentChange", "consistency", "Cs", "any_relay"]
    prof = df.groupby(C.TARGET)[flags].agg(lambda s: (s != 0).mean()).reindex(C.CLASS_ORDER)
    print(prof.round(2).to_string())

    print("\n[3] PER-CLASS MEDIANS of the protocol timing columns")
    print(df.groupby(C.TARGET)[["sqNum", "stnum", "sqDiff", "stDiff", "timeLastMsg"]]
          .median().reindex(C.CLASS_ORDER).round(1).to_string())
    print("\n  Read these as hypotheses for E1: protocol-timing columns are where a forged or replayed")
    print("  message should look different; MU angles describe the physics, which an attacker who")
    print("  replays or forges GOOSE frames may leave untouched.")


if __name__ == "__main__":
    main()
