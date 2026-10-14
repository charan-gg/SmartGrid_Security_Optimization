"""E4 only, on THIS machine.

E4 measures inference latency, which is hardware-dependent, so unlike E1-E3 it
must not be run anywhere but the machine whose numbers you intend to quote.

k* features come from E3's fold-stability table: only features selected in ALL
5 folds are used, since a fold-unstable feature has no place in a conduit spec.
"""
import pandas as pd
from src.loading import load_dataset
from src.experiments import e4_compute_profile as e4
from src import config as C

stability = pd.read_csv(C.OUTPUT_DIR / "e3_feature_stability.csv", index_col=0)
stability.columns = ["folds"]
stable = list(stability[stability["folds"] == C.N_SPLITS].index)
print(f"k* stable features from E3 ({len(stable)}): {stable}\n")

df, feature_cols, views = load_dataset()
e4.run(df, views, kstar_features=stable)
