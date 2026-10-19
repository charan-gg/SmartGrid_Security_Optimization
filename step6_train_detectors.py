"""
STEP 6 - train the three tier detectors and measure their recall with error bars.

Output feeds step 7 (default vs optimized grid). Nothing here knows about topology.

    python step6_train_detectors.py
    python step6_train_detectors.py --no-models        # skip writing models/*.joblib
"""
import argparse
import json

import joblib

from src import config as C
from src import detectors as DT
from src.data import load_dataset, rates
from src.design.params import dataset_fingerprint
from src.models import EncodedModel, backend_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-models", action="store_true")
    args = ap.parse_args()

    df, feature_cols, views = load_dataset()
    stable = DT.stable_features()
    cols = DT.tier_columns(views, stable)
    r = rates(df)
    print(f"backend {backend_name()} | {len(df):,} messages | operating FPR {C.OPERATING_FPR:.0%} of "
          f"non-attack | block bootstrap B={C.BOOTSTRAP_B} over {df[C.GROUP_KEY].nunique()} time blocks")
    for t in DT.TIERS:
        print(f"  {t:<9} {len(cols[t]):>2} features" + (f"  {cols[t]}" if t == "sentinel" else ""))

    scores = DT.score_tiers(df, views, stable)
    table, draws = DT.summarise(df, scores)
    table.to_csv(C.OUTPUT_DIR / "step6_tier_recall.csv", index=False)
    draws.to_csv(C.OUTPUT_DIR / "step6_recall_draws.csv", index=False)

    per_day = r["negative_rate_hz"] * 86400
    print("\n  tier       thr      real.FPR  false/day/unit  coverage [95% block-bootstrap]   Mas    Inj    Rep")
    for _, t in table.iterrows():
        print(f"  {t['tier']:<9} {t['threshold']:.3f}   {t['realised_fpr']:.4f}   {t['realised_fpr'] * per_day:>9,.0f}      "
              f"{t['coverage']:.3f}  [{t['coverage_lo']:.3f}, {t['coverage_hi']:.3f}]      "
              f"{t['recall_Masquerade']:.2f}   {t['recall_Injection']:.2f}   {t['recall_Replay']:.2f}")

    # paired differences: does adding evidence help beyond noise?
    wide = draws.pivot(index="b", columns="tier", values="coverage")
    print("\n  paired bootstrap differences in coverage (same block resamples):")
    for a, b in (("station", "sentinel"), ("fused", "station")):
        d = (wide[a] - wide[b]).dropna()
        lo, hi = d.quantile([0.025, 0.975])
        verdict = "real" if lo > 0 else ("negative" if hi < 0 else "NOT distinguishable from zero")
        print(f"    {a:>8} - {b:<9} {d.mean():+.3f}  [{lo:+.3f}, {hi:+.3f}]  -> {verdict}")

    models = []
    if not args.no_models:
        C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        for t in DT.TIERS:
            m = EncodedModel().fit(df[cols[t]], df[C.TARGET])
            row = table[table.tier == t].iloc[0]
            path = C.MODELS_DIR / f"{t}.joblib"
            joblib.dump(dict(tier=t, model=m, columns=cols[t], threshold=float(row["threshold"]),
                             operating_fpr=C.OPERATING_FPR, backend=backend_name(),
                             dataset=dataset_fingerprint()), path)
            models.append(str(path.relative_to(C.PROJECT_ROOT)))
        print(f"\n  final models (fitted on all of Train.csv) -> {', '.join(models)}")
        print("  their threshold comes from the OUT-OF-FOLD scores; a model fitted on more data is a")
        print("  little more confident, so re-check false alerts on a fresh capture before deploying.")

    summary = dict(dataset=dataset_fingerprint(), backend=backend_name(), operating_fpr=C.OPERATING_FPR,
                   bootstrap_B=C.BOOTSTRAP_B, seed=C.RANDOM_SEED, n_messages=len(df),
                   negative_rate_hz=r["negative_rate_hz"], sentinel_features=stable,
                   tiers=table.to_dict("records"), models=models,
                   limits=["recalls are per message, not per attack episode (the capture has none)",
                           "cross-substation fusion benefit is NOT measured here: one testbed substation"])
    (C.OUTPUT_DIR / "step6_detectors.json").write_text(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
