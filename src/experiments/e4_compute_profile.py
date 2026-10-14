"""
E4 - COMPUTE PROFILE.
Architectural parameter it fixes: which model can physically live in which tier.

A Tier 1 edge sentinel sits on bay-level hardware and must answer inside its
latency budget every sample. A Tier 2 correlator has a substation server. So we
measure, for each candidate configuration: inference latency per sample
(p50/p95/p99), model size on disk, and throughput.

Design rule R1 from the brief: the detector is ADVISORY, never protective.
Protection operates in 20-100 ms; if a detector cannot meet its own budget it
does not get to sit in the protection path - it gets demoted a tier. That is
the decision this experiment feeds.
"""
import pickle
import time

import numpy as np
import pandas as pd

from .. import config as C
from ..models import EncodedModel, backend_name
from ..splits import by_event_single

# Tier latency budgets from the architecture proposal, in milliseconds.
TIER_BUDGET_MS = {"tier1_edge": 100.0, "tier2_zone": 1000.0, "tier3_widearea": 10000.0}


def _time_predict(model, X, n_repeat=200, batch_size=1):
    """Wall-clock latency per predict() call, in ms. Batch size 1 is the honest
    edge case: a sentinel scores one sample as it arrives, it cannot wait to
    accumulate a batch without spending its whole latency budget waiting."""
    samples = []
    n = len(X)
    for i in range(n_repeat):
        chunk = X.iloc[(i * batch_size) % max(1, n - batch_size):][:batch_size]
        t0 = time.perf_counter()
        model.predict_proba(chunk)
        samples.append((time.perf_counter() - t0) * 1000.0)
    a = np.array(samples)
    return {"p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)),
            "mean_ms": float(a.mean())}


def run(df, views, kstar_features=None):
    tr, te = by_event_single(df)
    train_df, test_df = df.iloc[tr], df.iloc[te]

    # The configurations the architecture actually proposes.
    configs = {
        "tier2_zone_fused": views["all"],
        "tier3_widearea_fused": views["all"],
    }
    if kstar_features:
        configs["tier1_edge_kstar"] = list(kstar_features)
    configs["tier1_edge_pmu_only"] = views["pmu"]

    rows = []
    for name, cols in configs.items():
        model = EncodedModel(len(C.CLASS_ORDER)).fit(train_df[cols], train_df[C.TARGET])
        single = _time_predict(model, test_df[cols], batch_size=1)
        batch = _time_predict(model, test_df[cols], n_repeat=50, batch_size=128)
        size_bytes = len(pickle.dumps(model.model))

        tier = name.split("_")[0]
        budget = TIER_BUDGET_MS.get(
            {"tier1": "tier1_edge", "tier2": "tier2_zone", "tier3": "tier3_widearea"}[tier])
        rows.append({
            "config": name, "tier": tier, "n_features": len(cols),
            "p50_ms": single["p50_ms"], "p95_ms": single["p95_ms"], "p99_ms": single["p99_ms"],
            "batch128_per_sample_ms": batch["mean_ms"] / 128,
            "model_bytes": size_bytes,
            "budget_ms": budget,
            "fits_budget": single["p95_ms"] <= budget,
        })
        print(f"  {name:<24} n={len(cols):>4}  "
              f"p50={single['p50_ms']:6.3f}ms  p95={single['p95_ms']:6.3f}ms  "
              f"p99={single['p99_ms']:6.3f}ms  size={size_bytes/1024:8.1f} KiB  "
              f"budget={budget:.0f}ms  {'OK' if single['p95_ms'] <= budget else 'OVER BUDGET'}")

    res = pd.DataFrame(rows).set_index("config")
    print(f"\n  backend: {backend_name()}")
    print("  CAVEAT for the report: these are measured on the development")
    print("  machine, not on substation hardware. They are an upper bound on")
    print("  throughput and a lower bound on latency. State the CPU model, and")
    print("  present the RATIO between tiers as the durable finding rather than")
    print("  the absolute milliseconds.")
    res.to_csv(C.OUTPUT_DIR / "e4_compute_profile.csv")
    return res
