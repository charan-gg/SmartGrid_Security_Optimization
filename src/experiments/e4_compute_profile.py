"""
E4 - COMPUTE PROFILE.
Architectural parameter it fixes: which model can physically live in which tier.

A Tier 1 edge sentinel sits on bay-level hardware and must answer inside its latency
budget for every message. A Tier 2 correlator has a substation server. So for each
candidate configuration we measure inference latency per message (p50/p95/p99),
model size and batch throughput.

Design rule: the detector is ADVISORY, never protective. Protection operates in
20-100 ms; a detector that cannot meet its own budget does not sit in the protection
path, it gets demoted a tier.

LATENCY IS HARDWARE-SPECIFIC. Re-run this on the machine you will quote
(python step4_run_experiments.py --only e4) and state the CPU in the report.
"""
import pickle
import platform
import time

import numpy as np
import pandas as pd

from .. import config as C
from ..models import EncodedModel, backend_name
from ..splits import blocked_single

TIER_BUDGET_MS = {"tier1": 100.0, "tier2": 1000.0, "tier3": 10000.0}


def _time_predict(model, X, n_repeat=200, batch_size=1):
    """Wall-clock latency per predict_proba() call, ms. Batch size 1 is the honest edge
    case: a sentinel scores one message as it arrives."""
    samples, n = [], len(X)
    for i in range(n_repeat):
        chunk = X.iloc[(i * batch_size) % max(1, n - batch_size):][:batch_size]
        t0 = time.perf_counter()
        model.predict_proba(chunk)
        samples.append((time.perf_counter() - t0) * 1000.0)
    a = np.array(samples)
    return {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)), "mean_ms": float(a.mean())}


def run(df, views, kstar_features=None):
    tr, te = blocked_single(df)
    train_df, test_df = df.iloc[tr], df.iloc[te]

    configs = {"tier2_zone_fused": views["all"], "tier3_widearea_fused": views["all"]}
    if kstar_features:
        configs["tier1_edge_kstar"] = list(kstar_features)
    configs["tier1_edge_process_only"] = views["process"]

    rows = []
    for name, cols in configs.items():
        model = EncodedModel().fit(train_df[cols], train_df[C.TARGET])
        single = _time_predict(model, test_df[cols], batch_size=1)
        batch = _time_predict(model, test_df[cols], n_repeat=50, batch_size=128)
        size = len(pickle.dumps(model.model))
        budget = TIER_BUDGET_MS[name.split("_")[0]]
        rows.append({"config": name, "tier": name.split("_")[0], "n_features": len(cols),
                     "p50_ms": single["p50_ms"], "p95_ms": single["p95_ms"], "p99_ms": single["p99_ms"],
                     "batch128_per_sample_ms": batch["mean_ms"] / 128, "model_bytes": size,
                     "budget_ms": budget, "fits_budget": single["p95_ms"] <= budget})
        print(f"  {name:<24} n={len(cols):>3}  p50={single['p50_ms']:6.3f}ms  "
              f"p95={single['p95_ms']:6.3f}ms  p99={single['p99_ms']:6.3f}ms  "
              f"size={size / 1024:7.1f} KiB  budget={budget:.0f}ms  "
              f"{'OK' if single['p95_ms'] <= budget else 'OVER BUDGET'}")
    res = pd.DataFrame(rows).set_index("config")
    print(f"\n  backend: {backend_name()}   cpu: {platform.processor() or platform.machine()}   "
          f"python {platform.python_version()}")
    print("  Measured on THIS machine, not substation hardware. Quote the RATIO between tiers as")
    print("  the durable finding and re-run on the machine you cite.")
    res.to_csv(C.OUTPUT_DIR / "e4_compute_profile.csv")
    return res
