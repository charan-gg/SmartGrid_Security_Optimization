"""
STEP 7 - default vs optimized smart grid: performance and security, with a verdict that can say no.

  default    radial lines (cheapest connected network), a PMU in every substation, one control
             centre, everything streamed to it                                   [naive]
  default+   the same spend as the optimized grid, but untargeted: shortest extra corridors,
             a standby control centre                                            [budget-matched]
  optimized  N-1 line augmentation, minimum PMU set, k-medoid zones, tiered detection

Detector recalls are MEASURED (step 6, Train.csv). The grids are SIMULATED (synthetic layouts and
standard test cases) - the Kaggle data has no topology. See src/design/compare.py for the model
and the verdict rules, which are fixed before any result is looked at.

    python step6_train_detectors.py          # first
    python step7_compare_grids.py
    python step7_compare_grids.py --require-measured --sensitivity
    python step7_compare_grids.py --sizes 40 --seeds 10         # quick look
"""
import argparse
import json
import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from src import config as C
from src import detectors as DT
from src.design import cases as CS
from src.design import compare as CP
from src.design.params import (default_assumptions, dataset_fingerprint, load_measured,
                               provenance_problems)

DEFAULTS = ("naive", "budget-matched", "bm-lines+tiered")      # last one is a CONTROL, see compare.py
RULES = [
    "per metric : 'better'/'worse' only if the whole 95% interval of (optimized - default) is beyond the tolerance;",
    "             'tie' if the whole interval is inside +-tolerance; otherwise 'inconclusive'",
    "per family : counts of those outcomes; price is never netted against security",
    "security   : OBJECTIVES (what the algorithms minimise: bridges, zone share, routes) show the algorithm worked;",
    "             OUTCOMES (attack detection that survives a failure) are what was never optimised",
    "overall    : OPTIMIZED BETTER needs >=1 security OUTCOME win, 0 security losses, 0 performance losses",
    "             ('narrow' unless most outcome metrics are clear wins); otherwise NOT ESTABLISHED + reasons",
]


TAG = {"performance": "performance", "security-outcome": "SEC-OUTCOME", "security-objective": "sec-objective",
       "security-secondary": "sec-second.", "price": "price", "info": "info"}


def fmt(x, key):
    if key.startswith(("peak_link", "max_node_cores")):
        return f"{x:.4f}"
    if key.startswith(("cov_", "worst_node", "blast", "lateral")):
        return f"{x:.3f}"
    if key.startswith("cost"):
        return f"{x / 1e3:,.0f}k"
    return f"{x:,.1f}"


def print_verdict(v, default, tally, label, reasons):
    print(f"\n  OPTIMIZED vs {default.upper()}")
    print(f"  {'':<12}{'metric':<64}{'optimized':>10}{'default':>10}{'diff':>9}   95% interval        outcome")
    for fam in ("performance", "security-outcome", "security-objective", "security-secondary", "price", "info"):
        for _, r in v[v.family == fam].iterrows():
            d = f"{100 * r['diff']:+.1f}%" if r["relative"] else f"{r['diff']:+.3f}"
            ci = (f"[{100 * r.ci_lo:+.1f}%, {100 * r.ci_hi:+.1f}%]" if r["relative"]
                  else f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]")
            print(f"  {TAG[fam]:<12}{r.label[:62]:<64}{fmt(r.optimized, r.metric):>10}"
                  f"{fmt(r.baseline, r.metric):>10}{d:>9}   {ci:<19} {r.outcome}")
    print(f"\n  => {CP.verdict_text(default, v, tally, label, reasons)}")


def stale_check(require):
    problems = []
    try:
        meta = json.loads((C.OUTPUT_DIR / "step6_detectors.json").read_text())
        if meta.get("dataset") != dataset_fingerprint():
            problems.append("step6 recalls were produced from a DIFFERENT dataset than the one in dataset/")
    except (OSError, ValueError):
        problems.append("outputs/step6_detectors.json missing")
    try:
        t6 = (C.OUTPUT_DIR / "step6_tier_recall.csv").stat().st_mtime
        t3 = (C.OUTPUT_DIR / "e3_feature_stability.csv").stat().st_mtime
        if t6 < t3:
            problems.append("step6 recalls are older than the E3 feature set they depend on; re-run step 6")
    except OSError:
        problems.append("outputs/step6_tier_recall.csv or e3_feature_stability.csv missing")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="*", default=list(C.ENSEMBLE_SIZES))
    ap.add_argument("--seeds", type=int, default=C.ENSEMBLE_SEEDS)
    ap.add_argument("--require-measured", action="store_true")
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--no-cases", action="store_true")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    A, M = default_assumptions(), load_measured()
    bad = provenance_problems(M) + [("step6", p) for p in stale_check(args.require_measured)]
    if bad:
        for n, p in bad:
            print(f"  ! {n}: {p}")
        if args.require_measured:
            sys.exit(2)
        print("  ! continuing on non-fresh inputs; the numbers below are NOT reportable.\n")

    table, draws = DT.load_recall_draws()
    rec = CP.Recalls.from_draws(draws)
    print("=" * 100 + "\nDEFAULT vs OPTIMIZED SMART GRID\n" + "=" * 100)
    print(f"  detector recall (MEASURED, step 6, FPR budget {C.OPERATING_FPR:.0%}): station {rec.station.mean():.3f}, "
          f"fused {rec.fused.mean():.3f}  ({rec.B} block-bootstrap replicates)")
    print(f"  layouts (SIMULATED): synthetic sites, sizes {args.sizes} x {args.seeds} seeds")
    for line in RULES:
        print("  " + line)

    # ---------------------------------------------------------------- ensemble
    t0 = time.time()
    layouts, meta = [], []
    for n in args.sizes:
        for sd in range(args.seeds):
            res, _ = CP.run_layout(A, M, rec, n, sd)
            layouts.append(res)
            meta.append((n, sd))
    print(f"  {len(layouts)} layouts evaluated in {time.time() - t0:.0f}s")

    long = []
    for (n, sd), res in zip(meta, layouts):
        for cand, m in res.items():
            for k, v in m.items():
                long.append(dict(n_sites=n, seed=sd, candidate=cand, metric=k, value=float(np.mean(v))))
    pd.DataFrame(long).to_csv(C.OUTPUT_DIR / "step7_layouts.csv", index=False)

    verdicts, tallies, summary = [], [], {}
    for default in DEFAULTS:
        v = CP.verdict_table(layouts, A, default)
        tally = CP.family_tally(v)
        label, reasons = CP.overall(tally)
        print_verdict(v, default, tally, label, reasons)
        v.insert(0, "scope", "ensemble")
        verdicts.append(v)
        tally.insert(0, "default", default)
        tallies.append(tally)
        summary[default] = dict(overall=label, reasons=reasons, text=CP.verdict_text(default, v, tally.drop(columns="default"), label, reasons))
    pd.concat(verdicts).to_csv(C.OUTPUT_DIR / "step7_verdict.csv", index=False)
    pd.concat(tallies).to_csv(C.OUTPUT_DIR / "step7_tally.csv", index=False)

    # does the answer depend on grid size?
    rows = []
    for n in args.sizes:
        sub = [l for l, (m, _) in zip(layouts, meta) if m == n]
        for default in DEFAULTS:
            v = CP.verdict_table(sub, A, default)
            tl = CP.family_tally(v)
            lab, why = CP.overall(tl)
            row = dict(n_sites=n, default=default, overall=lab, reasons="; ".join(why))
            for _, r in v[v.tallied].iterrows():
                row[r.metric] = r.outcome
            rows.append(row)
    by_size = pd.DataFrame(rows)
    by_size.to_csv(C.OUTPUT_DIR / "step7_by_size.csv", index=False)
    print("\n  BY GRID SIZE: " + "   ".join(f"{r.n_sites} sites vs {r.default}: {r.overall}" for r in by_size.itertuples()))

    # can the machinery say no? run it on cases where the right answer is known
    aa = CP.verdict_table(layouts, A, "optimized", J=400)
    rev = CP.verdict_table([{"optimized": l["naive"], "naive": l["optimized"]} for l in layouts], A, "naive", J=400)
    print("\n  SELF-CHECK (the verdict must be able to say no)")
    for name, v, want in (("optimized vs ITSELF", aa, "all ties -> NOT ESTABLISHED"),
                          ("roles SWAPPED (naive plays 'optimized')", rev, "default better -> NOT ESTABLISHED")):
        tl = CP.family_tally(v)
        lab, why = CP.overall(tl)
        c = v[v.tallied].outcome.value_counts().to_dict()
        print(f"    {name:<42} -> {lab:<16} {c}   [expected: {want}]")
        summary.setdefault("self_check", {})[name] = dict(overall=lab, counts=c)

    # which half of the design does the work?
    names = sorted({k for l in layouts for k in l})
    attr_keys = ["spof_total", "min_routes_to_cc", "blast_share", "cov_delivered_single", "cov_delivered_regional",
                 "cov_delivered_n2_targeted", "cov_delivered_worst", "cost_total_kusd"]
    attr = pd.DataFrame({k: {c: float(np.mean([np.mean(l[c][k]) for l in layouts])) for c in names}
                         for k in attr_keys}).loc[["naive", "budget-matched", "opt-lines+central", "radial+tiered", "bm-lines+tiered", "optimized"]]
    attr.to_csv(C.OUTPUT_DIR / "step7_attribution.csv")
    print("\n  WHICH HALF DOES THE WORK? (mean over all layouts)")
    print("  lines = optimized N-1 line design; deployment = tiered detection + minimum PMU set")
    print(attr.to_string(float_format=lambda x: f"{x:,.3f}" if abs(x) < 1000 else f"{x:,.0f}"))

    # ---------------------------------------------------------- strict sensitivity of the model itself
    if args.sensitivity:
        print("\n  SENSITIVITY (n=40, 12 layouts each): does the verdict survive different assumptions?")
        variants = {"base": (A, True),
                    "no credit for PMU-inferred observability": (A, False),
                    "regional outage 25 km": (replace(A, failure_radius_km=25.0), True),
                    "regional outage 100 km": (replace(A, failure_radius_km=100.0), True),
                    "lines cost 0.5x": (replace(A, line_kusd_per_km=A.line_kusd_per_km * 0.5), True),
                    "lines cost 2x": (replace(A, line_kusd_per_km=A.line_kusd_per_km * 2.0), True),
                    "fibre cost 4x": (replace(A, fibre_kusd_per_km=A.fibre_kusd_per_km * 4.0), True),
                    "slower hardware (5x)": (replace(A, hardware_slowdown=5.0), True),
                    "detector recall = lower 95% bound": (A, True)}
        rows = []
        for name, (Av, credit) in variants.items():
            r_use = rec
            if name.startswith("detector"):
                lo = np.quantile(rec.station, 0.025)
                r_use = CP.Recalls(station=np.full(1, lo), fused=np.full(1, max(lo, np.quantile(rec.fused, 0.025))))
            lays = [CP.run_layout(Av, M, r_use, 40, sd, credit_inferred=credit)[0] for sd in range(12)]
            row = dict(variant=name)
            for default in DEFAULTS:
                v = CP.verdict_table(lays, Av, default, J=800)
                lab, why = CP.overall(CP.family_tally(v))
                row[f"vs_{default}"] = lab
                sec = v.family.isin(["security-outcome", "security-objective"]) & v.tallied
                row[f"{default}_security_better"] = int((sec & (v.outcome == "optimized better")).sum())
                row[f"{default}_security_worse"] = int((sec & (v.outcome == "default better")).sum())
                row[f"{default}_performance_worse"] = int(((v.family == "performance") & (v.outcome == "default better")).sum())
                for m, short in (("cov_intact", "intact_cov"), ("latency_ms", "latency")):
                    r = v[v.metric == m].iloc[0]
                    row[f"{default}_{short}_diff"] = round(float(r["diff"]), 4)
                    row[f"{default}_{short}_outcome"] = r["outcome"]
                if why:
                    row[f"{default}_why"] = "; ".join(why)
            rows.append(row)
            print(f"    {name:<44} vs naive: {row['vs_naive']:<17} vs budget-matched: {row['vs_budget-matched']:<24} vs control: {row['vs_bm-lines+tiered']}")
            print(f"    {'':<44} intact coverage {row['naive_intact_cov_diff']:+.3f} ({row['naive_intact_cov_outcome']}), "
                  f"latency {row['naive_latency_diff']:+.1f} ms ({row['naive_latency_outcome']}) vs naive"
                  + (f"   <- {row['naive_why']}" if 'naive_why' in row else ""))
        pd.DataFrame(rows).to_csv(C.OUTPUT_DIR / "step7_sensitivity.csv", index=False)

    # ----------------------------------------------------------------- standard test cases
    if not args.no_cases:
        rows = []
        print("\n  STANDARD TEST GRIDS (one layout each; interval = detector bootstrap only)")
        print("  CAUTION: these coordinates are schematic and N-1 is topological. The optimized lines are NOT checked")
        print("  against the case's loads or line ratings, so 'cheaper than as-built' is NOT a finding. Read the")
        print("  security-outcome and single-point-of-failure columns; ignore the cost column for as-built.")
        point = CP.Recalls.from_draws(draws)
        for name in CS.available():
            case = CS.load_case(name)
            t = time.time()
            res, _ = CP.run_layout(A, M, point, len(case.sites), 0, sites=case.sites, case=case)
            row = dict(case=name, n_sites=len(case.sites), as_built_lines=len(case.as_built))
            for default in ("naive", "budget-matched", "as-built"):
                v = CP.verdict_table([res], A, default, J=400)
                tl = CP.family_tally(v)
                lab, why = CP.overall(tl)
                row[f"vs_{default}"] = lab
                row[f"{default}_why"] = "; ".join(why)
                for m in ("cov_delivered_worst", "spof_total", "cost_total_kusd"):
                    r = v[v.metric == m].iloc[0]
                    row[f"{default}_{m}"] = f"{fmt(r.optimized, m)} vs {fmt(r.baseline, m)} ({r.outcome})"
            rows.append(row)
            print(f"    {name:<10} {len(case.sites):>3} sites  vs naive: {row['vs_naive']:<17} vs budget-matched: "
                  f"{row['vs_budget-matched']:<17} vs as-built: {row['vs_as-built']:<17} ({time.time() - t:.0f}s)")
            if not row["vs_as-built"].startswith("OPTIMIZED BETTER"):
                print(f"               as-built: {row['as-built_why']}")
        pd.DataFrame(rows).to_csv(C.OUTPUT_DIR / "step7_cases.csv", index=False)

    # ------------------------------------------------------------------------------- figure
    if not args.no_figure:
        try:
            plot(pd.concat(verdicts), attr)
        except ImportError:
            print("  (matplotlib missing: figure skipped)")

    summary["inputs"] = dict(dataset=dataset_fingerprint(), operating_fpr=C.OPERATING_FPR,
                             recall_station=float(rec.station.mean()), recall_fused=float(rec.fused.mean()),
                             layouts=len(layouts), sizes=args.sizes, seeds=args.seeds,
                             provenance_problems=[f"{n}: {p}" for n, p in bad])
    summary["limits"] = [
        "grids are simulated: synthetic layouts and standard test cases with schematic coordinates",
        "N-1 is topological (connectivity), not a power-flow check",
        "detector recall is per message and from one testbed substation; cross-substation fusion gain is not credited",
        "attacks and failures are scenario sets, not a threat model",
        "costs are relative kUSD assumptions (config.GRID_ASSUMPTIONS)"]
    (C.OUTPUT_DIR / "step7_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\n  LIMITS")
    for s in summary["limits"]:
        print("   - " + s)


def plot(v, attr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    import textwrap
    t = v[v.tallied].copy()
    fig, axes = plt.subplots(1, len(DEFAULTS), figsize=(6.2 * len(DEFAULTS), 9), sharey=True)
    colour = {"optimized better": "#2a9d8f", "tie": "#8d99ae", "default better": "#e76f51", "inconclusive": "#e9c46a"}
    for ax, default in zip(axes, DEFAULTS):
        s = t[t.default == default].iloc[::-1]
        y = np.arange(len(s))
        sign = np.where(s.higher_is_better, 1, -1)       # right = optimized better
        tol = np.where(s.relative, s.tolerance, s.tolerance)
        mid, lo, hi = s["diff"] * sign / tol, s.ci_lo * sign / tol, s.ci_hi * sign / tol
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        ax.axvspan(-1, 1, color="#eee", zorder=0)
        for i, (_, r) in enumerate(s.iterrows()):
            ax.plot([lo.iloc[i], hi.iloc[i]], [i, i], color=colour[r.outcome], lw=3)
            ax.plot(mid.iloc[i], i, "o", color=colour[r.outcome])
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_yticks(y)
        ax.set_yticklabels([textwrap.fill(l, 46) for l in s.label], fontsize=8)
        ax.set_xticks([-10, -1, 0, 1, 10, 100])
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}"))
        ax.set_title(f"optimized vs {default}", fontsize=11)
        ax.set_xlabel("difference in units of tolerance  (right = optimized better; grey band = tie)")
    fig.suptitle("Default vs optimized grid: every scored metric (95% interval)", fontsize=12)
    fig.tight_layout()
    fig.savefig(C.OUTPUT_DIR / "step7_scorecard.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
