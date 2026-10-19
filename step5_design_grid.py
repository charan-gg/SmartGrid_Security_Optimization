"""
STEP 5 - SYNTHESISE the grid instead of fitting models to a given one.

Steps 1-4 took a fixed hierarchy (bay / substation / wide-area) and measured
which model suits each level. This step builds the grid itself with graph
algorithms and reads the levels off the result:

  A  power network     Kruskal MST backbone + greedy augmentation to N-1 secure
  B  PMU placement     minimum dominating set (exact integer program)
  C  hierarchy         k-medoid zoning on all-pairs Dijkstra latency; smallest
                       k that satisfies blast-radius / hop / size / latency / compute
  D  stress tests      BFS failure sweeps, lateral movement, max-flow path diversity
  E  comparison        centralised (A, B) vs tiered (C) on the same physical grid

Measured inputs (message rate, E1 layer coverage, E3 feature budget, E4 latencies) are read
from outputs/*.csv|json, which steps 1-4 wrote from dataset/Train.csv. Everything else is an
ASSUMPTION in config.GRID_ASSUMPTIONS. If the dataset files are newer than an output, that
output is STALE; the INPUTS table marks each input as measured / placeholder / stale, and
--require-measured refuses to design from anything that is not a fresh measurement:

    python step5_design_grid.py --require-measured    # the run to quote in the report

    python step5_design_grid.py                       # default 40-substation region
    python step5_design_grid.py --n 80 --seed 7
    python step5_design_grid.py --sites my_region.csv # id,x_km,y_km,kind,mw
    python step5_design_grid.py --sensitivity         # also re-design under changed assumptions
    python step5_design_grid.py --scaling             # also 20/40/80/120 substations x 5 seeds
"""
import argparse
import json
import time
from dataclasses import replace

from src import config as C
from src.design import design as D
from src.design.params import (dataset_fingerprint, default_assumptions, load_measured,
                               provenance_problems)
from src.design.sites import load_sites_csv


def banner(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def table(rows, cols, widths=None, indent="  "):
    if not rows:
        return
    widths = widths or [max(len(str(c)), *(len(str(r.get(c, ""))) for r in rows)) for c in cols]
    print(indent + "  ".join(str(c).ljust(w) for c, w in zip(cols, widths)))
    for r in rows:
        print(indent + "  ".join(str(r.get(c, "")).ljust(w) for c, w in zip(cols, widths)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="substations (synthetic region)")
    ap.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    ap.add_argument("--sites", default=None, help="CSV: id,x_km,y_km,kind,mw")
    ap.add_argument("--pmu-k", type=int, default=None, help="k-fold PMU observability")
    ap.add_argument("--slowdown", type=float, default=None,
                    help="substation CPU vs the E4 dev machine (default from config)")
    ap.add_argument("--zone-share", type=float, default=None, help="max share of bays per hub")
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--scaling", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--require-measured", action="store_true",
                    help="exit 2 unless every input is a fresh measurement from the current dataset")
    args = ap.parse_args()

    A = default_assumptions()
    ov = {}
    if args.n:
        ov["n_sites"] = args.n
        ov["region_km"] = A.region_km * (args.n / A.n_sites) ** 0.5
    if args.pmu_k:
        ov["pmu_obs_k"] = args.pmu_k
    if args.slowdown:
        ov["hardware_slowdown"] = args.slowdown
    if args.zone_share:
        ov["max_zone_share"] = args.zone_share
    A = replace(A, **ov)
    M = load_measured()
    sites = load_sites_csv(args.sites) if args.sites else None
    t0 = time.perf_counter()

    banner("INPUTS - what was measured vs what is assumed")
    table([dict(name=n, value=(f"{v:.4g}" if isinstance(v, float) else v), source=p)
           for n, v, p in M.table()], ["name", "value", "source"])
    ds = dataset_fingerprint()
    print(f"\n  dataset fingerprint (Train.csv + Test.csv): {ds or 'dataset files not found'}")
    problems = provenance_problems(M)
    for n, p in problems:
        why = ("is a PLACEHOLDER; the output that measures it is missing"
               if str(p).startswith("FALLBACK") else "is STALE: its output is older than the dataset files")
        print(f"  !! {n}: {why}")
    if problems and args.require_measured:
        print("\n  --require-measured: refusing to design from inputs that are not fresh "
              "measurements.\n  Re-run steps 1-4 (python step4_run_experiments.py) on the "
              "current dataset, then this step.")
        raise SystemExit(2)
    print("\n  Assumptions (config.GRID_ASSUMPTIONS) - NOT measured, relative kUSD:")
    print("   " + ", ".join(f"{k}={v}" for k, v in A.__dict__.items()))

    d = D.design_grid(A, M, seed=args.seed, sites=sites)
    s = D.summarize(d)

    # ---------------------------------------------------------------- A
    banner("A  PHYSICAL NETWORK -> Kruskal MST + N-1 augmentation")
    print(f"  {s['n_substations']} substations, {s['n_bays']} bays "
          f"(line terminals + generator/load ties)\n")
    table([dict(variant=k, lines=v["lines"], km=v["km"], cost_kusd=v["cost_kusd"],
                bridges=v["bridges"], cut_vertices=v["cut_vertices"],
                worst_node_loss_islands=v["worst_node_outage_islanded"],
                diameter_hops=v["diameter_hops"]) for k, v in d.power.audit.items()],
          ["variant", "lines", "km", "cost_kusd", "bridges", "cut_vertices",
           "worst_node_loss_islands", "diameter_hops"])
    p = s["power"]
    print(f"\n  -> N-1 security costs +{p['n1_premium_over_radial_pct']}% over the radial "
          f"MST ({p['lines_added_beyond_mst']} extra lines) and is "
          f"{abs(p['cost_vs_mesh_pct'])}% cheaper than building every candidate line.")
    print("     N-1 is TOPOLOGICAL (grid stays connected); line capacity under "
          "rerouting is not checked.")

    # ---------------------------------------------------------------- B
    banner("B  PMU PLACEMENT -> minimum dominating set")
    q = s["pmu"]
    print(f"  {q['k_fold']}-fold observability of {s['n_substations']} substations")
    print(f"  standard rule : {q['n_standard']} PMUs   ({q['method']})")
    print(f"  greedy        : {q['n_greedy']} PMUs   (optimality gap "
          f"{q['n_greedy'] - q['n_standard']})")
    print(f"  + zero-inject.: {q['n_pmus']} PMUs   = {q['fraction_of_substations']:.0%} of "
          f"substations (PMU-everywhere baseline: {s['n_substations']})")
    print(f"  E1-weighted expected attack coverage: {q['expected_detection']:.3f} intact, "
          f"{q['worst_after_single_pmu_loss']:.3f} after the worst single-PMU loss")
    print(f"  E1 coverage: station bus alone {M.det_station:.3f}, + MU phasors {M.det_all:.3f}, "
          f"phasors alone {M.det_process:.3f}: "
          f"{'sensor placement buys little detection' if M.det_all - M.det_station < 0.05 else 'phasors add detection'}")
    print("\n  !! E1 was measured on a testbed with every merging unit present. Whether detection holds")
    print("     on buses observed only by inference is NOT in the data (see observability.py).")

    # ---------------------------------------------------------------- C
    banner("C  HIERARCHY -> k-medoid zoning, smallest feasible k")
    table([dict(k=r["k"], feasible=r["feasible"], max_share=r["max_zone_share"],
                stations_min_max=f"{r['min_zone_stations']}-{r['max_zone_stations']}",
                max_hops=r["max_hops"], t_hub_ms=r["t_hub_verdict_ms"],
                hub_cores=r["max_hub_cores"], violations=r["violations"])
           for r in d.k_sweep],
          ["k", "feasible", "max_share", "stations_min_max", "max_hops", "t_hub_ms",
           "hub_cores", "violations"])
    hh = s["hierarchy"]
    print(f"\n  -> {hh['k_zones']} zone hub(s) => {hh['n_levels']} levels: "
          f"{' > '.join(hh['levels'])}")
    print(f"     hubs at substations {hh['hubs']}, control centre at {hh['control_centre']}")
    print(f"     forced by: {', '.join(hh['binding_constraints']) or 'nothing (k=1 feasible)'}")
    print("     slack at the chosen design:")
    for k_, v in hh["headroom"].items():
        print(f"       {k_:<24} {v}")
    if not hh["feasible"]:
        print("  !! NO k satisfied every constraint; showing the least-violating design.")

    if d.security:
        # ------------------------------------------------------------ D
        banner("D  STRESS TESTS -> BFS failure sweeps, lateral movement, max-flow")
        sec = d.security
        def line(tag, res):
            print(f"  {tag}")
            for lab, key in (("centralised, one CC", "central"),
                             (f"centralised + backup CC (site {sec['backup_cc']})", "central_backup"),
                             ("tiered, hub/CC-level fusion", "tiered_fused")):
                print(f"    {lab:<44} mean {res[key]['mean']:.3f}   worst {res[key]['worst']:.3f}")
        print("  share of surviving bays that still reach a fusion point:")
        line("single-substation loss (every substation in turn)", sec["node"])
        line(f"regional outage ({A.failure_radius_km:.0f} km disc around every substation)", sec["region"])
        print("  (tiered 'any-level' coverage is 1.0 by architecture: each station keeps its own")
        print("   correlator. How accurate station-only detection is, is what step 6 measures.)")
        lat, gw, dv = sec["lateral"], sec["gateways"], sec["diversity"]
        print(f"\n  lateral movement from one compromised substation: flat network reaches "
              f"{lat['flat']:.0%} of bays;")
        print(f"    zoned reaches {lat['zoned_mean']:.1%} on average (worst {lat['zoned_max']:.1%})")
        print(f"  inter-zone links (authenticated crossings, rule R6): {gw['edge_cut']} of "
              f"{d.comm.m} ({gw['cut_fraction']:.0%});  border gateway stations: "
              f"{len(gw['border_stations'])}")
        print(f"  independent routes to the CC (Menger): min {dv['min_edge_disjoint_to_cc']} "
              f"edge-disjoint, min {dv['min_vertex_disjoint_to_cc']} vertex-disjoint")

        # ------------------------------------------------------------ E
        banner("E  DEPLOYMENT COMPARISON -> same grid, different place for detection")
        table(d.comparison, ["deployment", "cc_ingress_KBps", "peak_link_KBps",
                             "peak_link_util", "max_node_cores", "worst_bay_to_verdict_ms",
                             "blast_share", "single_node_loss_coverage_worst",
                             "regional_loss_coverage_worst", "attack_coverage_intact"])
        cmp_ = {r["deployment"].split()[0]: r for r in d.comparison}
        cB, cC, cA = cmp_["B"], cmp_["C"], cmp_["A"]
        if cB["peak_link_util"] < 0.01:
            print(f"\n  -> link load is NOT binding: even raw centralised fusion peaks at "
                  f"{cB['peak_link_util']:.2%} of a corridor link at the measured "
                  f"{M.sample_rate_hz:.1f} rows/s per station.")
        else:
            print(f"\n  -> peak corridor load {cB['peak_link_KBps'] / max(1e-9, cC['peak_link_KBps']):.1f}x "
                  f"lower than raw centralised fusion.")
        print(f"     What tiering buys is blast radius: one fusion node serves at most "
              f"{cC['blast_share']:.0%} of bays (centralised: {cB['blast_share']:.0%}), "
              f"and the worst single-node loss keeps {cC['single_node_loss_coverage_worst']:.0%} "
              f"of bays covered (centralised, one CC: {cB['single_node_loss_coverage_worst']:.0%}).")
        print(f"     Pipeline latency is NOT where tiering wins: centralised "
              f"{cB['worst_bay_to_verdict_ms']} ms vs tiered {cC['worst_bay_to_verdict_ms']} ms "
              f"(one row period is {M.sample_period_ms:.0f} ms of that; see --sensitivity).")
        print(f"     Detection (E1 attack coverage): fused layers {M.det_all:.3f}, "
              f"station bus alone {M.det_station:.3f}, phasors alone {M.det_process:.3f}. "
              f"A physical-only design keeps {cA['attack_coverage_intact']:.3f}; "
              f"B and C use the same inputs, so C ~= B on accuracy.")

    # ---------------------------------------------------------------- F
    banner("F  ROLE -> MODEL MAP (roles read off the designed graph)")
    table(d.roles, ["level", "role", "count", "model", "features", "p95_ms", "load",
                    "bytes_out_Bps", "note"])

    banner("G  COST ACCOUNTS (relative kUSD, ASSUMED unit costs)")
    for k_, v in d.costs.items():
        print(f"  {k_:<20} {v:>14,.1f}")
    print(f"  detection plane (sensing + compute, fibre excluded: it is shared with SCADA) "
          f"= {100 * d.costs['detection_plane'] / d.costs['power_lines']:.2f}% of the line cost")

    D.write_outputs(d, C.OUTPUT_DIR, plot=not args.no_plot)
    (C.OUTPUT_DIR / "step5_inputs.json").write_text(json.dumps(dict(
        dataset=ds, all_inputs_fresh=not problems, require_measured=args.require_measured,
        problems=[dict(input=n, provenance=p) for n, p in problems],
        inputs=[dict(name=n, value=v, provenance=p) for n, v, p in M.table()],
        seed=args.seed), indent=2, default=str), encoding="utf-8")
    if args.sensitivity:
        banner("SENSITIVITY - re-design under changed assumptions")
        rows = D.sensitivity(A, M, seed=args.seed, sites=sites)
        table(rows, ["parameter", "value", "k_zones", "n_levels", "feasible", "binding",
                     "n_pmus", "max_hub_cores", "max_station_cores", "latency_central_ms",
                     "latency_tiered_ms", "edge_rate_cap_hz"])
        D.write_csv(C.OUTPUT_DIR / "step5_sensitivity.csv", rows)
    if args.scaling:
        banner("SCALING - same method on 20/40/80/120 substations x 5 seeds")
        rows = D.scaling(A, M)
        D.write_csv(C.OUTPUT_DIR / "step5_scaling.csv", rows)
        import statistics as st
        agg = []
        for n in sorted({r["n_substations"] for r in rows}):
            rs = [r for r in rows if r["n_substations"] == n]
            f = lambda key: round(st.mean(r[key] for r in rs), 2)
            agg.append(dict(n=n, bays=f("n_bays"), n1_premium_pct=f("n1_premium_pct"),
                            pmu_fraction=f("pmu_fraction"), k_zones=f("k_zones"),
                            k_min=min(r["k_zones"] for r in rs), k_max=max(r["k_zones"] for r in rs),
                            feasible=f"{sum(r['feasible'] for r in rs)}/{len(rs)}",
                            B_over_C_peak_link=f("B_over_C_peak_link"),
                            B_peak_util=f("B_peak_link_util"),
                            regional_worst_central=f("regional_worst_central"),
                            regional_worst_tiered=f("regional_worst_tiered")))
        table(agg, list(agg[0].keys()))

    print(f"\nDONE in {time.perf_counter() - t0:.1f}s. Outputs: outputs/step5_*")


if __name__ == "__main__":
    main()
