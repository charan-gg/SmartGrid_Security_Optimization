"""
Orchestrator: sites -> power grid -> PMUs -> comm graph + hierarchy -> stress
tests -> comparisons. `design_grid` returns one `GridDesign` object; the writers
turn it into outputs/step5_*.csv|json|png and `summarize` into the console report.

The order matters and each stage consumes only the previous stages' output:
  A power.synthesize_power_grid      lines (MST + N-1 augmentation)
  B observability.place_pmus         PMUs  (dominating set)
  C hierarchy.design_hierarchy       zones, hubs, CC  (k-medoid + constraints)
  D security.*                       failure / lateral-movement / path-diversity
  E evaluate.*                       A/B/C comparison, role->model map, costs
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import evaluate as EV
from . import hierarchy as HI
from . import observability as OB
from . import power as PW
from . import security as SE
from .sites import synth_sites


@dataclass
class GridDesign:
    A: object
    M: object
    seed: int
    sites: list
    power: PW.PowerGrid
    pmu: OB.PMUPlan
    zib: list
    comm: object
    hier: HI.Hierarchy
    k_sweep: list
    security: dict = field(default_factory=dict)     # empty when security=False
    comparison: list = field(default_factory=list)
    roles: list = field(default_factory=list)
    pmu_loss: dict = field(default_factory=dict)
    costs: dict = field(default_factory=dict)


def design_grid(A, M, seed=42, sites=None, security=True):
    sites = sites or synth_sites(A.n_sites, seed, A.region_km, A.frac_gen, A.frac_junction)
    power = PW.synthesize_power_grid(sites, A)
    zib = ([u for u in power.g.nodes() if power.g.node[u]["kind"] == "junction"]
           if A.use_zib else [])
    pmu = OB.place_pmus(power.g, A.pmu_obs_k, A.use_zib)
    comm = HI.build_comm_graph(power.g, A)
    hier, sweep = HI.design_hierarchy(comm, power.bays, A, M, seed=seed)
    d = GridDesign(A=A, M=M, seed=seed, sites=sites, power=power, pmu=pmu, zib=zib,
                   comm=comm, hier=hier, k_sweep=sweep)
    d.roles = EV.role_model_map(comm, power.bays, hier, M, A)
    d.pmu_loss = EV.pmu_loss_sweep(power.g, power.bays, pmu, zib, M)
    d.costs = EV.cost_accounts(power, pmu, comm, power.bays, hier, A)
    if security:
        backup = SE.backup_cc(comm, power.bays, hier, A.failure_radius_km)
        nrows, nsum = SE.single_node_sweep(comm, power.bays, hier, backup)
        rrows, rsum = SE.regional_sweep(comm, power.bays, hier, A.failure_radius_km, backup)
        d.security = dict(backup_cc=backup, node_rows=nrows, node=nsum, region_rows=rrows,
                          region=rsum,
                          lateral=SE.lateral_reach(comm, power.bays, hier),
                          gateways=SE.gateways(comm, hier),
                          diversity=SE.path_diversity(comm, hier))
        d.comparison = EV.compare_deployments(comm, power.bays, hier, M, A, nsum, rsum, backup)
    return d


# ------------------------------------------------------------------ summary
def binding_constraints(d):
    """Constraints that forced the chosen k: the violations of the next-smaller k."""
    if d.hier.k == 1:
        return []
    prev = next(r for r in d.k_sweep if r["k"] == d.hier.k - 1)
    return [v for v in prev["violations"].split(",") if v != "none"]


def headroom(d):
    A, m = d.A, d.hier.metrics
    return {
        "zone_share": f"{m['max_zone_share']:.3f} of cap {A.max_zone_share}",
        "zone_hops": f"{m['max_hops']} of cap {A.max_zone_hops}",
        "latency_to_hub_verdict": f"{m['t_hub_verdict_ms']:.1f} ms of {A.zone_budget_ms:.0f} ms",
        "latency_to_cc_verdict": f"{m['t_cc_verdict_ms']:.1f} ms of {A.wide_budget_ms:.0f} ms",
        "hub_cores": f"{m['max_hub_cores']:.2f} of cap {A.max_cores_per_node}",
        "station_cores": f"{m['max_station_cores']:.2f} of cap {A.max_cores_per_node}",
    }


def summarize(d):
    pa = d.power.audit
    rad, des, mesh = (pa["radial (MST only)"], pa["designed (N-1 secure)"],
                      pa["mesh (all candidate lines)"])
    return dict(
        seed=d.seed, n_substations=d.comm.n, n_bays=sum(d.power.bays.values()),
        power=dict(lines=des["lines"], km=des["km"], cost_kusd=des["cost_kusd"],
                   n1_premium_over_radial_pct=round(100 * (des["cost_kusd"] / rad["cost_kusd"] - 1), 1),
                   cost_vs_mesh_pct=round(100 * (des["cost_kusd"] / mesh["cost_kusd"] - 1), 1),
                   lines_added_beyond_mst=len(d.power.added_lines),
                   bridges=des["bridges"], cut_vertices=des["cut_vertices"],
                   diameter_hops=des["diameter_hops"]),
        pmu=dict(k_fold=d.pmu.k, n_pmus=d.pmu.n_final, n_standard=d.pmu.n_standard,
                 n_greedy=d.pmu.n_greedy, method=d.pmu.method,
                 fraction_of_substations=round(d.pmu.n_final / d.comm.n, 3),
                 expected_detection=round(d.pmu_loss["intact"], 4),
                 worst_after_single_pmu_loss=round(d.pmu_loss["worst_after_single_pmu_loss"], 4)),
        hierarchy=dict(k_zones=d.hier.k, n_levels=d.hier.n_levels, levels=d.hier.level_names(),
                       control_centre=d.hier.cc, hubs=d.hier.hubs,
                       binding_constraints=binding_constraints(d), headroom=headroom(d),
                       feasible=d.hier.feasible),
        costs=d.costs,
        security={k: v for k, v in d.security.items() if k in ("backup_cc", "node", "region",
                                                                "lateral", "diversity")}
        if d.security else {},
        gateways={k: v for k, v in d.security.get("gateways", {}).items()
                  if k in ("edge_cut", "cut_fraction", "border_stations")},
    )


# ------------------------------------------------------------------- writers
def write_csv(path, rows, cols=None):
    if not rows:
        return
    cols = cols or list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _jsonable(o):
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"not serialisable: {type(o)}")


def node_rows(d):
    g, h = d.power.g, d.hier
    obs = OB.observed_set(g, d.pmu.pmus, d.zib)
    gw = d.security.get("gateways", {}).get("border_stations", [])
    out = []
    for u in g.nodes():
        n = g.node[u]
        out.append(dict(id=u, x_km=round(n["x_km"], 2), y_km=round(n["y_km"], 2), kind=n["kind"],
                        mw=round(n["mw"], 1), degree=g.degree(u), bays=d.power.bays[u],
                        has_pmu=u in d.pmu.pmus, observed=u in obs,
                        zone_hub=h.zone_of[u], is_hub=u in h.hubs, is_cc=u == h.cc,
                        hops_to_hub=h.hops[u], latency_to_hub_ms=round(h.lat_ms[u], 3),
                        border_gateway=u in gw))
    return out


def edge_rows(d):
    h = d.hier
    lc = EV.link_loads(h, d.power.bays, d.M, d.A, "C")
    lb = EV.link_loads(h, d.power.bays, d.M, d.A, "B")
    out = []
    for u, v, a in d.power.g.edges():
        c = d.comm.edge(u, v)
        out.append(dict(u=u, v=v, length_km=round(a["length_km"], 2),
                        power_cost_kusd=round(a["cost_kusd"], 1),
                        fibre_cost_kusd=round(c["cost_kusd"], 1),
                        latency_ms=round(c["latency_ms"], 4),
                        inter_zone=h.zone_of[u] != h.zone_of[v],
                        added_for_n1=(u, v) in set(d.power.added_lines),
                        load_C_Bps=round(lc.get((u, v), 0.0), 1),
                        load_B_Bps=round(lb.get((u, v), 0.0), 1)))
    return out


def write_outputs(d, out_dir, plot=True):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "step5_grid_nodes.csv", node_rows(d))
    write_csv(out / "step5_grid_edges.csv", edge_rows(d))
    write_csv(out / "step5_k_sweep.csv", [dict(r, hubs=" ".join(map(str, r["hubs"]))) for r in d.k_sweep])
    write_csv(out / "step5_comparison.csv", d.comparison)
    write_csv(out / "step5_role_model_map.csv", d.roles)
    write_csv(out / "step5_physical_variants.csv",
         [dict(variant=k, **v) for k, v in d.power.audit.items()])
    write_csv(out / "step5_pmu_curve.csv",
         [dict(n_pmus=n, fraction_observed=round(f, 4)) for n, f in d.pmu.curve])
    if d.security:
        write_csv(out / "step5_failure_single_node.csv", d.security["node_rows"])
        write_csv(out / "step5_failure_regional.csv", d.security["region_rows"])
    with open(out / "step5_design_summary.json", "w") as fh:
        json.dump(dict(summary=summarize(d),
                       measured={n: (v, p) for n, v, p in d.M.table()},
                       assumptions=d.A.__dict__,
                       zones={str(h): m for h, m in d.hier.zones.items()}),
                  fh, indent=2, default=_jsonable)
    if plot:
        try:
            plot_grid(d, out / "step5_grid.png")
        except ImportError:
            pass


def plot_grid(d, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, g = d.hier, d.power.g
    hubs = sorted(h.hubs)
    colors = plt.cm.tab10.colors
    zc = {hub: colors[i % 10] for i, hub in enumerate(hubs)}
    fig, ax = plt.subplots(figsize=(9, 8), facecolor="#14161a")
    ax.set_facecolor("#14161a")
    added = set(d.power.added_lines)
    for u, v, a in g.edges():
        p, q = g.node[u], g.node[v]
        cross = h.zone_of[u] != h.zone_of[v]
        ax.plot([p["x_km"], q["x_km"]], [p["y_km"], q["y_km"]],
                color="#e6e6e6" if cross else "#5c6370", lw=1.6 if cross else 1.0,
                ls="--" if (u, v) in added else "-", alpha=0.9 if cross else 0.7, zorder=1)
    for u in g.nodes():
        n = g.node[u]
        pmu = u in d.pmu.pmus
        ax.scatter(n["x_km"], n["y_km"], s=170 if u in h.hubs else 70,
                   marker="*" if u in h.hubs else ("s" if pmu else "o"),
                   color=zc[h.zone_of[u]], edgecolor="white" if pmu else "none",
                   linewidth=1.2, zorder=3)
    c = g.node[h.cc]
    ax.scatter(c["x_km"], c["y_km"], s=520, marker="*", facecolor="none",
               edgecolor="#ffd166", linewidth=2.2, zorder=4)
    ax.set_title(f"Synthesised grid: {g.n} substations, {h.k} zones, {d.pmu.n_final} PMUs "
                 f"(squares), control centre ringed  |  dashed = line added for N-1",
                 color="#e6e6e6", fontsize=9)
    ax.set_aspect("equal")
    ax.tick_params(colors="#9aa0a6")
    for s in ax.spines.values():
        s.set_color("#3a3f47")
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------- sweeps
def sensitivity(A, M, seed=42, sites=None):
    """Re-design under changed assumptions: which conclusions are fragile?"""
    rows = []

    def run(param, value, **ov):
        d = design_grid(replace(A, **ov), M, seed=seed, sites=sites, security=False)
        lc, lt = EV.latency_pair(d.hier, M, d.A)
        rows.append(dict(parameter=param, value=value, k_zones=d.hier.k,
                         n_levels=d.hier.n_levels, feasible=d.hier.feasible,
                         binding=",".join(binding_constraints(d)) or "-",
                         n_pmus=d.pmu.n_final, max_hub_cores=d.hier.metrics["max_hub_cores"],
                         max_station_cores=d.hier.metrics["max_station_cores"],
                         t_hub_verdict_ms=d.hier.metrics["t_hub_verdict_ms"],
                         latency_central_ms=round(lc, 2), latency_tiered_ms=round(lt, 2),
                         edge_rate_cap_hz=next(r["sustainable_score_rate_hz"]
                                               for r in d.roles if r["role"] == "bay sentinel"),
                         violations=d.hier.metrics["violations"]))

    for s in (1, 10, 50):
        run("hardware_slowdown", s, hardware_slowdown=s)
    for c in (0.15, 0.25, 0.35, 0.5, 1.0):
        run("max_zone_share", c, max_zone_share=c)
    for hp in (2, 3, 4, 6, 12):
        run("max_zone_hops", hp, max_zone_hops=hp)
    for hm in (0.1, 1.0, 5.0, 20.0):
        run("hop_ms", hm, hop_ms=hm)
    for k in (1, 2):
        run("pmu_obs_k", k, pmu_obs_k=k)
    for z in (True, False):
        run("use_zib", z, use_zib=z)
    for m in ("edge", "vertex"):
        run("n1_mode", m, n1_mode=m)
    return rows


def scaling(A, M, sizes=(20, 40, 80, 120), seeds=range(5)):
    """Same method on regions of different size: how do the architecture
    parameters scale? Region side grows with sqrt(n) to hold site density."""
    rows = []
    for n in sizes:
        for sd in seeds:
            A2 = replace(A, n_sites=n, region_km=A.region_km * (n / A.n_sites) ** 0.5)
            d = design_grid(A2, M, seed=sd)
            s = summarize(d)
            cmp_ = {r["deployment"].split()[0]: r for r in d.comparison}
            rows.append(dict(
                n_substations=n, seed=sd, n_bays=s["n_bays"],
                n1_premium_pct=s["power"]["n1_premium_over_radial_pct"],
                n_pmus=s["pmu"]["n_pmus"], pmu_fraction=s["pmu"]["fraction_of_substations"],
                k_zones=s["hierarchy"]["k_zones"], n_levels=s["hierarchy"]["n_levels"],
                feasible=s["hierarchy"]["feasible"],
                binding=",".join(s["hierarchy"]["binding_constraints"]) or "-",
                B_over_C_peak_link=round(cmp_["B"]["peak_link_KBps"]
                                         / max(1e-9, cmp_["C"]["peak_link_KBps"]), 1),
                B_peak_link_util=cmp_["B"]["peak_link_util"],
                C_peak_link_util=cmp_["C"]["peak_link_util"],
                regional_worst_central=cmp_["B"]["regional_loss_coverage_worst"],
                regional_worst_tiered=cmp_["C"]["regional_loss_coverage_worst"],
                border_fraction=round(len(s["gateways"]["border_stations"]) / n, 3),
                min_vertex_disjoint_to_cc=s["security"]["diversity"]["min_vertex_disjoint_to_cc"]))
    return rows
