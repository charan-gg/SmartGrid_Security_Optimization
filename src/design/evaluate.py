"""
Stage E - turn the designed graph into the numbers an architecture review asks for.

Three comparisons, all on the SAME designed physical grid so the only thing that
changes is where detection runs:

  A   physical-only, centralised   every bay streams its process-bus (MU phasor) columns to one CC
  B   centralised fusion, raw      every bay streams every feature column to one CC
  B+  B with a hot-standby CC      same, plus a geo-separate backup site (so the
                                   centralised baseline is not a straw man)
  C   tiered (this design)         bay sentinels -> station correlators -> hubs -> CC

plus the physical-network comparison (radial / designed / mesh), the PMU plan's
single-PMU-loss behaviour, and the role -> model table that replaces step 4's
fixed tier assignment (roles now come from the graph).

Traffic is routed by shortest-latency paths (Dijkstra trees) and summed per
link, so "bandwidth" is a per-corridor load, not a single aggregate.
"""
from __future__ import annotations

import math
from collections import defaultdict

from .graph import dijkstra, ek
from .hierarchy import cores_needed
from .observability import observed_set


# ------------------------------------------------------------------ traffic
def _route(next_hop, src, flow, load):
    v = src
    while next_hop.get(v) is not None:
        u = next_hop[v]
        load[ek(v, u)] += flow
        v = u


def verdict_rate_hz(M, A):
    """Verdicts/s a station emits: at most one per scored row, so never above the measured
    row rate (A.verdict_hz is only the ceiling for faster streams)."""
    return min(A.verdict_hz, M.sample_rate_hz)


def link_loads(h, bays, M, A, arch, to_backup=None):
    load = defaultdict(float)
    nodes = list(h.zone_of)
    if arch == "C":
        for v in nodes:                                   # evidence up to the hub
            _route(h.to_hub, v, bays[v] * M.evidence_Bps_per_bay, load)
        for hub in h.hubs:                                # hub digest up to the CC
            _route(h.to_cc, hub, len(h.zones[hub]) * A.verdict_bytes * verdict_rate_hz(M, A), load)
    else:
        per_bay = M.process_Bps_per_bay if arch == "A" else M.raw_Bps_per_bay
        for v in nodes:
            _route(h.to_cc, v, bays[v] * per_bay, load)
            if arch == "B+" and to_backup is not None:    # streams are mirrored
                _route(to_backup, v, bays[v] * per_bay, load)
    return load


def _ingress(load, cc, cg):
    return sum(load.get(ek(cc, v), 0.0) for v in cg.adj[cc])


def latency_pair(h, M, A):
    """(centralised, tiered) worst-case bay -> first fused verdict, in ms,
    pipeline only (transport + compute). One message is one decision, so no
    detector window is added."""
    central = M.sample_period_ms + A.lan_ms + max(h.cc_lat_ms.values()) + M.wide_p95_ms * A.hardware_slowdown
    return central, h.metrics["t_hub_verdict_ms"]


def compare_deployments(cg, bays, h, M, A, node_sweep, region_sweep, backup):
    total = sum(bays.values())
    central_ms, tiered_ms = latency_pair(h, M, A)
    to_backup = dijkstra(cg, backup)[1]
    rows = []
    for arch, label, key in (("A", "A physical-only centralised", "central"),
                             ("B", "B centralised fusion (raw)", "central"),
                             ("B+", "B+ B with backup CC", "central_backup"),
                             ("C", "C tiered (this design)", "tiered_fused")):
        load = link_loads(h, bays, M, A, arch, to_backup)
        peak = max(load.values()) if load else 0.0
        ing = _ingress(load, h.cc, cg)
        if arch == "C":
            cores = max(h.metrics["max_hub_cores"], h.metrics["max_station_cores"])
            latency, blast, n_ctrl = tiered_ms, h.metrics["max_zone_share"], len(h.hubs) + 1
        else:
            cores = cores_needed(total, M.wide_batch_ms, A, M)
            latency, blast, n_ctrl = central_ms, 1.0, 2 if arch == "B+" else 1
        nres, rres = node_sweep[key], region_sweep[key]
        coverage = M.det_process if arch == "A" else M.det_all     # E1 attack coverage of what it sees
        rows.append({
            "deployment": label,
            "cc_ingress_KBps": round(ing / 1e3, 1),
            "peak_link_KBps": round(peak / 1e3, 1),
            "peak_link_util": round(peak / A.link_capacity_Bps, 4),
            "links_over_capacity": sum(1 for x in load.values() if x > A.link_capacity_Bps),
            "max_node_cores": round(cores, 3),
            "worst_bay_to_verdict_ms": round(latency, 2),
            "blast_share": round(blast, 4),
            "attack_coverage_intact": round(coverage, 3),
            "fusion_points": n_ctrl,
            "single_node_loss_coverage_mean": round(nres["mean"], 4),
            "single_node_loss_coverage_worst": round(nres["worst"], 4),
            "regional_loss_coverage_mean": round(rres["mean"], 4),
            "regional_loss_coverage_worst": round(rres["worst"], 4),
        })
    return rows


# ------------------------------------------------------------- role -> model
def role_model_map(cg, bays, h, M, A):
    """Which model runs where - but where is now read off the designed graph."""
    s = A.hardware_slowdown
    edge_util = M.sample_rate_hz * M.edge_p50_ms * s / 1000.0
    rate_cap = min(M.sample_rate_hz, math.floor(A.util_cap * 1000.0 / (M.edge_p50_ms * s)))
    st_cores = [cores_needed(bays[v], M.station_batch_ms, A, M) for v in cg.nodes()]
    hub_cores = [cores_needed(sum(bays[v] for v in h.zones[x]), M.wide_batch_ms, A, M)
                 for x in h.hubs]
    zone_in = [sum(bays[v] for v in h.zones[x]) * M.evidence_Bps_per_bay for x in h.hubs]
    rows = [
        dict(level=h.level_names()[0], role="bay sentinel", count=sum(bays.values()),
             model="E4 tier1_edge_kstar", features=M.k_stable,
             p95_ms=round(M.edge_p95_ms * s, 3),
             load=f"{edge_util:.0%} of 1 core @ {M.sample_rate_hz:.1f} Hz",
             sustainable_score_rate_hz=rate_cap,
             bytes_out_Bps=round(M.evidence_Bps_per_bay, 1),
             note=("per-sample scoring sustainable" if rate_cap >= M.sample_rate_hz
                   else f"must decimate to {rate_cap} Hz or batch")),
        dict(level=h.level_names()[1], role="station correlator", count=cg.n,
             model="E4 tier2_zone_fused", features=M.n_features_full,
             p95_ms=round(M.station_p95_ms * s, 3),
             load=f"max {max(st_cores):.2f} cores",
             sustainable_score_rate_hz="", bytes_out_Bps=round(
                 max(bays.values()) * M.evidence_Bps_per_bay, 1),
             note="max over stations; forwards bay evidence to its hub"),
    ]
    if h.k > 1:
        rows.append(dict(level=h.level_names()[2], role="zone hub", count=h.k,
                         model="E4 tier3_widearea_fused", features=M.n_features_full,
                         p95_ms=round(M.wide_p95_ms * s, 3),
                         load=f"max {max(hub_cores):.2f} cores",
                         sustainable_score_rate_hz="",
                         bytes_out_Bps=round(max(len(h.zones[x]) for x in h.hubs)
                                             * A.verdict_bytes * verdict_rate_hz(M, A), 1),
                         note=f"max inbound {max(zone_in) / 1e3:.1f} KB/s; one zone = one "
                              "federated-training unit (E8)"))
    rows.append(dict(level=h.level_names()[-1], role="control centre", count=1,
                     model="E4 tier3_widearea_fused", features=M.n_features_full,
                     p95_ms=round(M.wide_p95_ms * s, 3), load="digests only",
                     sustainable_score_rate_hz="", bytes_out_Bps=0.0,
                     note="owns model updates (E8 transferability gap)"))
    return rows


# --------------------------------------------------------------- PMU analysis
def expected_detection(g, bays, pmus, zib, M):
    """Bay-weighted attack coverage: E1's all-layers figure where a sensor observes the
    substation, E1's station-bus-only figure where it does not. Population averages from
    E1 (blocked CV on Train.csv) applied per substation - a coarse bound, not a per-bus
    measurement. When the two E1 figures are close, sensor placement buys no detection."""
    obs = observed_set(g, pmus, zib)
    tot = sum(bays.values())
    return sum(bays[u] * (M.det_all if u in obs else M.det_station)
               for u in g.nodes()) / tot


def pmu_loss_sweep(g, bays, plan, zib, M):
    base = expected_detection(g, bays, plan.pmus, zib, M)
    after = [expected_detection(g, bays, plan.pmus - {p}, zib, M) for p in sorted(plan.pmus)]
    unobs = [g.n - len(observed_set(g, plan.pmus - {p}, zib)) for p in sorted(plan.pmus)]
    return dict(intact=base, worst_after_single_pmu_loss=min(after),
                mean_after_single_pmu_loss=sum(after) / len(after),
                max_unobserved_after_loss=max(unobs))


# --------------------------------------------------------------------- costs
def cost_accounts(power, pmu_plan, cg, bays, h, A, deployment="tiered"):
    """Relative kUSD. `deployment` decides which detection hardware exists:
    tiered           bay sentinels + a correlator at every station + k hubs + one CC
    central          one CC (+ the per-station collector price, A.central_gateway_kusd_per_station)
    central_backup   as central, plus a hot-standby CC
    The default deployments carry no sentinels, correlators or hubs; that is what
    'centralised' means, and it is favourable to them in the cost comparison."""
    comm = sum(a["cost_kusd"] for _, _, a in cg.edges())
    lines = sum(a["cost_kusd"] for _, _, a in power.g.edges())
    if deployment == "tiered":
        sentinels = sum(bays.values()) * A.sentinel_kusd_per_bay
        stations = cg.n * A.station_correlator_kusd
        hubs, ccs = h.k * A.hub_kusd, A.cc_kusd
    else:
        sentinels = 0.0
        stations = cg.n * A.central_gateway_kusd_per_station
        hubs = 0.0
        ccs = A.cc_kusd * (2 if deployment == "central_backup" else 1)
    d = dict(power_lines=lines, comm_fibre=comm,
             sensing_pmu=pmu_plan.n_final * A.pmu_kusd,
             sensing_sentinels=sentinels, compute_stations=stations,
             compute_hubs=hubs, compute_cc=ccs)
    d["detection_plane"] = (d["sensing_pmu"] + d["sensing_sentinels"] + d["compute_stations"]
                            + d["compute_hubs"] + d["compute_cc"])
    d["total"] = d["power_lines"] + d["comm_fibre"] + d["detection_plane"]
    return {k: round(v, 1) for k, v in d.items()}
