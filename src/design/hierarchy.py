"""
Stage C - let the hierarchy fall out of the graph.

Steps 1-4 fixed the levels (bay / substation / wide-area) and asked which model
suits each. Here the number of levels is an OUTPUT. Every substation hosts a
station correlator (bus-balance residuals need the bays that share a station
bus). What is decided is whether zone HUBS are needed between stations and the
control centre (CC), how many, and where:

  * ZONING is a k-medoid facility-location problem on shortest-path LATENCY
    (all-pairs Dijkstra), solved by PAM swaps with restarts. Zones are the
    multi-source Dijkstra cells of the chosen hubs, which are connected by
    construction, so a zone's members can reach their hub without leaving it.
  * The smallest k whose zones satisfy every constraint wins (cost rises with
    k). Constraints: blast-radius share, hop radius, zone size, latency
    budget, hub compute. The sweep is recorded in full, so the report can show
    which constraint forced each extra zone, and which were slack.
  * k = 1 collapses hub and CC: three levels instead of four. A looser
    exposure policy therefore REMOVES a level. Nothing in the code assumes four.
  * The CC is the non-hub substation with the smallest bay-weighted latency to
    the hubs (at k = 1 the hub and CC are necessarily the same node).

All compute figures scale E4's measured per-message cost by the number of
measured-unit-equivalents a node serves (bays / merging units per measured row).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from .graph import Graph, dijkstra


# ---------------------------------------------------------------- comm graph
def build_comm_graph(power_g, A):
    """Fibre rides on every transmission corridor, so the comm graph has the
    power graph's shape. Edge latency = propagation + per-hop forwarding."""
    cg = Graph()
    for u in power_g.nodes():
        cg.add_node(u, **power_g.node[u])
    for u, v, a in power_g.edges():
        km = a["length_km"]
        cg.add_edge(u, v, length_km=km,
                    latency_ms=km * A.fibre_us_per_km / 1000.0 + A.hop_ms,
                    cost_kusd=km * A.fibre_kusd_per_km,
                    capacity_Bps=A.link_capacity_Bps)
    return cg


def latency_matrix(cg):
    ids = cg.nodes()
    idx = {u: i for i, u in enumerate(ids)}
    D = np.full((len(ids), len(ids)), np.inf)
    for u in ids:
        dist, _, _ = dijkstra(cg, u)
        for v, d in dist.items():
            D[idx[u], idx[v]] = d
    return ids, idx, D


# ------------------------------------------------------------------- PAM
def _cost(D, w, med):
    return float((w * D[:, med].min(axis=1)).sum())


def _greedy_init(D, w, k):
    med = []
    for _ in range(k):
        best, best_c = None, math.inf
        for j in range(D.shape[0]):
            if j in med:
                continue
            c = _cost(D, w, med + [j])
            if c < best_c:
                best, best_c = j, c
        med.append(best)
    return med


def _refine(D, w, med, max_iter=60):
    med = list(med)
    n = D.shape[0]
    cur = _cost(D, w, med)
    for _ in range(max_iter):
        improved = False
        for pos in range(len(med)):
            others = [m for i, m in enumerate(med) if i != pos]
            base = D[:, others].min(axis=1) if others else np.full(n, np.inf)
            costs = (w[:, None] * np.minimum(base[:, None], D)).sum(axis=0)
            costs[others] = np.inf
            j = int(np.argmin(costs))
            if costs[j] < cur - 1e-9:
                med[pos], cur, improved = j, float(costs[j]), True
        if not improved:
            break
    return sorted(med)


def medoid_candidates(D, w, k, rng, restarts):
    """Distinct locally-optimal hub sets: one greedy start + random restarts."""
    n = D.shape[0]
    seen, out = set(), []
    for r in range(max(1, restarts)):
        start = _greedy_init(D, w, k) if r == 0 else rng.sample(range(n), k)
        med = tuple(_refine(D, w, start))
        if med not in seen:
            seen.add(med)
            out.append(list(med))
    return out


# ------------------------------------------------------------- hierarchy
@dataclass
class Hierarchy:
    k: int
    hubs: list
    cc: int
    zone_of: dict               # station -> hub
    to_hub: dict                # station -> next hop toward its hub (None at hub)
    hops: dict                  # station -> hops to hub
    lat_ms: dict                # station -> latency to hub
    to_cc: dict                 # node -> next hop toward the CC
    cc_lat_ms: dict             # node -> latency to the CC
    zones: dict                 # hub -> sorted member list
    metrics: dict = field(default_factory=dict)
    violations: list = field(default_factory=list)
    feasible: bool = True

    @property
    def n_levels(self):
        # bay, station, [hub], control centre. k == 1: the hub IS the CC.
        return 3 if self.k == 1 else 4

    def level_names(self):
        base = ["L1 bay sentinel", "L2 station correlator"]
        return base + (["L3 zone hub"] if self.k > 1 else []) + \
            [f"L{self.n_levels} control centre"]


def cores_needed(bays, batch_ms, A, M):
    """CPU cores to score `bays` bays at the sample rate, in batches, on
    hardware `hardware_slowdown`x slower than E4's, with headroom `util_cap`."""
    units = bays / M.ieds_per_unit
    busy = units * M.sample_rate_hz * batch_ms * A.hardware_slowdown / 1000.0
    return busy / A.util_cap


def _evaluate(cg, ids, idx, D, w, hubs_idx, bays, A, M):
    hubs = [ids[i] for i in hubs_idx]
    dist, parent, origin = dijkstra(cg, hubs)
    zones = {h: sorted(v for v, o in origin.items() if o == h) for h in hubs}
    total_bays = sum(bays.values())

    # hop count to the hub along each station's shortest-latency path
    # (parents always have strictly smaller distance, so one ordered pass works)
    hops = {}
    for v in sorted(dist, key=dist.get):
        hops[v] = 0 if parent[v] is None else hops[parent[v]] + 1

    zbays = {h: sum(bays[v] for v in m) for h, m in zones.items()}
    share = {h: zbays[h] / total_bays for h in hubs}

    # Control centre: bay-weighted 1-median of the hubs, taken over NON-hub
    # substations when k > 1. A node never carries two aggregation levels: the
    # single-node-loss sweep showed that a CC colocated with a hub takes a whole
    # zone's hub-level fusion down with it. At k == 1 the hub IS the CC (forced).
    if len(hubs) == 1:
        cc = hubs[0]
    else:
        pool = [u for u in ids if u not in hubs]
        cc = min(pool, key=lambda c: (sum(zbays[h] * D[idx[h], idx[c]] for h in hubs), c))
    cc_dist, cc_parent, _ = dijkstra(cg, cc)

    worst_station = max(dist.values())
    t_hub = (M.sample_period_ms + (M.edge_p95_ms + M.station_p95_ms + M.wide_p95_ms)
             * A.hardware_slowdown + A.lan_ms + worst_station)
    worst_hub_to_cc = max(cc_dist[h] for h in hubs)
    t_cc = t_hub + worst_hub_to_cc + M.wide_p95_ms * A.hardware_slowdown
    hub_cores = {h: cores_needed(zbays[h], M.wide_batch_ms, A, M) for h in hubs}
    station_cores = {v: cores_needed(bays[v], M.station_batch_ms, A, M) for v in cg.nodes()}

    k = len(hubs)
    bad = []
    if max(share.values()) > A.max_zone_share + 1e-12:
        bad.append("zone_share")
    if max(hops.values()) > A.max_zone_hops:
        bad.append("zone_hops")
    if k > 1 and min(len(m) for m in zones.values()) < A.min_zone_stations:
        bad.append("zone_min_size")
    if t_hub > A.zone_budget_ms or t_cc > A.wide_budget_ms:
        bad.append("latency")
    if max(hub_cores.values()) > A.max_cores_per_node:
        bad.append("hub_compute")
    if max(station_cores.values()) > A.max_cores_per_node:
        bad.append("station_compute")

    metrics = dict(
        k=k, max_zone_share=round(max(share.values()), 4),
        min_zone_stations=min(len(m) for m in zones.values()),
        max_zone_stations=max(len(m) for m in zones.values()),
        max_hops=max(hops.values()),
        mean_hops=round(sum(hops.values()) / len(hops), 3),
        worst_station_to_hub_ms=round(worst_station, 3),
        t_hub_verdict_ms=round(t_hub, 3), t_cc_verdict_ms=round(t_cc, 3),
        max_hub_cores=round(max(hub_cores.values()), 3),
        max_station_cores=round(max(station_cores.values()), 3),
        objective=round(float((w * D[:, hubs_idx].min(axis=1)).sum()), 3),
        hardware_kusd=A.hub_kusd * k + A.cc_kusd,
        violations=",".join(bad) or "none")
    return Hierarchy(k=k, hubs=hubs, cc=cc, zone_of=dict(origin), to_hub=parent, hops=hops,
                     lat_ms=dist, to_cc=cc_parent, cc_lat_ms=cc_dist, zones=zones,
                     metrics=metrics, violations=bad, feasible=not bad)


def design_hierarchy(cg, bays, A, M, seed=0):
    """Sweep k, keep the smallest feasible one. Returns (Hierarchy, sweep rows)."""
    ids, idx, D = latency_matrix(cg)
    w = np.array([bays[u] for u in ids], dtype=float)
    rng = random.Random(seed)
    k_max = max(1, min(A.k_zone_max, len(ids) // max(1, A.min_zone_stations)))
    rows, chosen, fallback = [], None, None
    for k in range(1, k_max + 1):
        cands = [_evaluate(cg, ids, idx, D, w, m, bays, A, M)
                 for m in medoid_candidates(D, w, k, rng, A.pam_restarts)]
        best = min(cands, key=lambda h: (len(h.violations), h.metrics["max_zone_share"],
                                         h.metrics["objective"]))
        rows.append(dict(best.metrics, feasible=best.feasible, hubs=best.hubs, cc=best.cc,
                         restarts_tried=len(cands)))
        if best.feasible and chosen is None:
            chosen = best
        rank = (len(best.violations), best.metrics["max_zone_share"])
        if fallback is None or rank < fallback[0]:
            fallback = (rank, best)
    if chosen is None:        # nothing feasible: fewest violations, then smallest blast radius
        chosen = fallback[1]
    return chosen, rows
