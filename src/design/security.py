"""
Stage D - stress the designed grid: failures, lateral movement, path diversity.

Everything here is a TRAVERSAL of the designed graph, not a model fit:

  * failure sweeps      BFS on "the graph minus the failed nodes": which bays can
                        still reach a fusion point? One sweep deletes each node
                        in turn (N-1); the other deletes a geographic disc around
                        each node (a storm, or a coordinated attack - fibre rides
                        the same towers as the power lines, so a regional outage
                        takes out comms and power together).
  * lateral movement    BFS from a compromised node over the links a segmented
                        network still permits (intra-zone only) vs a flat one.
  * path diversity      max-flow between each node and the control centre:
                        Menger's theorem turns "how many independent routes?"
                        into a number.

What the coverage numbers mean. `central` is a bay whose station can still reach
the control centre; `central_backup` adds a hot-standby CC at a geographically
separate site. `tiered_fused` is a bay whose station can still reach its hub
or the CC. `tiered_any` is 1.0 BY ARCHITECTURE: every station hosts its own
correlator (design rule R3 - fail open for the grid), so a surviving station keeps
detecting locally. That is a structural property of the design, not a measured
accuracy; how much accuracy station-only detection keeps is what the "station" tier of step 6 measures.
"""
from __future__ import annotations

import math

from .graph import Graph, bfs, dijkstra, disjoint_paths


def backup_cc(cg, bays, h, min_sep_km):
    """Where a conventional centralised design would put its hot-standby control
    centre: the bay-weighted median among sites at least `min_sep_km` from the
    primary (geographic diversity). Included so the centralised baselines are not
    a straw man - nobody runs a single control centre with no backup."""
    pos = {u: (cg.node[u]["x_km"], cg.node[u]["y_km"]) for u in cg.nodes()}
    pool = [v for v in cg.nodes() if v != h.cc and math.dist(pos[v], pos[h.cc]) >= min_sep_km]
    pool = pool or [v for v in cg.nodes() if v != h.cc]

    def spread(c):
        dist = dijkstra(cg, c)[0]
        return (sum(bays[v] * dist[v] for v in cg.nodes()), c)
    return min(pool, key=spread)


def _coverage(cg, bays, h, failed, backup=None):
    failed = set(failed)
    survivors = [v for v in cg.nodes() if v not in failed]
    alive = sum(bays[v] for v in survivors)
    if alive == 0:
        return None
    to_cc = bfs(cg, h.cc, skip_nodes=failed)[0] if h.cc not in failed else {}
    reach_hub = {hub: bfs(cg, hub, skip_nodes=failed)[0] for hub in h.hubs if hub not in failed}
    central = sum(bays[v] for v in survivors if v in to_cc)
    to_bk = bfs(cg, backup, skip_nodes=failed)[0] if backup is not None and backup not in failed else {}
    central_bk = sum(bays[v] for v in survivors if v in to_cc or v in to_bk)
    fused = sum(bays[v] for v in survivors
                if v in to_cc or v in reach_hub.get(h.zone_of[v], {}))
    return dict(surviving_bays=alive, central=central / alive,
                central_backup=central_bk / alive,
                tiered_fused=fused / alive, tiered_any=1.0)


def _summarise(rows):
    out = {}
    for arch in ("central", "central_backup", "tiered_fused", "tiered_any"):
        vals = sorted(r[arch] for r in rows)
        out[arch] = dict(mean=sum(vals) / len(vals), worst=vals[0],
                         p10=vals[int(0.1 * len(vals))])
    return out


def single_node_sweep(cg, bays, h, backup=None):
    rows = []
    for f in cg.nodes():
        c = _coverage(cg, bays, h, {f}, backup)
        if c:
            rows.append(dict(failed=f, is_cc=f == h.cc, is_hub=f in h.hubs, **c))
    return rows, _summarise(rows)


def regional_sweep(cg, bays, h, radius_km, backup=None):
    pos = {u: (cg.node[u]["x_km"], cg.node[u]["y_km"]) for u in cg.nodes()}
    rows = []
    for epi in cg.nodes():
        failed = {v for v in cg.nodes() if math.dist(pos[v], pos[epi]) <= radius_km}
        c = _coverage(cg, bays, h, failed, backup)
        if c:
            rows.append(dict(epicentre=epi, n_failed=len(failed),
                             cc_lost=h.cc in failed, **c))
    return rows, _summarise(rows)


def lateral_reach(cg, bays, h):
    """Share of all bays an attacker can reach from one compromised substation.
    Flat network: everything connected. Segmented: only links inside the zone."""
    total = sum(bays.values())
    zoned = Graph()
    for u in cg.nodes():
        zoned.add_node(u)
    for u, v, a in cg.edges():
        if h.zone_of[u] == h.zone_of[v]:
            zoned.add_edge(u, v)
    shares = [sum(bays[x] for x in bfs(zoned, u)[0]) / total for u in cg.nodes()]
    return dict(flat=1.0, zoned_mean=sum(shares) / len(shares), zoned_max=max(shares))


def gateways(cg, h):
    """Inter-zone links are the only places traffic may cross zones (rule R6).
    Their end-points are the substations that must carry authenticated,
    filtered gateways (IEC 62351 territory); the partition's edge cut is the
    size of that attack surface."""
    cut = [(u, v) for u, v, _ in cg.edges() if h.zone_of[u] != h.zone_of[v]]
    border = sorted({x for e in cut for x in e})
    return dict(inter_zone_links=cut, border_stations=border,
                edge_cut=len(cut), cut_fraction=len(cut) / max(1, cg.m))


def path_diversity(cg, h):
    """Independent routes to the CC (and, in the zoning's favour, from each
    station to its hub). Menger: >= 2 means no single link or node isolates it."""
    to_cc_edge, to_cc_vertex = [], []
    for v in cg.nodes():
        if v == h.cc:
            continue
        to_cc_edge.append(disjoint_paths(cg, v, h.cc))
        to_cc_vertex.append(disjoint_paths(cg, v, h.cc, vertex=True))
    hub_vertex = [disjoint_paths(cg, v, h.zone_of[v], vertex=True)
                  for v in cg.nodes() if v != h.zone_of[v]]
    return dict(min_edge_disjoint_to_cc=min(to_cc_edge),
                min_vertex_disjoint_to_cc=min(to_cc_vertex),
                min_vertex_disjoint_to_hub=min(hub_vertex) if hub_vertex else None)
