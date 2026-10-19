"""
Stage A - synthesise the physical network.

Problem: given substation sites, choose transmission lines so the grid is
connected, survives any single line (or substation) outage, and costs as little
as possible. This is survivable network design; the exact problem is NP-hard,
so the construction is the classic two-phase heuristic:

  1. KRUSKAL MST over all site pairs  -> the cheapest connected backbone. It is
     radial (a tree): every line is a bridge, every interior node a cut vertex.
  2. GREEDY AUGMENTATION             -> repeatedly add the candidate line that
     repairs the most weak points per km, where weak points are found by a
     DFS low-link pass (bridges, and cut vertices in "vertex" mode). A candidate
     (u, v) repairs a weak point iff u and v lie on different sides of it.

N-1 here is TOPOLOGICAL (the grid stays connected). It does not check that the
surviving lines can carry the rerouted power; that needs a power-flow model and
is out of scope for a detection-architecture project. Say so in the report.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .graph import (Graph, bfs, bridges_and_cut_vertices, connected_components,
                    ek, kruskal)
from .sites import dist_km


@dataclass
class PowerGrid:
    g: Graph                    # the designed grid
    sites: dict                 # id -> Site
    bays: dict                  # id -> bay count (line terminals + gen/load tie)
    variants: dict              # {"radial": Graph, "designed": Graph, "mesh": Graph}
    audit: dict = field(default_factory=dict)
    added_lines: list = field(default_factory=list)


def _pair_lengths(sites):
    return {ek(a.id, b.id): dist_km(a, b)
            for i, a in enumerate(sites) for b in sites[i + 1:]}


def _graph(sites, edges, line_kusd_per_km):
    g = Graph()
    for s in sites:
        g.add_node(s.id, kind=s.kind, mw=s.mw, x_km=s.x_km, y_km=s.y_km)
    for (u, v), d in edges.items():
        g.add_edge(u, v, length_km=d, cost_kusd=d * line_kusd_per_km)
    return g


def _weak_points(g, mode):
    """One {node: side-label} map per weak point. A bridge yields two sides; a
    cut vertex yields one label per component left after deleting it (the
    vertex itself is absent from the map)."""
    bridges, cuts = bridges_and_cut_vertices(g)
    parts = []
    for (u, v) in sorted(bridges):
        reach, _ = bfs(g, u, skip_edges={(u, v)})
        parts.append({x: (0 if x in reach else 1) for x in g.nodes()})
    if mode == "vertex":
        for a in sorted(cuts):
            lab = {}
            for i, comp in enumerate(connected_components(g, skip_nodes={a})):
                for x in comp:
                    lab[x] = i
            parts.append(lab)
    return parts


def _covers(part, u, v):
    lu, lv = part.get(u), part.get(v)
    return lu is not None and lv is not None and lu != lv


def _augment(g, pool, mode, line_kusd_per_km):
    added = []
    while True:
        parts = _weak_points(g, mode)
        if not parts:
            return added, True
        best = None
        for (u, v), d in sorted(pool.items()):
            if g.has_edge(u, v):
                continue
            c = sum(1 for p in parts if _covers(p, u, v))
            if c == 0:
                continue
            score = (c / d, -d)            # most repairs per km; shorter on ties
            if best is None or score > best[0]:
                best = (score, u, v, d)
        if best is None:
            return added, False            # candidate pool cannot finish the job
        _, u, v, d = best
        g.add_edge(u, v, length_km=d, cost_kusd=d * line_kusd_per_km)
        added.append((u, v))


def survivability(g):
    """Topological N-1 audit: how much of the grid a single outage cuts off."""
    bridges, cuts = bridges_and_cut_vertices(g)
    n = g.n

    def islanded(comps, lost):
        return 0 if len(comps) <= 1 else (n - lost) - max(len(c) for c in comps)

    worst_node = max(islanded(connected_components(g, skip_nodes={f}), 1) for f in g.nodes())
    worst_line = max((islanded(connected_components(g, skip_edges={(u, v)}), 0)
                      for u, v, _ in g.edges()), default=0)
    diameter = max(max(bfs(g, s)[0].values()) for s in g.nodes())
    return dict(connected=len(connected_components(g)) == 1,
                lines=g.m, bridges=len(bridges), cut_vertices=len(cuts),
                worst_node_outage_islanded=worst_node,
                worst_line_outage_islanded=worst_line,
                diameter_hops=diameter,
                mean_degree=round(2 * g.m / n, 2),
                cost_kusd=round(sum(a["cost_kusd"] for _, _, a in g.edges()), 1),
                km=round(sum(a["length_km"] for _, _, a in g.edges()), 1))


def candidate_pool(sites, A):
    """(all pair lengths, MST edges, candidate pool = MST + k nearest neighbours).
    Shared by the optimized synthesis and by the default builders, so both choose from
    the same corridors."""
    L = _pair_lengths(sites)
    ids = [s.id for s in sites]
    mst = kruskal(ids, ((d, u, v) for (u, v), d in L.items()))
    mst_edges = {(u, v): d for u, v, d in mst}
    pool = dict(mst_edges)
    for i in ids:
        for d, j in sorted((L[ek(i, j)], j) for j in ids if j != i)[:A.candidate_knn]:
            pool[ek(i, j)] = d
    return L, mst_edges, pool


def power_grid_from_edges(sites, edges, A, label="selected"):
    """A PowerGrid for an arbitrary corridor list [(u, v), ...] (lengths from the geometry).
    Used for the default builds and for a test case's own as-built lines."""
    by_id = {s.id: s for s in sites}
    e = {ek(u, v): dist_km(by_id[u], by_id[v]) for u, v in edges}
    g = _graph(sites, e, A.line_kusd_per_km)
    bays = {u: g.degree(u) + (0 if by_id[u].kind == "junction" else 1) for u in g.nodes()}
    return PowerGrid(g=g, sites=by_id, bays=bays, variants={label: g},
                     audit={label: survivability(g)}, added_lines=[])


def synthesize_power_grid(sites, A):
    L, mst_edges, pool = candidate_pool(sites, A)

    radial = _graph(sites, mst_edges, A.line_kusd_per_km)
    designed = radial.copy()
    added, ok = _augment(designed, pool, A.n1_mode, A.line_kusd_per_km)
    if not ok:                                   # fall back to every possible pair
        more, ok = _augment(designed, L, A.n1_mode, A.line_kusd_per_km)
        added += more
    if not ok:
        raise RuntimeError("could not reach N-1 security even with all site pairs")
    mesh = _graph(sites, pool, A.line_kusd_per_km)

    by_id = {s.id: s for s in sites}
    bays = {u: designed.degree(u) + (0 if by_id[u].kind == "junction" else 1)
            for u in designed.nodes()}
    variants = {"radial (MST only)": radial, "designed (N-1 secure)": designed,
                "mesh (all candidate lines)": mesh}
    audit = {name: survivability(g) for name, g in variants.items()}
    return PowerGrid(g=designed, sites=by_id, bays=bays, variants=variants,
                     audit=audit, added_lines=added)
