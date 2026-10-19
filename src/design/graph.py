"""
Graph primitives for grid synthesis. Pure Python, no third-party imports.

Why hand-rolled rather than networkx: every design decision in step 5 is
justified by NAMING the traversal that made it (BFS for hop radius, Dijkstra for
latency zones, DFS low-link for N-1 audits, Kruskal for the backbone, max-flow
for independent paths). Having them in one readable file makes that mapping
auditable, and keeps the project's dependency list unchanged. The test suite
cross-checks every routine against networkx when networkx is installed.

Conventions
-----------
* Undirected simple graph, nodes are ints (substation ids).
* An edge is a dict of attributes shared by both directions.
* Neighbour iteration is sorted, so every result is deterministic.
* `skip_nodes` / `skip_edges` let a traversal run on "the graph minus X"
  without copying it - that is how failure sweeps are done.
"""
from __future__ import annotations

import heapq
import math
from collections import deque


def ek(u, v):
    """Canonical (sorted) key for an undirected edge."""
    return (u, v) if u <= v else (v, u)


def _seq(x):
    return list(x) if isinstance(x, (list, tuple, set, frozenset)) else [x]


class Graph:
    def __init__(self):
        self.adj = {}      # node -> {neighbour -> edge attr dict (shared)}
        self.node = {}     # node -> attr dict

    # ------------------------------------------------------------- building
    def add_node(self, u, **attrs):
        self.adj.setdefault(u, {})
        self.node.setdefault(u, {}).update(attrs)

    def add_edge(self, u, v, **attrs):
        if u == v:
            raise ValueError(f"self-loop on {u}")
        for x in (u, v):
            if x not in self.adj:
                self.add_node(x)
        d = self.adj[u].get(v)
        if d is None:
            d = {}
            self.adj[u][v] = d
            self.adj[v][u] = d
        d.update(attrs)
        return d

    def remove_edge(self, u, v):
        del self.adj[u][v]
        del self.adj[v][u]

    # -------------------------------------------------------------- queries
    def has_edge(self, u, v):
        return u in self.adj and v in self.adj[u]

    def edge(self, u, v):
        return self.adj[u][v]

    def nodes(self):
        return sorted(self.adj)

    def neighbors(self, u):
        return sorted(self.adj[u])

    def degree(self, u):
        return len(self.adj[u])

    def edges(self):
        """[(u, v, attrs)] with u < v, sorted."""
        out = []
        for u in sorted(self.adj):
            for v in sorted(self.adj[u]):
                if u < v:
                    out.append((u, v, self.adj[u][v]))
        return out

    @property
    def n(self):
        return len(self.adj)

    @property
    def m(self):
        return sum(len(a) for a in self.adj.values()) // 2

    def __contains__(self, u):
        return u in self.adj

    def copy(self):
        h = Graph()
        for u, a in self.node.items():
            h.add_node(u, **a)
        for u, v, a in self.edges():
            h.add_edge(u, v, **a)
        return h

    def subgraph(self, keep):
        keep = set(keep)
        h = Graph()
        for u in sorted(keep):
            if u in self.adj:
                h.add_node(u, **self.node.get(u, {}))
        for u, v, a in self.edges():
            if u in keep and v in keep:
                h.add_edge(u, v, **a)
        return h


# ------------------------------------------------------------------ BFS / DFS
def bfs(g, sources, skip_nodes=(), skip_edges=()):
    """Multi-source breadth-first search.

    Returns (hops, parent): hop count from the nearest source and the BFS-tree
    parent. Only reached nodes appear in the dicts, so `v in hops` is the
    reachability test used throughout the failure sweeps.
    """
    skip_nodes, skip_edges = set(skip_nodes), set(skip_edges)
    hops, parent, dq = {}, {}, deque()
    for s in _seq(sources):
        if s in g.adj and s not in skip_nodes and s not in hops:
            hops[s], parent[s] = 0, None
            dq.append(s)
    while dq:
        u = dq.popleft()
        for v in g.neighbors(u):
            if v in hops or v in skip_nodes or ek(u, v) in skip_edges:
                continue
            hops[v], parent[v] = hops[u] + 1, u
            dq.append(v)
    return hops, parent


def connected_components(g, skip_nodes=(), skip_edges=()):
    """Iterative DFS. Returns a list of node sets, ordered by smallest member."""
    skip_nodes, skip_edges = set(skip_nodes), set(skip_edges)
    seen, comps = set(), []
    for root in g.nodes():
        if root in seen or root in skip_nodes:
            continue
        comp, stack = {root}, [root]
        seen.add(root)
        while stack:
            u = stack.pop()
            for v in g.neighbors(u):
                if v in seen or v in skip_nodes or ek(u, v) in skip_edges:
                    continue
                seen.add(v)
                comp.add(v)
                stack.append(v)
        comps.append(comp)
    return comps


def is_connected(g, skip_nodes=(), skip_edges=()):
    return len(connected_components(g, skip_nodes, skip_edges)) <= 1


def bridges_and_cut_vertices(g):
    """Tarjan low-link, iterative (no recursion limit).

    A BRIDGE is a line whose loss splits the grid; a CUT VERTEX is a substation
    whose loss does. A grid with neither is N-1 secure in the topological
    sense. Returns (set of (u, v) with u < v, set of nodes).
    """
    disc, low = {}, {}
    bridges, cuts = set(), set()
    t = 0
    for root in g.nodes():
        if root in disc:
            continue
        disc[root] = low[root] = t
        t += 1
        root_children = 0
        stack = [(root, None, iter(g.neighbors(root)))]
        while stack:
            u, p, it = stack[-1]
            descended = False
            for v in it:
                if v == p:
                    continue
                if v in disc:
                    low[u] = min(low[u], disc[v])
                else:
                    disc[v] = low[v] = t
                    t += 1
                    stack.append((v, u, iter(g.neighbors(v))))
                    if u == root:
                        root_children += 1
                    descended = True
                    break
            if descended:
                continue
            stack.pop()
            if stack:
                pu = stack[-1][0]
                low[pu] = min(low[pu], low[u])
                if low[u] > disc[pu]:
                    bridges.add(ek(pu, u))
                if pu != root and low[u] >= disc[pu]:
                    cuts.add(pu)
        if root_children > 1:
            cuts.add(root)
    return bridges, cuts


# ------------------------------------------------------------------- Dijkstra
def dijkstra(g, sources, weight="latency_ms", skip_nodes=(), skip_edges=()):
    """Multi-source Dijkstra.

    Returns (dist, parent, origin). `origin[v]` is the source that reaches v
    first. Because every node inherits `origin` from its parent, each source's
    cell is CONNECTED by construction - that is the property zoning relies on
    (a zone is a connected piece of the network, so its members can reach their
    hub without leaving the zone). Ties keep the first-found parent.
    """
    skip_nodes, skip_edges = set(skip_nodes), set(skip_edges)
    dist, parent, origin, heap = {}, {}, {}, []
    for s in sorted(_seq(sources)):
        if s in g.adj and s not in skip_nodes:
            dist[s], parent[s], origin[s] = 0.0, None, s
            heapq.heappush(heap, (0.0, s))
    done = set()
    while heap:
        d, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        for v in g.neighbors(u):
            if v in skip_nodes or ek(u, v) in skip_edges:
                continue
            nd = d + g.adj[u][v][weight]
            if nd < dist.get(v, math.inf) - 1e-12:
                dist[v], parent[v], origin[v] = nd, u, origin[u]
                heapq.heappush(heap, (nd, v))
    return dist, parent, origin


def path_to(parent, v):
    """Node list from the tree root to v (inclusive), from a parent map."""
    out = []
    while v is not None:
        out.append(v)
        v = parent[v]
    return out[::-1]


def all_pairs(g, weight="latency_ms"):
    """{u: {v: shortest distance}} by Dijkstra from every node."""
    return {u: dijkstra(g, u, weight)[0] for u in g.nodes()}


# ---------------------------------------------------------------- Kruskal MST
class UnionFind:
    def __init__(self, items):
        self.p = {x: x for x in items}
        self.s = {x: 1 for x in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.s[ra] < self.s[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        self.s[ra] += self.s[rb]
        return True


def kruskal(nodes, weighted_edges):
    """Minimum spanning forest. `weighted_edges` = iterable of (w, u, v).
    Returns [(u, v, w)] with u < v."""
    uf = UnionFind(nodes)
    out = []
    for w, u, v in sorted(weighted_edges):
        if uf.union(u, v):
            out.append((min(u, v), max(u, v), w))
    return out


# ------------------------------------------------------------------- max-flow
def _edmonds_karp(cap, s, t):
    """In-place Edmonds-Karp on a residual dict-of-dicts. Returns (flow,
    reachable-from-s set in the final residual graph)."""
    flow = 0
    while True:
        parent, dq = {s: None}, deque([s])
        while dq and t not in parent:
            u = dq.popleft()
            for v, c in cap[u].items():
                if c > 0 and v not in parent:
                    parent[v] = u
                    dq.append(v)
        if t not in parent:
            return flow, set(parent)
        b, v = math.inf, t
        while parent[v] is not None:
            b = min(b, cap[parent[v]][v])
            v = parent[v]
        v = t
        while parent[v] is not None:
            u = parent[v]
            cap[u][v] -= b
            cap[v][u] = cap[v].get(u, 0) + b
            v = u
        flow += b


def _residual(g, vertex, s, t, skip_nodes=()):
    skip_nodes = set(skip_nodes)
    cap = {}

    def arc(a, b, c):
        cap.setdefault(a, {})[b] = cap.get(a, {}).get(b, 0) + c
        cap.setdefault(b, {}).setdefault(a, 0)

    big = g.n + 5
    for u in g.nodes():
        if u in skip_nodes:
            continue
        if vertex:
            arc((u, 0), (u, 1), big if u in (s, t) else 1)
    for u, v, _ in g.edges():
        if u in skip_nodes or v in skip_nodes:
            continue
        if vertex:
            # Edge arcs must be effectively uncuttable, otherwise the min cut
            # could land on a LINK instead of a substation. The one exception
            # is a direct s-t link, which counts as exactly one path.
            c = 1 if {u, v} == {s, t} else big
            arc((u, 1), (v, 0), c)
            arc((v, 1), (u, 0), c)
        else:
            arc(u, v, 1)
            arc(v, u, 1)
    return cap


def disjoint_paths(g, s, t, vertex=False, skip_nodes=()):
    """Number of edge-disjoint (or, with vertex=True, internally vertex-disjoint)
    s-t paths. By Menger's theorem this equals the minimum number of links
    (substations) whose loss separates s from t."""
    if s == t:
        raise ValueError("s == t")
    cap = _residual(g, vertex, s, t, skip_nodes)
    src, dst = ((s, 1), (t, 0)) if vertex else (s, t)
    if src not in cap or dst not in cap:
        return 0
    return _edmonds_karp(cap, src, dst)[0]


def min_vertex_cut(g, s, t):
    """Smallest set of nodes (other than s, t) whose removal separates s from t.
    None if s and t are adjacent (no vertex cut exists)."""
    if g.has_edge(s, t):
        return None
    cap = _residual(g, True, s, t)
    _, reach = _edmonds_karp(cap, (s, 1), (t, 0))
    return {u for u in g.nodes() if (u, 0) in reach and (u, 1) not in reach}
