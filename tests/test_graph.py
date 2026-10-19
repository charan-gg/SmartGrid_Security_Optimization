"""
Cross-checks the hand-written graph routines against networkx on random graphs.
Run:  python -m unittest discover -s tests -v      (from the project root)

networkx is OPTIONAL. Without it the cross-checks are skipped and only the
self-contained property tests run.
"""
import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.design import graph as G  # noqa: E402

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None


def random_graph(n, p, seed, connected=True):
    rng = random.Random(seed)
    g = G.Graph()
    for i in range(n):
        g.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                g.add_edge(i, j, w=rng.uniform(1, 10), latency_ms=rng.uniform(1, 10))
    if connected:  # chain to guarantee connectivity
        for i in range(n - 1):
            if not g.has_edge(i, i + 1):
                g.add_edge(i, i + 1, w=rng.uniform(1, 10), latency_ms=rng.uniform(1, 10))
    return g


def to_nx(g, key="w"):
    h = nx.Graph()
    h.add_nodes_from(g.nodes())
    for u, v, a in g.edges():
        h.add_edge(u, v, **a)
    return h


class TestProperties(unittest.TestCase):
    def test_bfs_hops_on_path(self):
        g = G.Graph()
        for i in range(5):
            g.add_edge(i, i + 1, latency_ms=1.0)
        hops, _ = G.bfs(g, 0)
        self.assertEqual(hops, {i: i for i in range(6)})
        hops, _ = G.bfs(g, 0, skip_nodes={3})
        self.assertEqual(set(hops), {0, 1, 2})

    def test_bridges_on_path_and_cycle(self):
        path = G.Graph()
        for i in range(4):
            path.add_edge(i, i + 1)
        b, c = G.bridges_and_cut_vertices(path)
        self.assertEqual(len(b), 4)
        self.assertEqual(c, {1, 2, 3})
        ring = G.Graph()
        for i in range(5):
            ring.add_edge(i, (i + 1) % 5)
        self.assertEqual(G.bridges_and_cut_vertices(ring), (set(), set()))

    def test_voronoi_cells_are_connected(self):
        for seed in range(20):
            g = random_graph(30, 0.08, seed)
            hubs = [0, 10, 20]
            _, _, origin = G.dijkstra(g, hubs)
            for h in hubs:
                cell = {v for v, o in origin.items() if o == h}
                sub = g.subgraph(cell)
                self.assertTrue(G.is_connected(sub), f"seed {seed} hub {h}")

    def test_menger_on_ring(self):
        ring = G.Graph()
        for i in range(6):
            ring.add_edge(i, (i + 1) % 6)
        self.assertEqual(G.disjoint_paths(ring, 0, 3), 2)
        self.assertEqual(G.disjoint_paths(ring, 0, 3, vertex=True), 2)
        self.assertEqual(G.min_vertex_cut(ring, 0, 3) & {1, 2, 4, 5}, G.min_vertex_cut(ring, 0, 3))
        self.assertEqual(len(G.min_vertex_cut(ring, 0, 3)), 2)
        self.assertIsNone(G.min_vertex_cut(ring, 0, 1))

    def test_kruskal_spans_and_is_acyclic(self):
        g = random_graph(25, 0.2, 3)
        t = G.kruskal(g.nodes(), [(a["w"], u, v) for u, v, a in g.edges()])
        self.assertEqual(len(t), 24)
        tree = G.Graph()
        for u, v, _ in t:
            tree.add_edge(u, v)
        self.assertTrue(G.is_connected(tree))


@unittest.skipIf(nx is None, "networkx not installed")
class TestAgainstNetworkx(unittest.TestCase):
    def test_mst_weight(self):
        for seed in range(15):
            g = random_graph(30, 0.15, seed)
            ours = sum(w for _, _, w in G.kruskal(g.nodes(), [(a["w"], u, v) for u, v, a in g.edges()]))
            ref = nx.minimum_spanning_tree(to_nx(g), weight="w").size(weight="w")
            self.assertAlmostEqual(ours, ref, places=9)

    def test_dijkstra_distances(self):
        for seed in range(15):
            g = random_graph(30, 0.12, seed)
            dist, _, _ = G.dijkstra(g, 0)
            ref = nx.single_source_dijkstra_path_length(to_nx(g), 0, weight="latency_ms")
            self.assertEqual(set(dist), set(ref))
            for v in ref:
                self.assertAlmostEqual(dist[v], ref[v], places=9)

    def test_bridges_and_articulation_points(self):
        for seed in range(25):
            g = random_graph(25, 0.09, seed, connected=False)
            h = to_nx(g)
            b, c = G.bridges_and_cut_vertices(g)
            self.assertEqual(b, {G.ek(u, v) for u, v in nx.bridges(h)})
            self.assertEqual(c, set(nx.articulation_points(h)))

    def test_components(self):
        for seed in range(15):
            g = random_graph(30, 0.04, seed, connected=False)
            ours = sorted(map(sorted, G.connected_components(g)))
            ref = sorted(map(sorted, nx.connected_components(to_nx(g))))
            self.assertEqual(ours, ref)

    def test_disjoint_paths_and_vertex_cut(self):
        for seed in range(12):
            g = random_graph(18, 0.22, seed)
            h = to_nx(g)
            for s, t in [(0, 17), (2, 9), (5, 14)]:
                self.assertEqual(G.disjoint_paths(g, s, t), len(list(nx.edge_disjoint_paths(h, s, t))))
                if not g.has_edge(s, t):
                    cut = G.min_vertex_cut(g, s, t)
                    self.assertEqual(len(cut), nx.node_connectivity(h, s, t))
                    # removing the cut really separates s from t
                    hops, _ = G.bfs(g, s, skip_nodes=cut)
                    self.assertNotIn(t, hops)
                    self.assertEqual(G.disjoint_paths(g, s, t, vertex=True), len(cut))


if __name__ == "__main__":
    unittest.main()
