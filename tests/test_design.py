"""
Invariants of the synthesised grid. These are properties the DESIGN promises, not
snapshots of one run's numbers, so they hold for any seed / size.

Run:  python -m unittest discover -s tests -v      (from the project root)
"""
import itertools
import os
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.design import design as D  # noqa: E402
from src.design import evaluate as EV  # noqa: E402
from src.design import graph as G  # noqa: E402
from src.design import observability as OB  # noqa: E402
from src.design import power as PW  # noqa: E402
from src.design import security as SE  # noqa: E402
from src.design.params import default_assumptions, load_measured  # noqa: E402
from src.design.sites import synth_sites  # noqa: E402

A0 = default_assumptions()
# Hermetic on purpose: these tests check properties of the design, so they must not read whatever
# measurements sit in outputs/ (E4 latencies differ by CPU, and on a slower laptop they once flipped a
# verdict). A folder with no measurements makes the loader return its fixed placeholders.
M = load_measured(output_dir=Path(__file__).parent / "_no_measurements")


def make(n=24, seed=1, **ov):
    A = replace(A0, n_sites=n, region_km=A0.region_km * (n / A0.n_sites) ** 0.5, **ov)
    return D.design_grid(A, M, seed=seed), A


class TestPhysical(unittest.TestCase):
    def test_n1_secure_for_many_seeds_and_sizes(self):
        for n, seed, mode in itertools.product((8, 20, 45), range(4), ("vertex", "edge")):
            d, _ = make(n, seed, n1_mode=mode)
            g = d.power.g
            br, cv = G.bridges_and_cut_vertices(g)
            self.assertEqual(br, set(), (n, seed, mode))
            if mode == "vertex":
                self.assertEqual(cv, set(), (n, seed, mode))
            self.assertTrue(G.is_connected(g))

    def test_cost_ordering_radial_designed_mesh(self):
        d, _ = make(30, 3)
        a = d.power.audit
        self.assertLess(a["radial (MST only)"]["cost_kusd"], a["designed (N-1 secure)"]["cost_kusd"])
        self.assertLess(a["designed (N-1 secure)"]["cost_kusd"], a["mesh (all candidate lines)"]["cost_kusd"])

    def test_designed_contains_the_mst(self):
        d, _ = make(30, 3)
        radial = d.power.variants["radial (MST only)"]
        for u, v, _a in radial.edges():
            self.assertTrue(d.power.g.has_edge(u, v))

    def test_single_outage_really_islands_nothing(self):
        d, _ = make(25, 2)
        g = d.power.g
        for f in g.nodes():
            self.assertTrue(G.is_connected(g, skip_nodes={f}))
        for u, v, _a in g.edges():
            self.assertTrue(G.is_connected(g, skip_edges={(u, v)}))


class TestObservability(unittest.TestCase):
    def path(self, n):
        g = G.Graph()
        for i in range(n - 1):
            g.add_edge(i, i + 1)
        return g

    def test_closed_neighbourhood_and_zero_injection(self):
        g = self.path(4)                       # 0-1-2-3
        self.assertEqual(OB.observed_set(g, {1}), {0, 1, 2})
        # node 2 is zero-injection: its closed nbhd {1,2,3} has only 3 missing -> deduced
        self.assertEqual(OB.observed_set(g, {1}, zib=[2]), {0, 1, 2, 3})

    def test_exact_matches_brute_force_and_beats_or_ties_greedy(self):
        for seed in range(6):
            d, _ = make(10, seed)
            g = d.power.g
            ids = g.nodes()
            best = next(r for r in range(1, len(ids) + 1)
                        for c in itertools.combinations(ids, r)
                        if len(OB.observed_set(g, set(c))) == g.n)
            greedy, _o = OB.greedy_pmus(g)
            self.assertGreaterEqual(len(OB.prune(g, greedy, 1, ())), best)
            exact = OB.exact_pmus(g, 1)
            if exact is not None:               # scipy present
                self.assertEqual(len(exact), best)

    def test_plan_is_valid_and_minimal_under_single_removal(self):
        for k in (1, 2):
            d, _ = make(30, 5, pmu_obs_k=k)
            g, plan = d.power.g, d.pmu
            self.assertTrue(OB._satisfied(g, plan.pmus, k, d.zib if k == 1 else ()))
            for p in plan.pmus:
                self.assertFalse(OB._satisfied(g, plan.pmus - {p}, k, d.zib if k == 1 else ()),
                                 f"PMU {p} was redundant")

    def test_two_fold_costs_more_than_one_fold(self):
        d1, _ = make(30, 5, pmu_obs_k=1)
        d2, _ = make(30, 5, pmu_obs_k=2)
        self.assertGreater(d2.pmu.n_final, d1.pmu.n_final)
        # 2-fold survives any single PMU loss without losing observability
        self.assertEqual(d2.pmu_loss["max_unobserved_after_loss"], 0)


class TestHierarchy(unittest.TestCase):
    def test_zones_partition_the_grid_and_are_connected(self):
        for n, seed in ((20, 0), (40, 1), (60, 2)):
            d, _ = make(n, seed)
            h, cg = d.hier, d.comm
            members = sorted(v for z in h.zones.values() for v in z)
            self.assertEqual(members, cg.nodes())
            for hub, z in h.zones.items():
                self.assertIn(hub, z)
                self.assertTrue(G.is_connected(cg.subgraph(z)), f"zone of {hub} disconnected")

    def test_chosen_design_re_verifies_its_own_constraints(self):
        d, A = make(40, 1)
        h, bays = d.hier, d.power.bays
        total = sum(bays.values())
        self.assertTrue(h.feasible)
        for hub, z in h.zones.items():
            self.assertLessEqual(sum(bays[v] for v in z) / total, A.max_zone_share + 1e-9)
        for v in d.comm.nodes():                 # walk each station's path to its hub
            hops, x = 0, v
            while h.to_hub[x] is not None:
                x, hops = h.to_hub[x], hops + 1
            self.assertEqual(x, h.zone_of[v])
            self.assertLessEqual(hops, A.max_zone_hops)
        if h.k > 1:
            self.assertTrue(all(len(z) >= A.min_zone_stations for z in h.zones.values()))

    def test_cc_is_not_a_hub_unless_forced(self):
        for n, seed in itertools.product((20, 40, 60), range(6)):
            d, _ = make(n, seed)
            if d.hier.k > 1:
                self.assertNotIn(d.hier.cc, d.hier.hubs, (n, seed))
            else:
                self.assertEqual(d.hier.cc, d.hier.hubs[0])

    def test_levels_are_an_output_not_an_assumption(self):
        loose, _ = make(40, 1, max_zone_share=1.0, max_zone_hops=99)
        tight, _ = make(40, 1)
        self.assertEqual(loose.hier.k, 1)
        self.assertEqual(loose.hier.n_levels, 3)
        self.assertEqual(tight.hier.n_levels, 4)

    def test_slower_hardware_never_reduces_zone_count(self):
        a, _ = make(40, 1, hardware_slowdown=1.0)
        b, _ = make(40, 1, hardware_slowdown=200.0)
        self.assertGreaterEqual(b.hier.k, a.hier.k)


class TestEvaluation(unittest.TestCase):
    def test_traffic_is_conserved(self):
        d, A = make(30, 4)
        h, bays = d.hier, d.power.bays
        load = EV.link_loads(h, bays, M, A, "B")
        expected = sum(bays[v] for v in bays if v != h.cc) * M.raw_Bps_per_bay
        self.assertAlmostEqual(EV._ingress(load, h.cc, d.comm), expected, places=3)

    def test_a_station_never_emits_more_verdicts_than_rows_it_scores(self):
        d, A = make(30, 4)
        self.assertLessEqual(EV.verdict_rate_hz(M, A), M.sample_rate_hz)
        fast = replace(A, verdict_hz=1e6)
        self.assertEqual(EV.verdict_rate_hz(M, fast), M.sample_rate_hz)
        slow = replace(A, verdict_hz=0.5 * M.sample_rate_hz)
        self.assertEqual(EV.verdict_rate_hz(M, slow), 0.5 * M.sample_rate_hz)

    def test_tiering_reduces_peak_link_load(self):
        d, _ = make(40, 1)
        c = {r["deployment"][:2].strip(): r for r in d.comparison}
        self.assertLess(c["C"]["peak_link_KBps"], c["B"]["peak_link_KBps"])
        self.assertLess(c["C"]["blast_share"], c["B"]["blast_share"])

    def test_single_non_cc_failure_never_cuts_off_a_2connected_comm_graph(self):
        d, _ = make(30, 4)
        for r in d.security["node_rows"]:
            if not r["is_cc"]:
                self.assertAlmostEqual(r["central"], 1.0)

    def test_menger_connectivity_reported(self):
        d, _ = make(30, 4)
        self.assertGreaterEqual(d.security["diversity"]["min_vertex_disjoint_to_cc"], 2)

    def test_backup_cc_is_a_different_site(self):
        d, _ = make(30, 4)
        self.assertNotEqual(d.security["backup_cc"], d.hier.cc)

    def test_deterministic(self):
        a, _ = make(30, 7)
        b, _ = make(30, 7)
        self.assertEqual(D.summarize(a), D.summarize(b))


if __name__ == "__main__":
    unittest.main()
