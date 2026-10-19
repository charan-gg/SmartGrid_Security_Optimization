"""
Default-vs-optimized machinery: tier detectors (pure functions), standard-case loader, default
builders, the failure-scenario model and the verdict rules. Properties, not snapshots.

Run:  python -m unittest discover -s tests -v      (from the project root)
"""
import itertools
import math
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config as C  # noqa: E402
from src import detectors as DT  # noqa: E402
from src import modalities as MD  # noqa: E402
from src.design import baseline as BL  # noqa: E402
from src.design import cases as CS  # noqa: E402
from src.design import compare as CP  # noqa: E402
from src.design import design as D  # noqa: E402
from src.design import graph as G  # noqa: E402
from src.design import observability as OB  # noqa: E402
from src.design.params import default_assumptions, load_measured  # noqa: E402
from src.design.sites import synth_sites  # noqa: E402

A0 = default_assumptions()
# Hermetic on purpose (see test_design.py): never read this machine's measurements from outputs/.
M = load_measured(output_dir=Path(__file__).parent / "_no_measurements")
REC =CP.Recalls(station=np.array([0.50, 0.55]), fused=np.array([0.60, 0.58]))
_CACHE = {}


def layout(n=24, seed=1):
    """(A, sites, design, candidates) - built once per (n, seed)."""
    if (n, seed) not in _CACHE:
        A = CP.scaled(A0, n)
        sites = synth_sites(n, seed, A.region_km, A.frac_gen, A.frac_junction)
        d = D.design_grid(A, M, seed=seed, sites=sites, security=False)
        _CACHE[(n, seed)] = (A, sites, d, CP.build_candidates(d, A, M, sites, seed))
    return _CACHE[(n, seed)]


# ------------------------------------------------------------------ detectors
class TestDetectorMaths(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.y = np.array(["Normal"] * 400 + ["Fault"] * 100 + ["Masquerade"] * 100
                          + ["Injection"] * 100 + ["Replay"] * 100)
        self.score = np.concatenate([rng.beta(1, 8, 500), rng.beta(5, 2, 100), rng.beta(3, 3, 100),
                                     rng.beta(2, 4, 100)])
        self.neg = np.isin(self.y, ["Normal", "Fault"])

    def test_realised_false_alert_rate_never_exceeds_the_budget(self):
        for fpr in (0.0, 0.005, 0.01, 0.05, 0.2):
            thr = DT.threshold_at_fpr(self.score, self.neg, fpr)
            self.assertLessEqual((self.score[self.neg] >= thr).mean(), fpr + 1e-12, fpr)

    def test_a_stricter_budget_never_lowers_the_threshold_or_raises_recall(self):
        prev_thr, prev_cov = -np.inf, 2.0
        for fpr in (0.2, 0.05, 0.01, 0.0):
            thr = DT.threshold_at_fpr(self.score, self.neg, fpr)
            cov = DT.coverage(DT.recalls_at(self.score, self.y, thr))
            self.assertGreaterEqual(thr, prev_thr)
            self.assertLessEqual(cov, prev_cov + 1e-12)
            prev_thr, prev_cov = thr, cov

    def test_zero_budget_flags_no_negative_at_all(self):
        thr = DT.threshold_at_fpr(self.score, self.neg, 0.0)
        self.assertEqual(int((self.score[self.neg] >= thr).sum()), 0)

    def test_recall_is_per_class_and_absent_class_is_nan_not_zero(self):
        r = DT.recalls_at(np.array([0.9, 0.1, 0.8]), np.array(["Replay", "Replay", "Injection"]), 0.5)
        self.assertEqual(r["Replay"], 0.5)
        self.assertEqual(r["Injection"], 1.0)
        self.assertTrue(math.isnan(r["Masquerade"]))
        self.assertAlmostEqual(DT.coverage(r), 0.75)           # NaN class ignored, not averaged in as 0

    def test_ties_are_resolved_against_the_attacker_not_for_them(self):
        score = np.array([0.5] * 10 + [0.5] * 3)
        y = np.array(["Normal"] * 10 + ["Replay"] * 3)
        thr = DT.threshold_at_fpr(score, y == "Normal", 0.5)
        self.assertEqual((score[:10] >= thr).mean(), 0.0)      # all-tied negatives: flag none rather than 100%

    def test_bootstrap_is_reproducible_and_rederives_the_threshold_each_time(self):
        block = np.repeat(np.arange(20), 25)
        a = DT.block_bootstrap(self.score, self.y, block, 0.01, 40, 5)
        b = DT.block_bootstrap(self.score, self.y, block, 0.01, 40, 5)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(len(a), 40)
        self.assertLessEqual(a["realised_fpr"].max(), 0.01 + 1e-12)
        self.assertGreater(a["threshold"].nunique(), 1)         # a fixed threshold would understate the spread

    def test_bootstrap_resamples_whole_blocks_only(self):
        score = np.arange(100, dtype=float)
        y = np.array(["Normal"] * 50 + ["Replay"] * 50)
        block = np.repeat(np.arange(10), 10)
        out = DT.block_bootstrap(score, y, block, 0.1, 30, 1)
        self.assertTrue(out["coverage"].between(0, 1).all())

    def test_tier_columns_come_from_the_views_and_never_include_metadata(self):
        feats = (["sqNum", "stnum", "state_cb", "sqDiff", "stDiff", "timeLastMsg", "recentChange",
                  "MU1Cs", "consistency", "any_relay", "MU3Log", "MU1VoltageAngleA", "threePhaseVMU1"])
        views = MD.build_views(feats)
        cols = DT.tier_columns(views, ["consistency", "state_cb", "any_relay"])
        for tier, c in cols.items():
            self.assertFalse(set(c) & {"t_ms", C.TARGET, C.FAMILY_COL, C.GROUP_KEY, C.TIME_COL, C.LABEL_COL}, tier)
        self.assertEqual(set(cols["station"]) & set(views["process"]), set())      # station has no phasors
        self.assertEqual(set(cols["fused"]), set(feats))
        with self.assertRaises(ValueError):
            DT.tier_columns(views, [])

    def test_stable_features_are_the_ones_selected_in_every_fold(self):
        with tempfile.TemporaryDirectory() as d:
            pd.DataFrame({"0": {"a": C.N_SPLITS, "b": C.N_SPLITS - 1, "c": C.N_SPLITS}}).to_csv(
                Path(d) / "e3_feature_stability.csv")
            self.assertEqual(sorted(DT.stable_features(Path(d))), ["a", "c"])

    def test_missing_step6_output_says_how_to_fix_it(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(FileNotFoundError, "step6_train_detectors"):
                DT.load_recall_draws(Path(d))


# ---------------------------------------------------------------------- cases
class TestCases(unittest.TestCase):
    def test_every_shipped_case_loads_and_is_well_formed(self):
        names = CS.available()
        self.assertGreaterEqual(len(names), 5)
        for name in names:
            c = CS.load_case(name)
            n = len(c.sites)
            self.assertEqual([s.id for s in c.sites], list(range(n)), name)
            side = CS.region_km_for(n)
            for s in c.sites:
                self.assertIn(s.kind, ("gen", "load", "junction"))
            pts = [(s.x_km, s.y_km) for s in c.sites]
            self.assertGreaterEqual(min(math.dist(p, q) for p, q in itertools.combinations(pts, 2)), 0.99, name)
            self.assertLessEqual(max(max(p) for p in pts), side * 1.2, name)
            g = G.Graph()
            for s in c.sites:
                g.add_node(s.id)
            for u, v in c.edges:
                g.add_edge(u, v)
            self.assertTrue(G.is_connected(g), name)

    def test_the_layout_is_rescaled_to_the_project_region(self):
        for name in CS.available()[:4]:
            c = CS.load_case(name)
            xs, ys = [s.x_km for s in c.sites], [s.y_km for s in c.sites]
            span = max(max(xs) - min(xs), max(ys) - min(ys))
            self.assertGreaterEqual(span, 0.9 * CS.region_km_for(len(c.sites)), name)
            self.assertLessEqual(span, 1.1 * CS.region_km_for(len(c.sites)), name)

    def _case_dir(self, sites, lines):
        import csv
        d = tempfile.TemporaryDirectory()
        base = Path(d.name)
        with open(base / "t_sites.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "name", "x", "y", "kind", "mw"])
            for i, kind in enumerate(sites):
                w.writerow([i, f"s{i}", i * 3.0, (i * 7) % 5, kind, 10.0 if kind == "gen" else -5.0])
        with open(base / "t_lines.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["from_site", "to_site", "n_circuits", "rating_mva"])
            for u, v in lines:
                w.writerow([u, v, 1, 100.0])
        return d

    def test_the_loader_refuses_malformed_cases(self):
        from unittest import mock
        ok = ["gen", "load", "load", "load", "load"]
        for label, sites, lines, msg in (
                ("disconnected", ok, [(0, 1), (1, 2), (3, 4)], "not connected"),
                ("corridor out of range", ok, [(0, 1), (1, 2), (2, 3), (3, 9)], "bad corridor"),
                ("self loop", ok, [(0, 1), (1, 2), (2, 3), (3, 3), (3, 4)], "bad corridor"),
                ("too few sites", ok[:3], [(0, 1), (1, 2)], "at least 4")):
            d = self._case_dir(sites, lines)
            try:
                with mock.patch.object(C, "GRID_DIR", Path(d.name)):
                    with self.assertRaisesRegex(ValueError, msg, msg=label):
                        CS.load_case("t")
            finally:
                d.cleanup()
        d = self._case_dir(ok, [(0, 1), (1, 2), (2, 3), (3, 4)])
        try:
            with mock.patch.object(C, "GRID_DIR", Path(d.name)):
                c = CS.load_case("t")
                self.assertEqual(len(c.sites), 5)
                self.assertEqual(CS.available(), ["t"])
        finally:
            d.cleanup()

    def test_a_missing_case_names_the_alternatives(self):
        with self.assertRaisesRegex(FileNotFoundError, "available"):
            CS.load_case("no-such-grid")

    def test_nudging_keeps_a_minimum_separation_and_is_deterministic(self):
        pts = [(0.0, 0.0)] * 6 + [(10.0, 10.0)]
        a, b = CS._nudge_apart(pts, 1.0), CS._nudge_apart(pts, 1.0)
        self.assertEqual(a, b)
        self.assertGreaterEqual(min(math.dist(p, q) for p, q in itertools.combinations(a, 2)), 1.0 - 1e-9)

    def test_rescaling_follows_the_project_convention(self):
        self.assertAlmostEqual(CS.region_km_for(40), 300.0)
        self.assertAlmostEqual(CS.region_km_for(160), 600.0)


# ------------------------------------------------------------------- baselines
class TestDefaults(unittest.TestCase):
    def test_naive_default_is_a_tree_with_a_pmu_everywhere_and_one_flat_control_centre(self):
        A, sites, d, c = layout()
        naive = c["naive"]
        n = len(sites)
        self.assertEqual(naive.power.g.m, n - 1)
        br, _ = G.bridges_and_cut_vertices(naive.power.g)
        self.assertEqual(len(br), n - 1)                                     # every line is a single point of failure
        self.assertEqual(len(naive.pmu.pmus), n)
        self.assertEqual(naive.h.k, 1)
        self.assertEqual(naive.h.cc, BL.naive_cc(sites))
        self.assertEqual(naive.h.metrics["max_zone_share"], 1.0)
        self.assertIsNone(naive.backup)

    def test_budget_matched_spends_the_same_but_not_more(self):
        A, sites, d, c = layout()
        target, got = c["optimized"].costs["total"], c["budget-matched"].costs["total"]
        self.assertLessEqual(got, target + 1e-6)
        self.assertGreaterEqual(got, 0.9 * target)
        mst = {(u, v) for u, v, _ in c["naive"].power.g.edges()}
        self.assertTrue(mst <= {(u, v) for u, v, _ in c["budget-matched"].power.g.edges()})
        self.assertIsNotNone(c["budget-matched"].backup)
        self.assertNotEqual(c["budget-matched"].backup, c["budget-matched"].h.cc)

    def test_budget_matched_lines_are_the_shortest_available_extras(self):
        A, sites, d, c = layout()
        _, mst, pool = CS_pool(sites, A)
        extra = sorted(pool[e] for e in pool if e not in mst)
        chosen = sorted(a["length_km"] for u, v, a in c["budget-matched"].power.g.edges() if (u, v) not in mst)
        self.assertEqual(chosen, extra[:len(chosen)])

    def test_optimized_is_n1_secure_and_the_naive_default_is_not(self):
        A, sites, d, c = layout()
        self.assertEqual(G.bridges_and_cut_vertices(c["optimized"].power.g), (set(), set()))
        self.assertGreater(len(G.bridges_and_cut_vertices(c["naive"].power.g)[0]), 0)

    def test_central_detection_is_cheaper_than_tiered_and_a_standby_costs_one_more_control_centre(self):
        A, sites, d, c = layout()
        pw, comm, h = c["naive"].power, c["naive"].comm, c["naive"].h
        pm = BL.all_pmus(pw.g)
        cen = BL.EV.cost_accounts(pw, pm, comm, pw.bays, h, A, "central")
        bk = BL.EV.cost_accounts(pw, pm, comm, pw.bays, h, A, "central_backup")
        tier = BL.EV.cost_accounts(pw, pm, comm, pw.bays, h, A, "tiered")
        self.assertAlmostEqual(bk["compute_cc"] - cen["compute_cc"], A.cc_kusd)
        self.assertGreater(tier["detection_plane"], cen["detection_plane"])
        self.assertEqual(cen["compute_hubs"], 0.0)

    def test_naive_control_centre_is_the_largest_site(self):
        A, sites, d, c = layout()
        biggest = max(abs(s.mw) for s in sites)
        self.assertEqual(abs(next(s for s in sites if s.id == BL.naive_cc(sites)).mw), biggest)

    def test_a_tiered_deployment_on_any_lines_gets_the_minimum_pmu_set_not_one_per_site(self):
        A, sites, d, c = layout()
        for name in ("radial+tiered", "bm-lines+tiered"):
            cand = c[name]
            self.assertLess(len(cand.pmu.pmus), len(sites), name)
            self.assertEqual(OB.observed_set(cand.power.g, cand.pmu.pmus, cand.zib), set(cand.power.g.nodes()), name)
            self.assertEqual(len(cand.pmu.pmus), OB.place_pmus(cand.power.g, A.pmu_obs_k, A.use_zib).n_final, name)
        self.assertEqual(len(c["naive"].pmu.pmus), len(sites))

    def test_the_budget_matched_default_has_the_same_price_class_as_the_optimized_grid(self):
        A, sites, d, c = layout()
        res = {k: CP.metric_values(CP.evaluate(v, 1), REC) for k, v in c.items()}
        v = CP.verdict_table([res], A, "budget-matched", J=100)
        row = v[v.metric == "cost_total_kusd"].iloc[0]
        self.assertLess(abs(row["diff"]), 0.05)               # RELATIVE difference
        self.assertEqual(row["outcome"], "tie")

    def test_unknown_deployment_is_rejected(self):
        A, sites, d, c = layout()
        with self.assertRaises(ValueError):
            BL.make_candidate("x", "radial", c["naive"].power, "peer-to-peer", A, M, sites)


def CS_pool(sites, A):
    from src.design import power as PW
    return PW.candidate_pool(sites, A)


# ------------------------------------------------------------------- scenarios
class TestScenarios(unittest.TestCase):
    def test_shares_are_proportions_and_delivery_never_exceeds_detection(self):
        A, sites, d, c = layout()
        for cand in c.values():
            for failed in [set(), {0}, {1, 2}, set(list(cand.comm.nodes())[:6])]:
                s = CP.shares(cand, failed)
                self.assertLessEqual(s["det_f"] + s["det_s"], 1 + 1e-9)
                self.assertLessEqual(s["del_f"], s["det_f"] + 1e-9)
                self.assertLessEqual(s["del_s"], s["det_s"] + 1e-9)
                self.assertTrue(all(v >= -1e-12 for v in s.values()))

    def test_nothing_alive_means_no_scenario(self):
        A, sites, d, c = layout()
        self.assertIsNone(CP.shares(c["optimized"], set(c["optimized"].comm.nodes())))

    def test_intact_grid_is_fully_covered_by_every_design(self):
        A, sites, d, c = layout()
        for name, cand in c.items():
            s = CP.shares(cand, set())
            self.assertAlmostEqual(s["det_f"] + s["det_s"], 1.0, msg=name)
            self.assertAlmostEqual(s["del_f"] + s["del_s"], 1.0, msg=name)

    def test_central_design_goes_blind_when_its_only_control_centre_fails(self):
        A, sites, d, c = layout()
        naive = c["naive"]
        s = CP.shares(naive, {naive.h.cc})
        self.assertEqual(s["det_f"] + s["det_s"], 0.0)

    def test_a_standby_control_centre_keeps_a_central_design_alive(self):
        A, sites, d, c = layout()
        bm = c["budget-matched"]
        s = CP.shares(bm, {bm.h.cc})
        self.assertGreater(s["det_f"], 0.0)

    def test_tiered_detects_locally_whatever_the_network_does(self):
        A, sites, d, c = layout()
        opt = c["optimized"]
        s = CP.shares(opt, {opt.h.cc} | set(opt.h.hubs))
        self.assertAlmostEqual(s["det_f"] + s["det_s"], 1.0)

    def test_tiered_never_detects_less_than_central_on_the_same_lines(self):
        A, sites, d, c = layout()
        cen, tie = c["opt-lines+central"], c["optimized"]
        for f in cen.comm.nodes():
            a, b = CP.shares(cen, {f}), CP.shares(tie, {f})
            self.assertGreaterEqual(b["det_f"] + b["det_s"], a["det_f"] + a["det_s"] - 1e-9)

    def test_more_pmus_never_lower_the_fused_share(self):
        A, sites, d, c = layout()
        opt = c["optimized"]
        base = CP.shares(opt, {3})
        extra = {max(set(opt.comm.nodes()) - set(opt.pmu.pmus))} | set(opt.pmu.pmus)
        more = BL.Candidate(**{**opt.__dict__, "pmu": OB.PMUPlan(pmus=extra, k=1, method="t", n_standard=0,
                                                                 n_greedy=0, n_final=len(extra),
                                                                 optimal_standard=False, observed_frac=1.0, curve=[])})
        self.assertGreaterEqual(CP.shares(more, {3})["det_f"], base["det_f"] - 1e-12)

    def test_strict_mode_credits_only_a_substations_own_pmu(self):
        A, sites, d, c = layout()
        opt = c["optimized"]
        loose, strict = CP.shares(opt, set(), True), CP.shares(opt, set(), False)
        self.assertGreaterEqual(loose["det_f"], strict["det_f"])
        own = sum(opt.power.bays[v] for v in opt.pmu.pmus) / sum(opt.power.bays.values())
        self.assertAlmostEqual(strict["det_f"], own)

    def test_random_pairs_are_identical_across_candidates_and_distinct(self):
        A, sites, d, c = layout()
        a = CP.random_pairs(c["naive"].comm.nodes(), 30, 4)
        b = CP.random_pairs(c["optimized"].comm.nodes(), 30, 4)
        self.assertEqual(a, b)
        self.assertEqual(len(set(a)), 30)

    def test_the_adversary_gets_the_same_number_of_pairs_against_every_design(self):
        A, sites, d, c = layout(30, 2)
        for cand in c.values():
            singles = {v: CP.shares(cand, {v}) for v in cand.comm.nodes()}
            self.assertLessEqual(len(CP.targeted_pairs(cand, singles)), CP.N2_TARGETED_KEEP)

    def test_disc_failures_remove_every_site_within_the_radius(self):
        A, sites, d, c = layout()
        opt = c["optimized"]
        for f in CP.disc_sets(opt, 50.0):
            self.assertTrue(len(f) >= 1)
        self.assertEqual(max(len(f) for f in CP.disc_sets(opt, 1e9)), len(sites))


# -------------------------------------------------------------------- verdict
class TestVerdict(unittest.TestCase):
    def test_classify_boundaries(self):
        self.assertEqual(CP.classify(0.02, 0.05, 0.01, True), "optimized better")
        self.assertEqual(CP.classify(-0.05, -0.02, 0.01, True), "default better")
        self.assertEqual(CP.classify(-0.005, 0.005, 0.01, True), "tie")
        self.assertEqual(CP.classify(0.005, 0.03, 0.01, True), "inconclusive")
        self.assertEqual(CP.classify(-0.05, -0.02, 0.01, False), "optimized better")   # lower is better
        self.assertEqual(CP.classify(0.02, 0.05, 0.01, False), "default better")
        self.assertEqual(CP.classify(0.01, 0.01, 0.01, True), "tie")                   # exactly at tolerance is not a win

    def test_bootstrap_of_a_constant_difference_is_that_constant(self):
        m, lo, hi = CP.paired_bootstrap(np.full(20, 0.3))
        self.assertAlmostEqual(m, 0.3), self.assertAlmostEqual(lo, 0.3), self.assertAlmostEqual(hi, 0.3)

    def test_bootstrap_is_reproducible_and_widens_with_noise(self):
        rng = np.random.default_rng(0)
        tight, wide = 0.1 + 0.001 * rng.standard_normal((30, 50)), 0.1 + 0.2 * rng.standard_normal((30, 50))
        self.assertEqual(CP.paired_bootstrap(tight, seed=3), CP.paired_bootstrap(tight, seed=3))
        t, w = CP.paired_bootstrap(tight), CP.paired_bootstrap(wide)
        self.assertLess(t[2] - t[1], w[2] - w[1])

    def test_recalls_never_let_fused_fall_below_station(self):
        draws = pd.DataFrame({"b": [0, 0, 1, 1], "tier": ["station", "fused"] * 2,
                              "coverage": [0.6, 0.5, 0.4, 0.5]})
        r = CP.Recalls.from_draws(draws)
        self.assertTrue((r.fused >= r.station).all())
        self.assertEqual(list(r.fused), [0.6, 0.5])

    def test_overall_rule_can_say_no(self):
        def tally(**kw):
            base = dict(security_outcome=(5, 0, 0), security_objective=(5, 0, 0), performance=(0, 3, 0))
            base.update(kw)
            rows = []
            for fam, (b, t, w) in base.items():
                rows.append(dict(family=fam.replace("_", "-"), metrics=b + t + w, optimized_better=b,
                                 tie=t, default_better=w, inconclusive=0))
            return pd.DataFrame(rows)
        self.assertEqual(CP.overall(tally())[0], "OPTIMIZED BETTER")
        self.assertEqual(CP.overall(tally(security_outcome=(1, 4, 0)))[0], "OPTIMIZED BETTER (narrow)")
        self.assertEqual(CP.overall(tally(security_outcome=(0, 5, 0)))[0], "NOT ESTABLISHED")
        self.assertEqual(CP.overall(tally(security_objective=(4, 0, 1)))[0], "NOT ESTABLISHED")
        self.assertEqual(CP.overall(tally(performance=(0, 2, 1)))[0], "NOT ESTABLISHED")

    def test_objectives_alone_cannot_establish_a_win(self):
        t = pd.DataFrame([dict(family="security-outcome", metrics=5, optimized_better=0, tie=5, default_better=0, inconclusive=0),
                          dict(family="security-objective", metrics=5, optimized_better=5, tie=0, default_better=0, inconclusive=0),
                          dict(family="performance", metrics=3, optimized_better=0, tie=3, default_better=0, inconclusive=0)])
        self.assertEqual(CP.overall(t)[0], "NOT ESTABLISHED")

    def test_machinery_returns_no_when_a_design_is_compared_with_itself_or_reversed(self):
        A, sites, d, c = layout()
        res = {k: CP.metric_values(CP.evaluate(v, 1), REC) for k, v in c.items()}
        same = CP.verdict_table([res], A, "optimized", J=100)
        self.assertEqual(set(same[same.tallied].outcome), {"tie"})
        self.assertEqual(CP.overall(CP.family_tally(same))[0], "NOT ESTABLISHED")
        swapped = {"optimized": res["naive"], "naive": res["optimized"]}
        rev = CP.verdict_table([swapped], A, "naive", J=100)
        lab, why = CP.overall(CP.family_tally(rev))
        self.assertEqual(lab, "NOT ESTABLISHED")
        self.assertTrue(any("worse" in w for w in why))

    def test_price_is_never_netted_against_security(self):
        A, sites, d, c = layout()
        res = {k: CP.metric_values(CP.evaluate(v, 1), REC) for k, v in c.items()}
        v = CP.verdict_table([res], A, "naive", J=100)
        tally = CP.family_tally(v)
        self.assertIn("price", set(tally.family))
        self.assertEqual(v[v.metric == "cost_total_kusd"].outcome.iloc[0], "default better")
        self.assertTrue(CP.overall(tally)[0].startswith("OPTIMIZED BETTER"))     # a higher price does not veto it

    def test_evaluation_is_deterministic(self):
        A, sites, d, c = layout()
        a = CP.metric_values(CP.evaluate(c["optimized"], 3), REC)
        b = CP.metric_values(CP.evaluate(c["optimized"], 3), REC)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k], k)

    def test_the_optimized_grid_removes_every_single_point_of_failure_on_many_layouts(self):
        for n, seed in itertools.product((16, 30), range(3)):
            A, sites, d, c = layout(n, seed)
            self.assertEqual(CP.structure(c["optimized"])["spof_total"], 0, (n, seed))
            self.assertEqual(CP.structure(c["naive"])["spof_bridges"], n - 1, (n, seed))

    def test_intact_coverage_ties_between_designs(self):
        A, sites, d, c = layout()
        vals = {k: CP.coverage_metrics(CP.evaluate(v, 1), REC)["cov_intact"] for k, v in c.items()}
        for k, v in vals.items():
            np.testing.assert_allclose(v, vals["optimized"], err_msg=k)

    def test_coverage_is_bounded_by_the_fused_recall(self):
        A, sites, d, c = layout()
        for k, v in c.items():
            for name, arr in CP.coverage_metrics(CP.evaluate(v, 1), REC).items():
                self.assertTrue((arr <= REC.fused + 1e-12).all(), (k, name))
                self.assertTrue((arr >= -1e-12).all(), (k, name))

    def test_secondary_local_coverage_is_never_below_delivered(self):
        A, sites, d, c = layout()
        m = CP.coverage_metrics(CP.evaluate(c["optimized"], 1), REC)
        for fam in CP.FAMILIES:
            self.assertTrue((m[f"cov_local_{fam}"] >= m[f"cov_delivered_{fam}"] - 1e-12).all(), fam)

    def test_delivery_needs_a_surviving_fusion_point_but_detection_does_not(self):
        A, sites, d, c = layout()
        opt = c["optimized"]
        s = CP.shares(opt, {opt.h.cc} | set(opt.h.hubs))
        self.assertAlmostEqual(s["det_f"] + s["det_s"], 1.0)
        self.assertAlmostEqual(s["del_f"] + s["del_s"], 0.0)

    def test_the_adversary_goes_for_the_control_centre_of_a_central_design(self):
        A, sites, d, c = layout()
        naive = c["naive"]
        singles = {v: CP.shares(naive, {v}) for v in naive.comm.nodes()}
        pairs = CP.targeted_pairs(naive, singles)
        self.assertTrue(all(naive.h.cc in p for p in pairs[:5]))

    def test_scenario_sets_have_the_expected_sizes(self):
        A, sites, d, c = layout()
        ev = CP.evaluate(c["optimized"], 1)
        n = len(sites)
        self.assertEqual(len(ev.sets["single"]), n)
        self.assertEqual(len(ev.sets["n2_random"]), min(CP.N2_RANDOM_PAIRS, n * (n - 1) // 2))
        self.assertGreaterEqual(len(ev.sets["regional"]), 1)
        self.assertEqual(len(ev.sets["n2_targeted"]), CP.N2_TARGETED_KEEP)

    def test_the_naive_grid_can_island_most_of_the_network_and_the_optimized_one_never(self):
        A, sites, d, c = layout()
        self.assertGreater(CP.structure(c["naive"])["worst_node_islanded"], 0.1)
        self.assertEqual(CP.structure(c["optimized"])["worst_node_islanded"], 0.0)

    def test_bootstrap_interval_has_the_width_theory_predicts(self):
        rng = np.random.default_rng(1)
        D = rng.standard_normal(200)
        m, lo, hi = CP.paired_bootstrap(D, J=4000, seed=2)
        half = 1.96 * D.std() / math.sqrt(len(D))
        self.assertAlmostEqual(m, D.mean())
        self.assertGreater(hi - lo, 1.6 * half)
        self.assertLess(hi - lo, 2.4 * half)

    def test_independent_routes_reported_per_design(self):
        A, sites, d, c = layout()
        self.assertGreaterEqual(CP.structure(c["optimized"])["min_routes_to_cc"], 2)
        self.assertEqual(CP.structure(c["naive"])["min_routes_to_cc"], 1)

    def test_worst_case_is_no_better_than_any_scenario_family_mean(self):
        A, sites, d, c = layout()
        for name in ("optimized", "naive"):
            m = CP.coverage_metrics(CP.evaluate(c[name], 1), REC)
            for kind in ("delivered", "local"):
                for fam in CP.FAMILIES:
                    self.assertTrue((m[f"cov_{kind}_worst"] <= m[f"cov_{kind}_{fam}"] + 1e-12).all(), (name, kind, fam))

    def test_point_estimate_counts_fault_as_a_negative(self):
        y = np.array(["Normal"] * 50 + ["Fault"] * 50 + ["Replay"] * 20)
        score = np.concatenate([np.full(50, 0.1), np.full(50, 0.9), np.full(20, 0.8)])
        r = DT.point_estimate(score, y, 0.05)
        self.assertLessEqual(r["realised_fpr"], 0.05 + 1e-12)
        self.assertEqual(r["recall_Replay"], 0.0)            # a Fault that looks like an attack must not be free to flag


if __name__ == "__main__":
    unittest.main()
