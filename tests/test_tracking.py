"""
The forward (time hold-out) split and the run tracker. Properties, not snapshots.

Run:  python -m unittest discover -s tests -v      (from the project root)
"""
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import config as C  # noqa: E402
from src import forward_test as FT  # noqa: E402
from src import tracking as TR  # noqa: E402
import run_pipeline as RP  # noqa: E402
import runs as RN  # noqa: E402


def synthetic(n_blocks=20, per_block=40, seed=0):
    """A small labelled capture: every class in every block, 3 features that carry the label."""
    rng = np.random.default_rng(seed)
    mix = ["Normal"] * 22 + ["Masquerade"] * 8 + ["Injection"] * 4 + ["Replay"] * 4 + ["Fault"] * 2
    rows = []
    for b in range(n_blocks):
        for i, cls in enumerate(mix[:per_block]):
            k = C.CLASS_ORDER.index(cls)
            rows.append(dict(**{C.GROUP_KEY: b, C.TARGET: cls, "t": b * per_block + i},
                             f0=k + rng.normal(0, .5), f1=-k + rng.normal(0, .5), f2=rng.normal()))
    return pd.DataFrame(rows)


class ForwardSplit(unittest.TestCase):
    def test_test_blocks_are_strictly_later_and_disjoint(self):
        df = synthetic()
        tr, te = FT.split_forward(df)
        self.assertEqual(set(tr[C.GROUP_KEY]) & set(te[C.GROUP_KEY]), set())
        self.assertLess(tr[C.GROUP_KEY].max(), te[C.GROUP_KEY].min())

    def test_purge_block_is_dropped(self):
        df = synthetic()
        tr, te = FT.split_forward(df, n_test_blocks=5, purge_blocks=1)
        seen = set(tr[C.GROUP_KEY]) | set(te[C.GROUP_KEY])
        self.assertEqual(sorted(set(range(20)) - seen), [14])
        self.assertEqual(len(tr) + len(te), len(df) - 40)

    def test_default_split_is_blocks_0_13_and_15_19(self):
        tr, te = FT.split_forward(synthetic())
        self.assertEqual(sorted(tr[C.GROUP_KEY].unique()), list(range(14)))
        self.assertEqual(sorted(te[C.GROUP_KEY].unique()), list(range(15, 20)))

    def test_too_few_blocks_is_an_error_not_a_silent_tiny_test(self):
        with self.assertRaises(ValueError):
            FT.split_forward(synthetic(n_blocks=7))

    def test_threshold_comes_from_the_training_part_only(self):
        """Wrecking the test labels and features must not move the alert threshold."""
        df = synthetic()
        tr, te = FT.split_forward(df)
        cols = ["f0", "f1", "f2"]
        a = FT.tier_test(tr, te, cols, 0.05)
        te2 = te.copy()
        te2[cols] = te2[cols] * 50 + 100
        te2[C.TARGET] = "Normal"
        b = FT.tier_test(tr, te2, cols, 0.05)
        self.assertEqual(a["threshold"], b["threshold"])
        self.assertEqual(a["train_oof_coverage"], b["train_oof_coverage"])
        self.assertNotEqual(a["test_fpr"], b["test_fpr"])

    def test_counts_are_reported_next_to_rates(self):
        df = synthetic()
        tr, te = FT.split_forward(df)
        r = FT.tier_test(tr, te, ["f0", "f1", "f2"], 0.05)
        neg = int(te[C.TARGET].isin(["Normal", "Fault"]).sum())
        self.assertEqual(r["test_negatives"], neg)
        self.assertAlmostEqual(r["test_fpr"], r["test_false_alerts"] / neg)
        for c in C.ATTACK_CLASSES:
            self.assertEqual(r[f"n_{c}"], int((te[C.TARGET] == c).sum()))
            self.assertTrue(0.0 <= r[f"recall_{c}"] <= 1.0)

    def test_separable_signal_is_found_on_later_data(self):
        df = synthetic()
        tr, te = FT.split_forward(df)
        r = FT.tier_test(tr, te, ["f0", "f1"], 0.05)
        self.assertGreater(r["test_coverage"], 0.8)


class MissingInputs(unittest.TestCase):
    def test_step6_and_8_say_which_step_to_run_first(self):
        from src import detectors as DT
        with self.assertRaises(SystemExit) as cm:
            DT.stable_features(Path(tempfile.mkdtemp()))
        self.assertIn("step4_run_experiments.py", str(cm.exception))


class RunIdentity(unittest.TestCase):
    def test_run_id_is_sortable_and_tagged(self):
        t = datetime(2026, 10, 18, 9, 5, 3)
        self.assertEqual(TR.new_run_id(False, None, t), "20261018-090503")
        self.assertEqual(TR.new_run_id(True, "try depth 8!", t), "20261018-090503_smoke_try-depth-8")
        later = TR.new_run_id(False, None, datetime(2026, 10, 18, 9, 5, 4))
        self.assertLess(TR.new_run_id(False, None, t), later)

    def test_name_cannot_escape_the_runs_folder(self):
        rid = TR.new_run_id(False, "../../etc/passwd", datetime(2026, 1, 1))
        self.assertNotIn("/", rid)
        self.assertNotIn("..", rid)
        self.assertRegex(rid, r"^[A-Za-z0-9_-]+$")

    def test_relative_output_dir_resolves_against_project_root(self):
        os.environ["PP_TEST_DIR"] = "somewhere_else"
        try:
            self.assertEqual(C._env_path("PP_TEST_DIR", Path("x")), C.PROJECT_ROOT / "somewhere_else")
            os.environ["PP_TEST_DIR"] = str(Path(tempfile.gettempdir()).resolve())
            self.assertEqual(C._env_path("PP_TEST_DIR", Path("x")), Path(tempfile.gettempdir()).resolve())
        finally:
            del os.environ["PP_TEST_DIR"]
        self.assertEqual(C._env_path("PP_TEST_DIR", Path("x")), Path("x"))


class Harvest(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())

    def test_missing_files_give_nothing_not_an_error(self):
        for f in TR.HARVEST.values():
            self.assertEqual(f(self.d), ({}, {}))

    def test_tier_and_forward_files(self):
        pd.DataFrame(dict(tier=["fused"], coverage=[.6], coverage_lo=[.5], coverage_hi=[.7])).to_csv(self.d / "step6_tier_recall.csv", index=False)
        pd.DataFrame(dict(tier=["fused"], test_coverage=[.4], test_fpr=[.02])).to_csv(self.d / "step8_forward_tiers.csv", index=False)
        pd.DataFrame(dict(view=["all"], macro_f1=[.55])).to_csv(self.d / "step8_forward_classes.csv", index=False)
        m, _ = TR.harvest_detectors(self.d)
        self.assertEqual((m["detectors.fused.coverage"], m["detectors.fused.coverage_lo"], m["detectors.fused.coverage_hi"]), (.6, .5, .7))
        m, _ = TR.harvest_forward(self.d)
        self.assertEqual((m["forward.fused.test_coverage"], m["forward.fused.test_fpr"], m["forward.all.macro_f1"]), (.4, .02, .55))

    def test_kstar_is_the_smallest_k_within_tolerance_of_the_full_set(self):
        pd.DataFrame(dict(k=[1, 2, 4, 8, 52], macro_f1=[.3, .6, .70, .705, .71])).to_csv(self.d / "e3_feature_budget.csv", index=False)
        m, _ = TR.harvest_experiments(self.d)
        self.assertEqual(m["e3.k_star"], 4)             # .70 >= .71 - KSTAR_TOLERANCE(.01)

    def test_stable_features_are_those_chosen_in_every_fold(self):
        pd.DataFrame(dict(feature=list("abcd"), n_folds=[C.N_SPLITS, C.N_SPLITS - 1, C.N_SPLITS, 1])).to_csv(self.d / "e3_feature_stability.csv", index=False)
        m, _ = TR.harvest_experiments(self.d)
        self.assertEqual(m["e3.stable_features"], 2)

    def test_unittest_output_is_parsed(self):
        ok = "Ran 117 tests in 7.6s\n\nOK\n"
        bad = "Ran 117 tests in 7.6s\n\nFAILED (failures=1)\n"
        self.assertEqual(TR.harvest_tests(ok), ({"tests.ran": 117}, {"tests.result": "OK"}))
        self.assertEqual(TR.harvest_tests(bad)[1], {"tests.result": "FAILED (failures=1)"})
        self.assertEqual(TR.harvest_tests("garbage")[1], {"tests.result": "unknown"})

    def test_snapshot_skips_large_files(self):
        (self.d / "small.csv").write_text("a\n1\n")
        (self.d / "big.csv").write_bytes(b"x" * (TR.SNAPSHOT_MAX_BYTES + 1))
        run = Path(tempfile.mkdtemp())
        self.assertEqual(TR.snapshot_outputs(self.d, run), 1)
        self.assertTrue((run / "outputs" / "small.csv").exists())
        self.assertFalse((run / "outputs" / "big.csv").exists())


def fake_run(run_id, f1=0.7, smoke=False, steps=("experiments",), verdict="OPTIMIZED BETTER", cfg=None):
    return dict(run_id=run_id, name=None, note=None, smoke=smoke, reportable=not smoke, started="2026-10-18T10:00:00",
                status="ok", seconds=60.0, command="x", git=dict(commit="abc1234", dirty=False),
                environment=dict(python="3.13", packages={}),
                config=dict(dataset="d", random_seed=42, n_splits=5, n_blocks=20, model_params={}, kstar_tolerance=.01,
                            operating_fpr=.01, bootstrap_B=300, ensemble_sizes=[20], ensemble_seeds=30,
                            grid_assumptions={}, grid_assumptions_hash="h", env_overrides={}, **(cfg or {})),
                steps=[dict(key="experiments", status="ok", seconds=1.0, log="x", metrics={"e1.all.macro_f1": f1, "e1.all.macro_f1_sd": .03},
                            labels={}),
                       dict(key="compare", status="ok", seconds=1.0, log="y", metrics={}, labels={"compare.vs_naive": verdict})])


class Records(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())

    def put(self, rec):
        (self.d / rec["run_id"]).mkdir()
        TR.write_json(self.d / rec["run_id"] / "run.json", rec)

    def test_index_lists_runs_in_order_and_survives_a_corrupt_record(self):
        self.put(fake_run("20261018-100000", .70))
        self.put(fake_run("20261018-110000", .72, verdict="NOT ESTABLISHED"))
        (self.d / "20261018-120000").mkdir()
        (self.d / "20261018-120000" / "run.json").write_text("{ half written")
        df = TR.rebuild_index(self.d)
        self.assertEqual(list(df["run_id"]), ["20261018-100000", "20261018-110000"])
        self.assertEqual(list(df["macro_f1_blocked_all"]), [.70, .72])
        self.assertEqual(df["verdict_vs_naive"].iloc[1], "NOT ESTABLISHED")
        self.assertTrue((self.d / "index.csv").exists())

    def test_diff_reports_changed_settings_and_numbers_only(self):
        a, b = fake_run("a", .70), fake_run("b", .72, verdict="NOT ESTABLISHED", cfg=dict(extra_setting=1))
        b["config"]["n_splits"] = 3
        cfg, df = TR.diff_runs(a, b)
        self.assertEqual(cfg["n_splits"], (5, 3))
        self.assertIn("extra_setting", cfg)
        self.assertNotIn("random_seed", cfg)
        row = df[df["metric"] == "e1.all.macro_f1"].iloc[0]
        self.assertAlmostEqual(row["delta"], .02)
        same = df[df["metric"] == "e1.all.macro_f1_sd"].iloc[0]
        self.assertEqual(same["delta"], 0)
        v = df[df["metric"] == "compare.vs_naive"].iloc[0]
        self.assertEqual((v["a"], v["b"]), ("OPTIMIZED BETTER", "NOT ESTABLISHED"))

    def test_summary_leads_with_a_headline_and_flags_smoke(self):
        s = TR.summary_markdown(fake_run("x", .712, smoke=True))
        self.assertIn("never quote these numbers", s)
        self.assertLess(s.index("## Headline"), s.index("## Every recorded number"))
        self.assertIn("0.712 +/- 0.030", s)
        self.assertIn("OPTIMIZED BETTER", s)
        self.assertNotIn("DETECTOR TIERS", s)           # that step was not run, so no empty section

    def test_summary_without_results_does_not_crash(self):
        r = fake_run("x")
        r["steps"] = []
        s = TR.summary_markdown(r)
        self.assertNotIn("## Headline", s)

    def test_list_is_one_narrow_row_per_run_and_says_which_are_reportable(self):
        a = fake_run("20261018-100000", .71)
        b = fake_run("20261018-110000_smoke", .75, smoke=True, verdict="NOT ESTABLISHED")
        c = fake_run("20261018-120000", .70)
        c["status"] = "failed"
        d = fake_run("20261018-130000", .70)
        d["reportable"] = False                          # --only, or a changed setting
        t = RN.compact_table([a, b, c, d])
        self.assertEqual(len(t), 4)
        self.assertEqual(list(t["mode"]), ["full", "smoke", "failed", "partial"])
        self.assertEqual(t["vs naive"].iloc[1], "not est.")
        self.assertLess(len(t.to_string(index=False)) // 4, 140)

    def test_run_reference_resolution(self):
        runs = [fake_run("20261018-100000"), fake_run("20261018-110000_smoke_try"), fake_run("20261019-090000")]
        self.assertEqual(RN.resolve(runs, "last")["run_id"], "20261019-090000")
        self.assertEqual(RN.resolve(runs, "prev")["run_id"], "20261018-110000_smoke_try")
        self.assertEqual(RN.resolve(runs, "20261018-1000")["run_id"], "20261018-100000")
        self.assertEqual(RN.resolve(runs, "try")["run_id"], "20261018-110000_smoke_try")
        with self.assertRaises(SystemExit):
            RN.resolve(runs, "20261018")                # matches two runs: refuse to guess
        with self.assertRaises(SystemExit):
            RN.resolve([], "last")


try:
    import mlflow  # noqa: F401
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


class MlflowMirror(unittest.TestCase):
    def test_missing_mlflow_is_a_message_not_a_crash(self):
        saved = sys.modules.get("mlflow", "absent")
        sys.modules["mlflow"] = None                    # makes `import mlflow` raise ImportError
        try:
            msg = TR.mirror_mlflow(fake_run("x"), tempfile.mkdtemp(), root=tempfile.mkdtemp())
        finally:
            if saved == "absent":
                del sys.modules["mlflow"]
            else:
                sys.modules["mlflow"] = saved
        self.assertIn("not installed", msg)

    def test_uri_is_sqlite_inside_the_project(self):
        self.assertTrue(TR.mlflow_uri(Path("/some/dir")).startswith("sqlite:///"))
        self.assertTrue(TR.mlflow_uri(Path("/some/dir")).endswith("mlflow.db"))

    @unittest.skipUnless(HAS_MLFLOW, "mlflow not installed (optional)")
    def test_a_run_round_trips(self):
        root, rd = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
        (rd / "summary.md").write_text("s")
        (rd / "run.json").write_text("{}")
        rec = fake_run("20261018-100000_t", .7123)
        msg = TR.mirror_mlflow(rec, rd, root=root)
        self.assertTrue(msg.startswith("logged"), msg)
        import mlflow
        mlflow.set_tracking_uri(TR.mlflow_uri(root))
        df = mlflow.search_runs([mlflow.get_experiment_by_name(TR.MLFLOW_EXPERIMENT).experiment_id])
        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(df["metrics.e1.all.macro_f1"].iloc[0], .7123)
        self.assertEqual(df["tags.compare.vs_naive"].iloc[0], "OPTIMIZED BETTER")


class Watch(unittest.TestCase):
    def status(self, **kw):
        return dict(dict(run_id="r1", state="running", n_steps=9, step="experiments", step_index=4,
                         step_started=100.0, last_output=130.0, last_line="E1 view ablation"), **kw)

    def test_elapsed_is_computed_from_the_clock_not_from_the_last_line(self):
        a = RN.format_status(self.status(), 140.0)
        b = RN.format_status(self.status(), 160.0)
        self.assertIn("step 4/9 experiments  40s", a)
        self.assertIn("60s", b)
        self.assertIn("E1 view ablation", a)

    def test_a_silent_step_is_called_out(self):
        self.assertNotIn("quiet", RN.format_status(self.status(), 135.0))
        self.assertIn("quiet for 30s", RN.format_status(self.status(), 160.0))

    def test_finished_runs_show_their_final_state(self):
        self.assertIn("FAILED", RN.format_status(self.status(state="failed"), 999.0))
        self.assertIn("OK", RN.format_status(self.status(state="ok"), 999.0))


class Runner(unittest.TestCase):
    def test_step_selection_by_number_and_name(self):
        self.assertEqual([s[0] for s in RP.select(["4", "forward"], None)], ["experiments", "forward"])
        self.assertEqual(len(RP.select(None, None)), 9)
        self.assertNotIn("tests", [s[0] for s in RP.select(None, ["tests"])])
        with self.assertRaises(SystemExit):
            RP.select(["12"], None)

    def test_steps_run_in_pipeline_order_whatever_order_they_are_asked_in(self):
        self.assertEqual([s[0] for s in RP.select(["8", "4", "6"], None)], ["experiments", "detectors", "forward"])

    def test_only_a_whole_default_run_is_reportable(self):
        self.assertTrue(RP.is_reportable(False, None, {"PATH": "x"}))
        self.assertFalse(RP.is_reportable(True, None, {}))                       # smoke
        self.assertFalse(RP.is_reportable(False, ["4"], {}))                     # partial
        self.assertFalse(RP.is_reportable(False, None, {"PP_N_ESTIMATORS": "600"}))   # changed setting

    def test_every_step_script_exists(self):
        for key, _, script, _ in RP.STEPS:
            if script:
                self.assertTrue((Path(ROOT) / script).exists(), script)

    def test_smoke_never_writes_to_the_reportable_folders(self):
        self.assertNotEqual(RP.SMOKE_ENV["PP_OUTPUT_DIR"], "outputs")
        self.assertNotEqual(RP.SMOKE_ENV["PP_MODELS_DIR"], "models")
        cmd = RP.step_command(RP.STEPS[6], smoke=True, extras=True)           # compare
        self.assertIn("--no-cases", cmd)                                       # would overwrite dataset/grids otherwise
        self.assertNotIn("--sensitivity", cmd)                                 # extras are ignored in smoke

    def test_design_and_compare_demand_measured_inputs(self):
        for i in (4, 6):
            self.assertIn("--require-measured", RP.step_command(RP.STEPS[i], False, False))

    def test_tests_step_is_unittest_discover(self):
        cmd = RP.step_command(RP.STEPS[8], True, False)
        self.assertEqual(cmd[1:3], ["-u", "-m"])
        self.assertIn("unittest", cmd)


if __name__ == "__main__":
    unittest.main()
