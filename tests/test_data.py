"""
Contract of the data layer: loader asserts, layer views, blocked splits, the
Train/Test independence audit, measured-input provenance. Synthetic captures only -
no dataset needed.

Run:  python -m unittest discover -s tests -v      (from the project root)
"""
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config as C  # noqa: E402
from src import data as DT  # noqa: E402
from src import metrics as MX  # noqa: E402
from src import modalities as MD  # noqa: E402
from src import splits as SP  # noqa: E402
from src.design import params as P  # noqa: E402
from src.models import EncodedModel  # noqa: E402

GOOSE = ["sqNum", "stnum", "state_cb", "sqDiff", "stDiff", "timeLastMsg", "recentChange"]
CONSISTENCY = ["MU1Cs", "MU2Cs", "MU3Cs", "MU4Cs", "Cs", "consistency", "threePhaseVSum", "threePhaseCSum"]
RELAY = ["MU1Log", "MU2Log", "MU3Log", "MU4Log", "any_relay"]
PROCESS = ([f"threePhaseVMU{k}" for k in (1, 2, 3, 4)] + ["threePhaseCurrentMU1", "threePhaseCMU2",
           "threePhaseCMU3", "threePhaseCMU4", "IED4_iA", "IED4_iB", "IED4_iC"]
           + [f"MU{k}{w}Angle{p}" for k in (1, 2, 3, 4) for w in ("Voltage", "Current") for p in "ABC"])
FEATURES = GOOSE + CONSISTENCY + PROCESS[:11] + RELAY            # schema of the real file (subset of phasors)
CLASSES = C.CLASS_ORDER


def make_raw(n=500, seed=0):
    """A Train.csv-shaped frame whose derived timing columns match its own stream."""
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.integers(100, 1200, n)) + 1000
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    df["sqNum"] = rng.integers(1, 40, n)
    df["stnum"] = rng.integers(1, 90, n)
    df["time"] = t
    df["timeLastMsg"] = pd.Series(t).diff().fillna(t[0]).to_numpy()
    df["sqDiff"] = df["sqNum"].diff().fillna(df["sqNum"].iloc[0])
    df["stDiff"] = df["stnum"].diff().fillna(df["stnum"].iloc[0])
    df["class"] = np.resize(np.array(CLASSES), n)
    df["class"] = rng.permutation(df["class"].to_numpy())
    return df


class Base(unittest.TestCase):
    KEYS = ("TRAIN_CSV", "TEST_CSV", "OUTPUT_DIR")

    def setUp(self):
        self._saved = {k: getattr(C, k) for k in self.KEYS}
        self.tmp = Path(tempfile.mkdtemp())
        C.TRAIN_CSV, C.TEST_CSV = self.tmp / "Train.csv", self.tmp / "Test.csv"
        C.OUTPUT_DIR = self.tmp / "outputs"
        C.OUTPUT_DIR.mkdir()

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(C, k, v)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, train, test=None):
        train.to_csv(C.TRAIN_CSV, index=False)
        (test if test is not None else train).to_csv(C.TEST_CSV, index=False)


class Views(unittest.TestCase):
    def test_every_real_column_is_in_exactly_one_layer(self):
        cols = GOOSE + CONSISTENCY + PROCESS + RELAY
        layers = MD.modality_map(cols)
        self.assertNotIn("unknown", layers.values())
        views = MD.build_views(cols)
        self.assertEqual(sorted(sum((views[m] for m in MD.SINGLE_MODALITIES), [])), sorted(cols))
        self.assertEqual(views["station"], views["goose"] + views["consistency"] + views["relay"])
        self.assertEqual(views["all"], cols)

    def test_phasors_never_leak_into_the_station_view(self):
        views = MD.build_views(GOOSE + CONSISTENCY + PROCESS + RELAY)
        self.assertFalse(set(views["station"]) & set(PROCESS))
        self.assertEqual(len(views["process"]), len(PROCESS))

    def test_merging_units_are_counted_from_the_names(self):
        self.assertEqual(MD.n_units(PROCESS), 4)
        self.assertEqual(MD.n_units(["sqNum", "stnum"]), 0)

    def test_unknown_columns_are_reported_not_guessed(self):
        self.assertEqual(MD.classify_column("temperature"), "unknown")


class Loader(Base):
    def test_time_and_label_are_never_features(self):
        self.write(make_raw())
        df, feats, views = DT.load_dataset()
        for banned in ("time", "class", "t_ms", C.TARGET, C.FAMILY_COL, C.GROUP_KEY):
            self.assertNotIn(banned, feats)
        self.assertEqual(len(feats), len(FEATURES))
        self.assertTrue(df["t_ms"].is_monotonic_increasing)

    def test_family_mapping(self):
        self.write(make_raw())
        df, _, _ = DT.load_dataset()
        fam = df.groupby(C.TARGET)[C.FAMILY_COL].first().to_dict()
        self.assertEqual(fam, C.FAMILY)

    def test_blocks_are_contiguous_time_slices(self):
        self.write(make_raw(800))
        df, _, _ = DT.load_dataset()
        self.assertTrue(df[C.GROUP_KEY].is_monotonic_increasing)
        self.assertEqual(df[C.GROUP_KEY].nunique(), C.N_BLOCKS)
        self.assertEqual(df[C.GROUP_KEY].max(), C.N_BLOCKS - 1)

    def test_changed_schema_is_refused(self):
        tr = make_raw()
        self.write(tr, tr.drop(columns=["Cs"]))
        with self.assertRaisesRegex(AssertionError, "T1"):
            DT.load_published()

    def test_unknown_class_is_refused(self):
        tr = make_raw(); te = tr.copy(); te.loc[0, "class"] = "Jamming"
        self.write(tr, te)
        with self.assertRaisesRegex(AssertionError, "T2"):
            DT.load_published()

    def test_missing_values_and_text_features_are_refused(self):
        tr = make_raw(); bad = tr.copy(); bad.loc[3, "Cs"] = np.nan
        self.write(tr, bad)
        with self.assertRaisesRegex(AssertionError, "T3"):
            DT.load_published()
        txt = tr.copy(); txt["Cs"] = "x"
        self.write(tr, txt)
        with self.assertRaisesRegex(AssertionError, "T3"):
            DT.load_published()

    def test_missing_file_says_where_to_put_it(self):
        with self.assertRaisesRegex(FileNotFoundError, "Train.csv"):
            DT.load_published()

    def test_rates_are_measured_from_the_clock(self):
        df = make_raw(400)
        df["time"] = 1000 + np.arange(400) * 500                # exactly 2 messages/s
        self.write(df)
        d, _, _ = DT.load_dataset()
        r = DT.rates(d)
        self.assertAlmostEqual(r["row_rate_hz"], 400 / 199.5, places=3)
        self.assertEqual(r["median_gap_ms"], 500)
        share = d[C.FAMILY_COL].isin(["Normal", "Natural"]).mean()
        self.assertAlmostEqual(r["negative_rate_hz"], share * r["row_rate_hz"], places=6)


class Splits(Base):
    def frame(self):
        self.write(make_raw(1000, seed=3))
        return DT.load_dataset()[0]

    def test_blocked_folds_never_split_a_block(self):
        df = self.frame()
        for tr, te in SP.blocked(df):
            self.assertFalse(set(df.iloc[tr][C.GROUP_KEY]) & set(df.iloc[te][C.GROUP_KEY]))

    def test_blocked_folds_partition_the_rows_and_keep_every_class(self):
        df = self.frame()
        seen = []
        for tr, te in SP.blocked(df):
            seen += list(te)
            self.assertEqual(set(df.iloc[te][C.TARGET]), set(CLASSES))
            self.assertEqual(set(df.iloc[tr][C.TARGET]), set(CLASSES))
        self.assertEqual(sorted(seen), list(range(len(df))))

    def test_the_shuffled_split_does_straddle_blocks(self):
        df = self.frame()
        tr, te = next(SP.naive_rows(df))
        self.assertTrue(set(df.iloc[tr][C.GROUP_KEY]) & set(df.iloc[te][C.GROUP_KEY]))

    def test_describe(self):
        d = SP.describe(self.frame())
        self.assertEqual(d["n_blocks"], C.N_BLOCKS)
        self.assertEqual(d["n_rows"], 1000)


class Audit(Base):
    def test_a_resampled_copy_is_detected_and_only_timing_differs(self):
        tr = make_raw(600)
        te = tr.copy()
        rng = np.random.default_rng(9)
        for c in ("timeLastMsg", "sqDiff", "stDiff"):
            te[c] = rng.integers(0, 900, len(te))
        te = te.sample(frac=1, random_state=1)               # shipped in a different order
        self.write(tr, te)
        a = DT.audit()
        self.assertEqual(a["test_messages_present_in_train"], 1.0)
        self.assertEqual(sorted(a["columns_that_differ"]), ["sqDiff", "stDiff", "timeLastMsg"])
        self.assertTrue(a["class_counts_equal"])
        self.assertGreater(a["timing_reproducible_from_rows"]["train"]["timeLastMsg"], 0.99)
        self.assertLess(a["timing_reproducible_from_rows"]["test"]["timeLastMsg"], 0.5)

    def test_shared_timestamps_do_not_fake_differences(self):
        """The real files share timestamps between a Normal and a Masquerade message; an
        identical copy must still audit as identical (merge on time AND class)."""
        tr = make_raw(300)
        dup = tr.iloc[:20].copy()
        dup["time"] = tr["time"].iloc[:20].to_numpy()
        dup["class"] = np.where(tr["class"].iloc[:20] == "Normal", "Masquerade", "Normal")
        dup[FEATURES] = dup[FEATURES] + 1.0                       # same instant, different message
        tr = pd.concat([tr, dup], ignore_index=True)
        self.write(tr, tr.sample(frac=1, random_state=4))
        a = DT.audit()
        self.assertEqual(a["columns_that_differ"], {})
        self.assertEqual(a["test_messages_present_in_train"], 1.0)

    def test_independent_files_are_reported_as_independent(self):
        a, b = make_raw(500, seed=1), make_raw(500, seed=2)
        self.write(a, b)
        r = DT.audit()
        self.assertLess(r["test_messages_present_in_train"], 0.05)
        self.assertEqual(r["exact_duplicate_test_rows"], 0.0)


class Models(unittest.TestCase):
    def test_probability_columns_follow_the_class_order_even_when_a_class_is_missing(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(120, 3)), columns=list("abc"))
        y = np.resize(np.array(["Fault", "Replay", "Normal"]), 120)       # 2 classes never seen
        m = EncodedModel(n_estimators=10).fit(X, y)
        p = m.predict_proba(X)
        self.assertEqual(p.shape, (120, len(CLASSES)))
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)
        for absent in ("Masquerade", "Injection"):
            self.assertEqual(p[:, CLASSES.index(absent)].sum(), 0.0)
        pred = m.predict(X)
        self.assertEqual(set(pred), {"Fault", "Replay", "Normal"})

    def test_binary_model_reports_other_then_attack(self):
        rng = np.random.default_rng(1)
        X = pd.DataFrame({"a": np.r_[rng.normal(-3, 1, 80), rng.normal(3, 1, 80)]})
        y = np.array(["Other"] * 80 + ["Attack"] * 80)
        m = EncodedModel(["Other", "Attack"], n_estimators=20).fit(X, y)
        p = m.predict_proba(X)
        self.assertGreater(p[100:, 1].mean(), 0.9)
        self.assertGreater(p[:60, 0].mean(), 0.9)


class Metrics(unittest.TestCase):
    def test_attack_coverage_is_the_mean_over_attack_classes_only(self):
        res = {"recall__Normal": 1.0, "recall__Fault": 1.0, "recall__Masquerade": 0.9,
               "recall__Injection": 0.3, "recall__Replay": 0.0}
        self.assertAlmostEqual(MX.attack_coverage(res), 0.4)

    def test_false_alerts_scale_with_the_measured_rate(self):
        self.assertAlmostEqual(MX.false_alerts_per_day(0.01, 2.0), 0.01 * 2.0 * 86400)
        self.assertAlmostEqual(MX.upstream_bytes_per_second(10, 2.0, n_bays=3), 10 * 4 * 2.0 * 3)


class Provenance(Base):
    def outputs(self, e1=(0.6, 0.55, 0.2)):
        out = C.OUTPUT_DIR
        (out / "step1_profile.json").write_text(json.dumps({"n_units": 4, "rates": {"row_rate_hz": 2.5}}))
        (out / "e3_feature_stability.csv").write_text(",0\nstate_cb,5\nthreePhaseVMU1,5\nsqNum,2\n")
        (out / "e3_feature_budget.csv").write_text("k,macro_f1\n8,0.5\n52,0.6\n")
        (out / "e4_compute_profile.csv").write_text(
            "config,n_features,p50_ms,p95_ms,batch128_per_sample_ms\n"
            "tier2_zone_fused,52,4,5,0.05\ntier3_widearea_fused,52,4,6,0.05\n"
            "tier1_edge_kstar,2,1,2,0.02\ntier1_edge_process_only,32,3,4,0.04\n")
        (out / "e1_views_blocked.csv").write_text(
            "view,attack_coverage\nall,%s\nstation,%s\nprocess,%s\n" % e1)

    def set_mtimes(self, data_age):
        t = time.time()
        for p in (C.TRAIN_CSV, C.TEST_CSV):
            p.write_text("x")
            os.utime(p, (t - data_age, t - data_age))

    def test_fresh_outputs_are_read_and_reported_as_measured(self):
        self.set_mtimes(data_age=3600)
        self.outputs()
        M = P.load_measured()
        self.assertEqual((M.ieds_per_unit, M.sample_rate_hz, M.k_stable, M.k_stable_process),
                         (4, 2.5, 2, 1))
        self.assertEqual((M.n_features_full, M.n_features_process, M.edge_p95_ms), (52, 32, 2.0))
        self.assertEqual((M.det_all, M.det_process), (0.6, 0.2))
        self.assertEqual(P.provenance_problems(M), [])

    def test_outputs_older_than_the_dataset_are_stale(self):
        self.outputs()
        old = time.time() - 7200
        for f in C.OUTPUT_DIR.iterdir():
            os.utime(f, (old, old))
        self.set_mtimes(data_age=60)
        M = P.load_measured()
        names = [n for n, _ in P.provenance_problems(M)]
        for n in ("k_stable", "edge_p95_ms", "det_all", "sample_rate_hz"):
            self.assertIn(n, names)
        self.assertIn("STALE", M.provenance["k_stable"])

    def test_missing_outputs_are_placeholders(self):
        self.set_mtimes(data_age=60)
        M = P.load_measured()
        self.assertTrue(M.provenance["det_all"].startswith("FALLBACK"))
        self.assertIn("det_all", [n for n, _ in P.provenance_problems(M)])

    def test_sensor_removal_can_never_raise_detection(self):
        self.set_mtimes(data_age=3600)
        self.outputs(e1=(0.55, 0.61, 0.2))               # station "beat" all by fold noise
        M = P.load_measured()
        self.assertEqual(M.det_station, M.det_all)
        self.assertIn("capped", M.provenance["det_station"])

    def test_unit_override(self):
        self.set_mtimes(data_age=3600)
        self.outputs()
        self.assertEqual(P.load_measured(ieds_per_unit=1).ieds_per_unit, 1)

    def test_fingerprint_changes_with_the_data(self):
        self.write(make_raw(100, 1))
        a = P.dataset_fingerprint()
        self.write(make_raw(100, 2))
        self.assertNotEqual(a, P.dataset_fingerprint())
        self.assertIsNone(None if (self.tmp / "Train.csv").exists() else 1)


if __name__ == "__main__":
    unittest.main()
