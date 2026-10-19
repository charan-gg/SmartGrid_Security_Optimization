"""
Run tracking: what a run did, how long each step took, what numbers it produced, and what
settings and environment produced them - recorded in runs/<run_id>/ so two runs can be compared.

No dependency beyond pandas. `mirror_mlflow` copies a finished run into a local MLflow store if
MLflow happens to be installed (optional; the file records are the source of truth).

A run directory holds
    run.json       the full record (config, environment, per-step status/duration/metrics)
    summary.md     the same, readable (preview it in VS Code with Ctrl+Shift+V)
    status.json    live: current step and elapsed time while the run is in progress
    NN_<step>.log  everything a step printed
    outputs/       a snapshot of the small result files the run produced
"""
from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

import pandas as pd

from . import config as C

RUNS_DIR = C.PROJECT_ROOT / "runs"
SNAPSHOT_MAX_BYTES = 400_000
PACKAGES = ("numpy", "pandas", "scikit-learn", "xgboost", "scipy", "matplotlib", "joblib")


# ------------------------------------------------------------------ run identity
def new_run_id(smoke=False, name=None, now=None):
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-")[:30] if name else ""
    return "_".join(x for x in (stamp, "smoke" if smoke else "", slug) if x)


def git_state():
    def run(*a):
        return subprocess.run(["git", *a], cwd=C.PROJECT_ROOT, capture_output=True, text=True, timeout=10)
    try:
        h = run("rev-parse", "--short", "HEAD")
        if h.returncode != 0:
            return dict(commit=None, dirty=None)
        dirty = bool(run("status", "--porcelain", "--", "src", "tests", "*.py").stdout.strip())
        return dict(commit=h.stdout.strip(), dirty=dirty)
    except (OSError, subprocess.SubprocessError):
        return dict(commit=None, dirty=None)


def environment():
    pk = {}
    for p in PACKAGES:
        try:
            pk[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pk[p] = None
    return dict(python=platform.python_version(), platform=platform.platform(), packages=pk)


def config_snapshot():
    """The settings that change a number. A run is only comparable to another with the same ones."""
    import os
    from .design.params import dataset_fingerprint
    ga = json.dumps(C.GRID_ASSUMPTIONS, sort_keys=True, default=str)
    return dict(
        dataset=dataset_fingerprint(), random_seed=C.RANDOM_SEED, n_splits=C.N_SPLITS, n_blocks=C.N_BLOCKS,
        model_params=dict(C.MODEL_PARAMS), kstar_tolerance=C.KSTAR_TOLERANCE, operating_fpr=C.OPERATING_FPR,
        bootstrap_B=C.BOOTSTRAP_B, ensemble_sizes=list(C.ENSEMBLE_SIZES), ensemble_seeds=C.ENSEMBLE_SEEDS,
        grid_assumptions=dict(C.GRID_ASSUMPTIONS),
        grid_assumptions_hash=hashlib.sha256(ga.encode()).hexdigest()[:10],
        env_overrides={k: v for k, v in os.environ.items() if k.startswith("PP_")})


# ------------------------------------------------------------------ harvesting results
def _read_csv(out, name):
    try:
        return pd.read_csv(Path(out) / name)
    except (OSError, ValueError):
        return None


def _read_json(out, name):
    try:
        return json.loads((Path(out) / name).read_text())
    except (OSError, ValueError):
        return None


def harvest_profile(out):
    j = _read_json(out, "step1_profile.json")
    if not j:
        return {}, {}
    return {"data.rows": j["n_rows_train"], "data.row_rate_hz": j["rates"]["row_rate_hz"],
            "data.merging_units": j["n_units"]}, {}


def harvest_baselines(out):
    j = _read_json(out, "step3_baselines.json")
    if not j:
        return {}, {}
    m = {}
    for view, t in j["table"].items():
        m[f"split.{view}.blocked_cv_macro_f1"] = t["blocked_time_cv"]
        m[f"split.{view}.shuffled_cv_macro_f1"] = t["shuffled_rows_cv"]
        m[f"split.{view}.published_train_to_test_macro_f1"] = t["published_train_to_test"]
    return m, {}


def harvest_experiments(out):
    m = {}
    e1 = _read_csv(out, "e1_views_blocked.csv")
    if e1 is not None:
        for _, r in e1.iterrows():
            m[f"e1.{r['view']}.macro_f1"] = r["macro_f1"]
            m[f"e1.{r['view']}.macro_f1_sd"] = r["macro_f1__sd"]
            if "attack_coverage" in e1:
                m[f"e1.{r['view']}.attack_coverage"] = r["attack_coverage"]
        row = e1[e1["view"] == "all"]
        if len(row):
            for k in C.CLASS_ORDER:
                m[f"e1.all.recall_{k}"] = float(row.iloc[0][f"recall__{k}"])
    e3 = _read_csv(out, "e3_feature_budget.csv")
    if e3 is not None:
        e3 = e3.set_index("k")
        m["e3.k_star"] = int(e3[e3["macro_f1"] >= e3["macro_f1"].iloc[-1] - C.KSTAR_TOLERANCE].index.min())
    st = _read_csv(out, "e3_feature_stability.csv")
    if st is not None:
        m["e3.stable_features"] = int((st.iloc[:, 1] == C.N_SPLITS).sum())
    e6 = _read_csv(out, "e6_operating_points.csv")
    if e6 is not None:
        r = e6[e6["budget"] == 2000.0]
        if len(r):
            m["e6.recall_at_2000_false_alerts_per_day"] = float(r.iloc[0]["recall_attack"])
    e8 = _read_csv(out, "e8_unseen_attack.csv")
    if e8 is not None and "gap" in e8:
        m["e8.mean_gap_all"] = float(e8[e8["view"] == "all"]["gap"].mean())
    e4 = _read_csv(out, "e4_compute_profile.csv")
    if e4 is not None:
        for _, r in e4.iterrows():
            m[f"e4.{r['config']}.p95_ms"] = float(r["p95_ms"])
    return m, {}


def harvest_design(out):
    j = _read_json(out, "step5_design_summary.json")
    if not j:
        return {}, {}
    s = j["summary"]
    return {"design.lines": s["power"]["lines"], "design.n1_premium_pct": s["power"]["n1_premium_over_radial_pct"],
            "design.pmus": s["pmu"]["n_pmus"], "design.zones": s["hierarchy"]["k_zones"],
            "design.levels": s["hierarchy"]["n_levels"], "design.total_cost_kusd": s["costs"]["total"]}, \
        {"design.feasible": str(s["hierarchy"]["feasible"])}


def harvest_detectors(out):
    t = _read_csv(out, "step6_tier_recall.csv")
    if t is None:
        return {}, {}
    m = {}
    for _, r in t.iterrows():
        m[f"detectors.{r['tier']}.coverage"] = r["coverage"]
        m[f"detectors.{r['tier']}.coverage_lo"] = r["coverage_lo"]
        m[f"detectors.{r['tier']}.coverage_hi"] = r["coverage_hi"]
    return m, {}


def harvest_compare(out):
    j = _read_json(out, "step7_summary.json")
    v = _read_csv(out, "step7_verdict.csv")
    if not j:
        return {}, {}
    labels = {f"compare.vs_{k}": j[k]["overall"] for k in ("naive", "budget-matched", "bm-lines+tiered") if k in j}
    m = {}
    if v is not None:
        for d in ("naive", "budget-matched"):
            x = v[v["default"] == d].set_index("metric")
            for key in ("cov_delivered_single", "cov_delivered_n2_targeted", "cov_delivered_worst", "spof_total",
                        "cost_total_kusd"):
                if key in x.index:
                    m[f"compare.vs_{d}.{key}.optimized"] = float(x.loc[key, "optimized"])
                    m[f"compare.vs_{d}.{key}.default"] = float(x.loc[key, "baseline"])
    return m, labels


def harvest_forward(out):
    t, c = _read_csv(out, "step8_forward_tiers.csv"), _read_csv(out, "step8_forward_classes.csv")
    m = {}
    if t is not None:
        for _, r in t.iterrows():
            m[f"forward.{r['tier']}.test_coverage"] = r["test_coverage"]
            m[f"forward.{r['tier']}.test_fpr"] = r["test_fpr"]
    if c is not None:
        for _, r in c.iterrows():
            m[f"forward.{r['view']}.macro_f1"] = r["macro_f1"]
    return m, {}


def harvest_tests(text):
    ran = re.search(r"Ran (\d+) tests? in ([\d.]+)s", text)
    ok = bool(re.search(r"^OK\b", text, re.M))
    bad = re.search(r"FAILED \((.*?)\)", text)
    m, lab = {}, {}
    if ran:
        m["tests.ran"] = int(ran.group(1))
    lab["tests.result"] = "OK" if ok else (f"FAILED ({bad.group(1)})" if bad else "unknown")
    return m, lab


HARVEST = dict(profile=harvest_profile, baselines=harvest_baselines, experiments=harvest_experiments,
               design=harvest_design, detectors=harvest_detectors, compare=harvest_compare,
               forward=harvest_forward)


# ------------------------------------------------------------------ writing a run
def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def snapshot_outputs(out, run_dir):
    dest = Path(run_dir) / "outputs"
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(Path(out).glob("*")):
        if p.is_file() and p.stat().st_size <= SNAPSHOT_MAX_BYTES:
            (dest / p.name).write_bytes(p.read_bytes())
            n += 1
    return n


KEY_COLUMNS = [("macro_f1_blocked_all", "e1.all.macro_f1"), ("macro_f1_blocked_station", "e1.station.macro_f1"),
               ("k_star", "e3.k_star"), ("coverage_fused", "detectors.fused.coverage"),
               ("forward_macro_f1_all", "forward.all.macro_f1")]
KEY_LABELS = [("verdict_vs_naive", "compare.vs_naive"), ("verdict_vs_budget_matched", "compare.vs_budget-matched"),
              ("tests", "tests.result")]


def flat(record):
    m, lab = {}, {}
    for s in record["steps"]:
        m.update(s.get("metrics", {}))
        lab.update(s.get("labels", {}))
    return m, lab


def load_runs(runs_dir=None):
    out = []
    for p in sorted(Path(runs_dir or RUNS_DIR).glob("*/run.json")):
        try:
            out.append(json.loads(p.read_text()))
        except ValueError:
            continue
    return out


def index_rows(runs):
    rows = []
    for r in runs:
        m, lab = flat(r)
        row = dict(run_id=r["run_id"], name=r.get("name") or "", started=r["started"], minutes=round(r["seconds"] / 60, 1),
                   status=r["status"], mode="smoke" if r["smoke"] else "full", reportable=r["reportable"],
                   git=(r["git"]["commit"] or "") + ("*" if r["git"].get("dirty") else ""))
        for col, key in KEY_COLUMNS:
            row[col] = m.get(key)
        for col, key in KEY_LABELS:
            row[col] = lab.get(key)
        rows.append(row)
    return pd.DataFrame(rows)


def rebuild_index(runs_dir=None):
    rd = Path(runs_dir or RUNS_DIR)
    df = index_rows(load_runs(rd))
    rd.mkdir(parents=True, exist_ok=True)
    df.to_csv(rd / "index.csv", index=False)
    return df


def _num(v):
    if isinstance(v, float):
        return f"{v:,.0f}" if abs(v) >= 1e4 else f"{v:.4g}"
    return str(v)


_H = object()      # marks a section header row in headline_rows


def headline_rows(m, lab):
    """(what, value) pairs worth reading first; a row is left out when its step was not run."""
    def g(k, spec=".3f"):
        return format(m[k], spec) if k in m else None

    def pm(k):
        return f"{m[k]:.3f} +/- {m[k + '_sd']:.3f}" if k in m and k + "_sd" in m else g(k)

    def ci(t):
        k = f"detectors.{t}.coverage"
        return f"{m[k]:.3f}  (95% CI {m[k + '_lo']:.3f} to {m[k + '_hi']:.3f})" if k in m else None
    rows = [
        ("TRAIN + TEST, blocked CV", _H),
        ("macro-F1, all 52 features", pm("e1.all.macro_f1")),
        ("macro-F1, station layers (20)", pm("e1.station.macro_f1")),
        ("attack coverage, all features", g("e1.all.attack_coverage")),
        ("same data, shuffled CV (leaks neighbours)", g("split.all.shuffled_cv_macro_f1")),
        ("same data, published Train->Test (same recording)", g("split.all.published_train_to_test_macro_f1")),
        ("fold-stable features / smallest sufficient k", f"{m['e3.stable_features']:.0f} / {m['e3.k_star']:.0f}" if "e3.k_star" in m else None),
        ("TEST ON LATER DATA, forward hold-out", _H),
        ("macro-F1, all features", g("forward.all.macro_f1")),
        ("fused detector: attack coverage / false-alert rate", f"{m['forward.fused.test_coverage']:.3f} / {m['forward.fused.test_fpr']:.3f}  (target 0.010)" if "forward.fused.test_coverage" in m else None),
        ("sentinel detector: attack coverage / false-alert rate", f"{m['forward.sentinel.test_coverage']:.3f} / {m['forward.sentinel.test_fpr']:.3f}" if "forward.sentinel.test_coverage" in m else None),
        ("DETECTOR TIERS, attack coverage at 1% false alerts", _H),
        ("sentinel (6 features)", ci("sentinel")), ("station (20)", ci("station")), ("fused (52)", ci("fused")),
        ("GRID: DEFAULT vs OPTIMIZED", _H),
        ("verdict vs naive default", lab.get("compare.vs_naive")),
        ("verdict vs budget-matched default", lab.get("compare.vs_budget-matched")),
        ("verdict vs same-lines control", lab.get("compare.vs_bm-lines+tiered")),
        ("coverage, 2 targeted losses: naive -> optimized", f"{m['compare.vs_naive.cov_delivered_n2_targeted.default']:.3f} -> {m['compare.vs_naive.cov_delivered_n2_targeted.optimized']:.3f}" if "compare.vs_naive.cov_delivered_n2_targeted.default" in m else None),
        ("CHECKS", _H),
        ("unit tests", f"{lab['tests.result']} ({m.get('tests.ran', 0):.0f} ran)" if "tests.result" in lab else None),
    ]
    out, header = [], None
    for what, val in rows:
        if val is _H:
            header = what
        elif val is not None:
            if header:
                out.append((f"**{header}**", ""))
                header = None
            out.append((what, val))
    return out


def summary_markdown(r):
    m, lab = flat(r)
    L = [f"# Run {r['run_id']}", "",
         f"- status: **{r['status']}**  |  mode: {'SMOKE (not reportable)' if r['smoke'] else 'full'}  |  "
         f"{r['seconds'] / 60:.1f} min  |  started {r['started']}",
         f"- name/note: {r.get('name') or '-'} / {r.get('note') or '-'}",
         f"- git: {r['git']['commit']}{' (uncommitted changes)' if r['git'].get('dirty') else ''}  |  "
         f"python {r['environment']['python']}  |  dataset {r['config']['dataset']}", ""]
    if r["smoke"]:
        L += ["> SMOKE run: fewer folds/trees/replicates, written to outputs_smoke/. Checks that the "
              "pipeline runs; never quote these numbers.", ""]
    L += ["## Steps", "", "| step | status | seconds | log |", "|---|---|---|---|"]
    for s in r["steps"]:
        L.append(f"| {s['key']} | {s['status']} | {s['seconds']:.1f} | {s.get('log', '')} |")
    head = headline_rows(m, lab)
    if head:
        L += ["", "## Headline", "", "| | |", "|---|---|"] + [f"| {a} | {b} |" for a, b in head]
    cfg = r["config"]
    L += ["", "## Settings that change a number", "",
          f"- CV: {cfg['n_splits']} folds over {cfg['n_blocks']} time blocks, seed {cfg['random_seed']}",
          f"- model: {json.dumps(cfg['model_params'])}",
          f"- detectors: false-alert budget {cfg['operating_fpr']:.0%}, bootstrap B={cfg['bootstrap_B']}",
          f"- grid assumptions hash: {cfg['grid_assumptions_hash']}",
          f"- environment overrides: {cfg['env_overrides'] or 'none'}",
          f"- packages: {json.dumps(r['environment']['packages'])}"]
    L += ["", "## Every recorded number", "", "<details><summary>show</summary>", "", "| metric | value |", "|---|---|"]
    for k, v in sorted(m.items()):
        L.append(f"| {k} | {_num(v)} |")
    for k, v in sorted(lab.items()):
        L.append(f"| {k} | {v} |")
    L += ["", "</details>"]
    return "\n".join(L) + "\n"


def diff_runs(a, b):
    """({setting: (a, b)} that differ, DataFrame of metric a, b, delta)."""
    ca, cb = a["config"], b["config"]
    cfg = {}
    for k in sorted(set(ca) | set(cb)):
        if k in ("grid_assumptions",):
            continue
        if ca.get(k) != cb.get(k):
            cfg[k] = (ca.get(k), cb.get(k))
    ma, la = flat(a)
    mb, lb = flat(b)
    rows = []
    for k in sorted(set(ma) | set(mb)):
        x, y = ma.get(k), mb.get(k)
        rows.append(dict(metric=k, a=x, b=y, delta=(y - x) if x is not None and y is not None else None))
    for k in sorted(set(la) | set(lb)):
        if la.get(k) != lb.get(k):
            rows.append(dict(metric=k, a=la.get(k), b=lb.get(k), delta=None))
    return cfg, pd.DataFrame(rows)


# ------------------------------------------------------------------ optional MLflow mirror
MLFLOW_EXPERIMENT = "smart-grid-attack-detection"


def mlflow_uri(root=None):
    """SQLite backend: the plain-folder (./mlruns) backend is deprecated and refused by current MLflow."""
    return "sqlite:///" + (Path(root or C.PROJECT_ROOT) / "mlflow.db").as_posix()


def mirror_mlflow(record, run_dir, root=None):
    """Copy a finished run into mlflow.db (view: python -m mlflow ui --backend-store-uri sqlite:///mlflow.db).
    Returns a message; never raises, so a missing or broken MLflow cannot fail a run."""
    try:
        import mlflow
    except ImportError:
        return "mlflow is not installed (pip install mlflow); the file records in runs/ are complete without it"
    try:
        root = Path(root or C.PROJECT_ROOT)
        mlflow.set_tracking_uri(mlflow_uri(root))
        if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
            mlflow.create_experiment(MLFLOW_EXPERIMENT, artifact_location=(root / "mlartifacts").as_uri())
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        m, lab = flat(record)
        cfg = record["config"]
        with mlflow.start_run(run_name=record["run_id"]):
            mlflow.log_params({"mode": "smoke" if record["smoke"] else "full", "n_splits": cfg["n_splits"],
                               "n_blocks": cfg["n_blocks"], "seed": cfg["random_seed"],
                               "operating_fpr": cfg["operating_fpr"], "bootstrap_B": cfg["bootstrap_B"],
                               **{f"model.{k}": v for k, v in cfg["model_params"].items()}})
            mlflow.log_metrics({k.replace("+", "_"): float(v) for k, v in m.items()})
            mlflow.set_tags({**{k: str(v) for k, v in lab.items()}, "git": str(record["git"]["commit"]),
                             "status": record["status"], "dataset": str(cfg["dataset"])})
            for f in ("run.json", "summary.md"):
                if (Path(run_dir) / f).exists():
                    mlflow.log_artifact(str(Path(run_dir) / f))
            for f in sorted(Path(run_dir).glob("*.log")):
                mlflow.log_artifact(str(f), artifact_path="logs")
        return "logged to mlflow.db  (view: python -m mlflow ui --backend-store-uri sqlite:///mlflow.db)"
    except Exception as e:                                   # tracking must never fail a run
        return f"mlflow logging failed: {e}"
