"""
Train, test and track: run the whole project with one command, and keep a record of the run.

    python run_pipeline.py                       # everything: steps 1-8 + the unit tests  (~5-8 min)
    python run_pipeline.py --smoke               # 1-2 min check that it all runs; writes outputs_smoke/, NOT outputs/
    python run_pipeline.py --only 4 6 8          # just those steps (numbers, or: tests)
    python run_pipeline.py --skip 3 --name "try depth 8"
    python run_pipeline.py --extras              # + step-5 sensitivity/scaling and step-7 sensitivity (slower)
    python run_pipeline.py --mlflow              # also log the run to ./mlflow.db (needs: pip install mlflow)

What each step is (the model is trained and tested in 3, 4, 6 and 8):
    1 profile      data audit                      2 views     which column belongs to which layer
    3 baselines    how much the split flatters it  4 experiments  E1 E3 E4 E6 E8: blocked-CV training + testing
    5 design       build the grid from the results 6 detectors   final tier models + recall error bars (-> models/)
    7 compare      default vs optimized grid       8 forward     train early / test late hold-out
    tests          117 unit tests

Everything a step prints goes to runs/<run_id>/NN_<step>.log; the numbers it produced, the settings and
the environment go to runs/<run_id>/run.json and summary.md. See `python runs.py` to list and compare runs.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# (key, number, script, base args). "tests" has no number.
STEPS = [
    ("profile", "1", "step1_profile.py", []),
    ("views", "2", "step2_views.py", []),
    ("baselines", "3", "step3_split_and_baseline.py", []),
    ("experiments", "4", "step4_run_experiments.py", []),
    ("design", "5", "step5_design_grid.py", ["--require-measured"]),
    ("detectors", "6", "step6_train_detectors.py", []),
    ("compare", "7", "step7_compare_grids.py", ["--require-measured"]),
    ("forward", "8", "step8_forward_test.py", []),
    ("tests", "tests", None, []),
]
SMOKE_ENV = dict(PP_OUTPUT_DIR="outputs_smoke", PP_MODELS_DIR="models_smoke", PP_N_SPLITS="3",
                 PP_N_ESTIMATORS="40", PP_BOOTSTRAP_B="60", PP_ENSEMBLE_SEEDS="3")
SMOKE_ARGS = {"compare": ["--sizes", "20", "40", "--seeds", "3", "--no-cases", "--no-figure"],
              "design": ["--no-plot"]}
EXTRA_ARGS = {"design": ["--sensitivity", "--scaling"], "compare": ["--sensitivity"]}


def select(only, skip):
    keys = {k: n for k, n, _, _ in STEPS}
    numbers = {n: k for k, n, _, _ in STEPS}

    def norm(x):
        x = str(x).lower()
        if x in keys:
            return x
        if x in numbers:
            return numbers[x]
        raise SystemExit(f"unknown step {x!r}; use one of: " + ", ".join(f"{n}={k}" for k, n, _, _ in STEPS))
    only = [norm(x) for x in only] if only else [k for k, _, _, _ in STEPS]
    skip = {norm(x) for x in skip or []}
    return [s for s in STEPS if s[0] in only and s[0] not in skip]


def is_reportable(smoke, only, environ):
    """A run's numbers can go in the report only if it is the whole pipeline on the default settings."""
    return not smoke and not only and not any(k.startswith("PP_") for k in environ)


def step_command(step, smoke, extras):
    key, _, script, base = step
    if key == "tests":
        return [sys.executable, "-u", "-m", "unittest", "discover", "-s", "tests", "-v"]
    args = list(base) + (SMOKE_ARGS.get(key, []) if smoke else []) + (EXTRA_ARGS.get(key, []) if extras and not smoke else [])
    return [sys.executable, "-u", script] + args


def emit(line):
    try:
        sys.stdout.write(line)
    except UnicodeEncodeError:
        sys.stdout.write(line.encode("ascii", "replace").decode())
    sys.stdout.flush()


def run_step(step, idx, total, run_dir, env, smoke, extras, status):
    from src import tracking as TR
    key = step[0]
    cmd = step_command(step, smoke, extras)
    log_path = run_dir / f"{idx:02d}_{key}.log"
    emit(f"\n=== [{idx}/{total}] {key}: {' '.join(cmd[1:])}\n")
    t0, text = time.time(), []
    last_beat = 0.0
    status.update(step=key, step_index=idx, step_started=t0, last_output=t0, last_line="(starting)")
    TR.write_json(run_dir / "status.json", status)
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        step_env = {k: v for k, v in env.items() if not k.startswith("PP_")} if key == "tests" else env   # tests check the code, not the smoke settings
        proc = subprocess.Popen(cmd, cwd=ROOT, env=step_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        try:
            for line in proc.stdout:
                emit(line)
                log.write(line)
                text.append(line)
                if time.time() - last_beat > 1.0:
                    last_beat = time.time()
                    status.update(last_output=last_beat, last_line=line.strip()[:200])
                    TR.write_json(run_dir / "status.json", status)
            proc.wait()
        except KeyboardInterrupt:
            proc.kill()
            raise
    seconds = time.time() - t0
    ok = proc.returncode == 0
    rec = dict(key=key, status="ok" if ok else f"FAILED (exit {proc.returncode})", seconds=round(seconds, 1),
               log=log_path.name, metrics={}, labels={})
    from src import config as C
    out = C.OUTPUT_DIR
    if key == "tests":
        rec["metrics"], rec["labels"] = TR.harvest_tests("".join(text))
        if not ok and rec["labels"].get("tests.result") == "OK":
            rec["labels"]["tests.result"] = "unknown"
    elif ok and key in TR.HARVEST:
        rec["metrics"], rec["labels"] = TR.HARVEST[key](out)
    emit(f"=== {key}: {rec['status']} in {seconds:.1f}s" + (f"  |  {len(rec['metrics'])} numbers recorded" if rec["metrics"] else "") + "\n")
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="step numbers or names (1-8, tests)")
    ap.add_argument("--skip", nargs="*", help="step numbers or names to skip")
    ap.add_argument("--smoke", action="store_true", help="fast, non-reportable check; writes outputs_smoke/ and models_smoke/")
    ap.add_argument("--extras", action="store_true", help="include the slow sensitivity/scaling sweeps")
    ap.add_argument("--name", default=None, help="a short label for this run")
    ap.add_argument("--note", default=None, help="why you ran it")
    ap.add_argument("--keep-going", action="store_true", help="carry on after a failing step")
    ap.add_argument("--no-snapshot", action="store_true", help="do not copy the result files into the run folder")
    ap.add_argument("--mlflow", action="store_true", help="also log to ./mlflow.db (optional; pip install mlflow)")
    args = ap.parse_args()

    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    if args.smoke:
        env.update(SMOKE_ENV)
    os.environ.update({k: v for k, v in env.items() if k.startswith("PP_")})       # so src.config sees the same settings
    from src import config as C
    from src import tracking as TR

    steps = select(args.only, args.skip)
    if not steps:
        raise SystemExit("nothing to run")
    smoke = args.smoke
    run_id = TR.new_run_id(smoke, args.name)
    run_dir = TR.RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now()
    emit(f"run {run_id}   ({'SMOKE - numbers are not reportable' if smoke else 'full'}; results -> {C.OUTPUT_DIR.name}/, record -> runs/{run_id}/)\n")

    record = dict(run_id=run_id, name=args.name, note=args.note, smoke=smoke, reportable=is_reportable(smoke, args.only, os.environ),
                  started=started.isoformat(timespec="seconds"), status="running", seconds=0.0,
                  command=" ".join(sys.argv), git=TR.git_state(), environment=TR.environment(),
                  config=TR.config_snapshot(), steps=[])
    t_start = time.time()
    status = dict(run_id=run_id, state="running", n_steps=len(steps), step=None, step_index=0,
                  started=t_start, step_started=t_start, last_output=t_start, last_line="")
    TR.write_json(run_dir / "status.json", status)
    t0 = time.time()
    final = "ok"
    try:
        for i, step in enumerate(steps, 1):
            rec = run_step(step, i, len(steps), run_dir, env, smoke, args.extras, status)
            record["steps"].append(rec)
            if rec["status"] != "ok":
                final = "failed"
                if not args.keep_going:
                    break
    except KeyboardInterrupt:
        final = "interrupted"
        emit("\ninterrupted\n")
    record["status"], record["seconds"] = final, round(time.time() - t0, 1)
    TR.write_json(run_dir / "run.json", record)
    (run_dir / "summary.md").write_text(TR.summary_markdown(record), encoding="utf-8")
    if not args.no_snapshot and C.OUTPUT_DIR.exists():
        TR.snapshot_outputs(C.OUTPUT_DIR, run_dir)
    status.update(state=final, finished=time.time())
    TR.write_json(run_dir / "status.json", status)
    TR.rebuild_index()
    if args.mlflow:
        emit("mlflow: " + TR.mirror_mlflow(record, run_dir) + "\n")

    emit(f"\n{'=' * 72}\nrun {run_id}: {final.upper()} in {record['seconds'] / 60:.1f} min\n")
    for s in record["steps"]:
        emit(f"  {s['key']:<12} {s['status']:<18} {s['seconds']:>7.1f}s\n")
    m, lab = TR.flat(record)
    for label, key in (("blocked-CV macro-F1, all layers", "e1.all.macro_f1"), ("blocked-CV macro-F1, station layers", "e1.station.macro_f1"),
                       ("fused detector attack coverage", "detectors.fused.coverage"), ("forward-test macro-F1", "forward.all.macro_f1")):
        if key in m:
            emit(f"  {label:<38} {m[key]:.3f}\n")
    for k in ("compare.vs_naive", "compare.vs_budget-matched", "tests.result"):
        if k in lab:
            emit(f"  {k:<38} {lab[k]}\n")
    emit(f"  summary: runs/{run_id}/summary.md   |   compare runs: python runs.py\n")
    if smoke:
        emit("  SMOKE run: pipeline works end to end. Do not quote these numbers.\n")
    sys.exit(0 if final == "ok" else 1)


if __name__ == "__main__":
    main()
