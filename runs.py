"""
Look at the runs run_pipeline.py has recorded.

    python runs.py                 list every run (newest last)
    python runs.py show [ID|last]  the full summary of one run
    python runs.py diff A B        settings and numbers that changed between two runs (IDs, 'last', 'prev')
    python runs.py watch           follow the run in progress: current step, elapsed, last line printed
"""
import json
import sys
import time

import pandas as pd

from src import tracking as TR


def resolve(runs, ref):
    if not runs:
        raise SystemExit("no runs yet: python run_pipeline.py --smoke")
    ids = [r["run_id"] for r in runs]
    if ref in (None, "last"):
        return runs[-1]
    if ref == "prev":
        return runs[-2] if len(runs) > 1 else runs[-1]
    hits = [r for r in runs if r["run_id"] == ref or r["run_id"].startswith(ref) or ref in r["run_id"]]
    if len(hits) != 1:
        raise SystemExit(f"{ref!r} matches {len(hits)} runs; known: " + ", ".join(ids[-8:]))
    return hits[0]


def short_verdict(v):
    if not isinstance(v, str):
        return "-"
    return v.replace("OPTIMIZED BETTER", "better").replace("NOT ESTABLISHED", "not est.").replace(" (narrow)", "*")


def compact_table(runs):
    """One narrow row per run (fits a 120-column terminal); full detail is `runs.py show`."""
    df = TR.index_rows(runs)
    out = pd.DataFrame({
        "run": df["run_id"].str.slice(0, 36),
        "mode": [("smoke" if m == "smoke" else "full" if rep else "partial") for m, rep in zip(df["mode"], df["reportable"])],
        "min": df["minutes"].round(1),
        "F1 blk": df["macro_f1_blocked_all"], "F1 stn": df["macro_f1_blocked_station"],
        "k*": df["k_star"], "cov fused": df["coverage_fused"], "F1 fwd": df["forward_macro_f1_all"],
        "vs naive": df["verdict_vs_naive"].map(short_verdict), "vs b-match": df["verdict_vs_budget_matched"].map(short_verdict),
        "tests": df["tests"].map(lambda v: "OK" if v == "OK" else ("-" if not isinstance(v, str) else "FAIL"))})
    out.loc[df["status"] != "ok", "mode"] = df["status"]
    return out


def cmd_list():
    runs = TR.load_runs()
    if not runs:
        print("no runs yet: python run_pipeline.py --smoke")
        return
    pd.set_option("display.width", 200)
    print(compact_table(runs).to_string(index=False, na_rep="-", float_format=lambda x: f"{x:.3f}"))
    print("\nmode: full = whole pipeline on default settings (reportable). smoke, and partial (--only or changed settings), are not.   * = narrow win.   detail: python runs.py show <run>")


def cmd_show(ref):
    r = resolve(TR.load_runs(), ref)
    print((TR.RUNS_DIR / r["run_id"] / "summary.md").read_text(encoding="utf-8"))


def cmd_diff(a, b):
    runs = TR.load_runs()
    ra, rb = resolve(runs, a), resolve(runs, b)
    cfg, df = TR.diff_runs(ra, rb)
    print(f"A = {ra['run_id']}   B = {rb['run_id']}\n")
    print("settings that differ:" if cfg else "settings: identical")
    for k, (x, y) in cfg.items():
        print(f"  {k}: {x}  ->  {y}")
    df = df[(df["delta"].isna()) | (df["delta"].abs() > 1e-9)]
    print(f"\nnumbers that changed ({len(df)}):" if len(df) else "\nno number changed")
    if len(df):
        pd.set_option("display.width", 200)
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))


def format_status(s, now):
    """One line: where the run is, how long the step has taken, how long since it last printed."""
    if s["state"] != "running":
        return f"[{s['run_id']}] {s['state'].upper()}"
    in_step = now - s.get("step_started", now)
    quiet = now - s.get("last_output", now)
    quiet_txt = f"  (quiet for {quiet:.0f}s)" if quiet >= 10 else ""
    return (f"[{s['run_id']}] step {s['step_index']}/{s['n_steps']} {s['step']}  {in_step:.0f}s{quiet_txt}"
            f"  | {s['last_line']}")


def cmd_watch():
    prev_step = None
    while True:
        files = sorted(TR.RUNS_DIR.glob("*/status.json"))
        if not files:
            print("no run in progress")
            return
        try:
            s = json.loads(files[-1].read_text())
        except ValueError:                                   # caught mid-write: try again
            time.sleep(0.5)
            continue
        line = format_status(s, time.time())
        if s["state"] != "running":
            print("\r" + line.ljust(150))
            return
        if s["step_index"] != prev_step and prev_step is not None:
            print()
        prev_step = s["step_index"]
        sys.stdout.write("\r" + line[:150].ljust(150))
        sys.stdout.flush()
        time.sleep(1)


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "list"
    if cmd == "list":
        cmd_list()
    elif cmd == "show":
        cmd_show(argv[2] if len(argv) > 2 else "last")
    elif cmd == "diff":
        if len(argv) < 4:
            raise SystemExit("usage: python runs.py diff A B     (IDs, 'last', 'prev')")
        cmd_diff(argv[2], argv[3])
    elif cmd == "watch":
        cmd_watch()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
