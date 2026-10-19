# Train, test and track on your laptop

## Where the model is trained and tested

There is no single "train" button, because the question "how good is it?" has three honest answers here.

| where | what it trains | what it is tested on | use it for |
|---|---|---|---|
| steps 3-4 (blocked CV) | 5 models per experiment, one per fold | the time blocks that fold never saw | **the numbers you report** |
| step 6 | the three final detectors (sentinel / station / fused) -> `models/*.joblib` | out-of-fold scores set the alert threshold; a block bootstrap gives the error bars | what the grid comparison uses |
| step 8 (forward test) | one model on the first 70% of the capture | the last 5 of 20 time blocks, scored once | "what happens on data that comes after" |

`Train.csv -> Test.csv` is **not** a test: Test.csv is the same recording (0.989 macro-F1 is memorisation).
Steps 1, 2, 5 and 7 do not train anything: audit, layer membership, grid design, grid comparison.

## One-time setup (PowerShell, project folder)

```powershell
cd C:\Users\ggcha\Documents\predictive_analytics\predictive_project
py -m venv .venv
.\.venv\Scripts\Activate.ps1          # blocked? run once: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import pandas, sklearn, xgboost; print('ok')"
```

Python 3.11 or newer. Checked on 3.13 with pandas 3.0 and with pandas 2.3.

## Run it

```powershell
python run_pipeline.py --smoke      # ~1-3 min (2.4 on a laptop).  Does every step run?  Writes outputs_smoke/, never outputs/.
python run_pipeline.py              # ~5-15 min (12.8 on a laptop).  Train + test everything, 153 unit tests, record the run.
```

Useful variations:

```powershell
python run_pipeline.py --only 4 6 8                 # just the train/test steps (names work too: --only forward)
python run_pipeline.py --skip 3 --name "no step 3"
python run_pipeline.py --extras                     # + step 5 sensitivity/scaling and step 7 sensitivity (slower)
python run_pipeline.py --keep-going                 # do not stop at the first failing step
python run_pipeline.py --note "after changing max_depth"
```

**Smoke numbers are never reportable.** Smoke uses 3 folds, 40 trees, 60 bootstrap replicates and 3 seeds;
the numbers are visibly different (blocked-CV macro-F1 0.75 instead of 0.71). A run is flagged
`reportable` only when it is the whole pipeline on default settings.

To try a different setting without touching `config.py`, set an environment variable and give the run a name:

```powershell
$env:PP_N_ESTIMATORS = "600"; $env:PP_OUTPUT_DIR = "outputs_try"
python run_pipeline.py --only 4 6 8 --name "600 trees"
Remove-Item Env:PP_N_ESTIMATORS, Env:PP_OUTPUT_DIR
```

Overrides: `PP_N_ESTIMATORS`, `PP_N_SPLITS`, `PP_BOOTSTRAP_B`, `PP_ENSEMBLE_SEEDS`, `PP_OUTPUT_DIR`, `PP_MODELS_DIR`.
They are written into the run record, and any override makes the run non-reportable. A normal run overwrites
`outputs/`; the previous results are still in `runs/<old id>/outputs/`.

## Watch it live

- The terminal shows each step as it prints. Every line is also saved to `runs/<id>/NN_<step>.log`.
- In a second terminal: `python runs.py watch` shows step i of n, seconds in the current step, and the last line printed.
- Step 4 is the slow one (about half the time); step 7 prints a progress line per grid size.
- Ctrl+C stops cleanly and records the run as `interrupted`.

## What a run leaves behind

```
runs/20261018-143055/
    summary.md        read this first: headline numbers, steps, settings (Ctrl+Shift+V in VS Code)
    run.json          everything, machine-readable: every number, settings, package versions, git commit
    status.json       what `runs.py watch` reads
    01_profile.log ... 09_tests.log
    outputs/          copy of the result files from that run (files under 400 KB)
runs/index.csv        one row per run: macro-F1, k*, coverage, forward macro-F1, verdicts, tests
```

`run.json` records the dataset fingerprint, seed, folds, model parameters, false-alert budget, bootstrap size and
a hash of the grid assumptions. Two runs are comparable only if those match, and `runs.py diff` shows when they do not.

```powershell
python runs.py                    # every run, newest last
python runs.py show last          # the full summary of one run (IDs can be abbreviated)
python runs.py diff prev last     # settings that differ, then every number that moved
python runs.py watch              # follow a run in progress
```

## In VS Code

1. **File > Open Folder** on the project. Accept the extension prompt (Python, Rainbow CSV, Data Wrangler).
2. **Ctrl+Shift+P > Python: Select Interpreter > `.venv`**. Everything below uses that interpreter.
3. **Terminal > Run Task...** lists: smoke run, full run, train+test only, unit tests, watch, list/show/compare runs, MLflow.
   *Run Build Task* (Ctrl+Shift+B) starts the full run.
4. **Testing** (beaker icon in the sidebar): the 153 unit tests with green/red marks; run or debug one test.
   Six show as *skipped* until you `pip install networkx mlflow` (optional cross-checks); that is normal.
5. **Run and Debug** (Ctrl+Shift+D): set a breakpoint anywhere in `src/`, pick *Debug step 6* (or *the file I have open*), F5.
6. **Explorer**: open `runs/<id>/summary.md` and press Ctrl+Shift+V; open `runs/index.csv` (Rainbow CSV colours it,
   Data Wrangler sorts and filters it); open any `outputs/*.csv` the same way.
7. **Source Control**: `run.json` and `summary.md` diff nicely; commit them as your experiment log. Logs, the
   result copies and `index.csv` are git-ignored.

## Optional: MLflow dashboard

Only worth it if you want charts across many runs in a browser; the files above already hold everything.

```powershell
pip install mlflow
python run_pipeline.py --mlflow                          # also logs the run to mlflow.db
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db   # open http://127.0.0.1:5000
```

If MLflow is missing or fails, the run still completes and says so.

## Reading the results without fooling yourself

- **Report blocked-CV numbers** (E1 0.711 +/- 0.032). Shuffled CV (0.788) and Train->Test (0.989) are shown
  only to size the inflation.
- **The forward test is the harsher, more realistic number.** Expect about 0.15 lower macro-F1 on later data
  (0.551). It is one split of 330 messages: it says "worse", not by exactly how much. Fault has 8 test messages;
  read its recall as an anecdote.
- The three detector tiers are statistically indistinguishable; do not present one as best.
- E4 latencies depend on your CPU, and so does one step-7 verdict. The tiered design scores a message three times
  (sentinel, station, hub), so it is slower than a centralised design by (sentinel p95 + station p95) minus about
  4 ms of network time it saves. The fixed tolerance is 10 ms. On a fast CPU (tier-2 p95 5.7 ms) the penalty is
  +4.6 ms and the label is OPTIMIZED BETTER; on a laptop (tier-2 p95 10.5 ms) it is +11.4 ms, the performance
  check fails, and the label is NOT ESTABLISHED although all 10 security metrics still win. Larger grids (80
  sites, IEEE 300, Illinois 200) save more network time and pass. Report it as it is; `explaination.txt` PART 6.6
  has the wording and the options.
- Two full runs on the same data give identical numbers except timing columns (E4, step 4 timings).

## When something breaks

| symptom | cause / fix |
|---|---|
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, reopen the terminal |
| `ModuleNotFoundError: sklearn` | the venv is not active, or VS Code picked another interpreter (step 2 above) |
| `Train.csv not found` | the two Kaggle CSVs go in `dataset\` |
| step 5 or 7 exits 2 (`STALE` / `FALLBACK`) | they refuse inputs older than the data; run steps 4 and 6 first (the full run does) |
| a step failed | read `runs\<id>\NN_<step>.log`, fix, rerun just that step: `python run_pipeline.py --only N` |
| unit test fails after `--smoke` | the tests always run on default settings; a failure is real |
| output looks garbled | `$env:PYTHONIOENCODING = "utf-8"` (VS Code's terminal already sets it) |
