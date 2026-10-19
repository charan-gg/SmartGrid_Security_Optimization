# Code map

```
dataset/            Train.csv, Test.csv (not committed)
dataset/grids/      7 standard test grids as CSV (IEEE 14/30/39/57/118/300, Illinois 200) - committed; see its README
src/config.py       every choice in one place: paths, classes, blocks, model budget, grid assumptions
src/data.py         loader + asserts (T1-T4), time-ordered blocks, rates(), audit() of Train vs Test
src/modalities.py   column -> layer (process / goose / consistency / relay), views, MU count
src/splits.py       blocked (honest) and naive_rows (the leaky one, kept to size the inflation)
src/models.py       one frozen model budget; labels re-indexed to the classes a fold has seen
src/metrics.py      macro-F1, per-class recall, PR-AUC, attack_coverage, false alerts/day, bytes/s
src/experiments/    e1 views, e3 feature budget, e4 compute profile, e6 operating point, e8 unseen attack
src/detectors.py    step 6: three detector tiers, alert threshold at a fixed false-alert budget, block bootstrap
src/forward_test.py step 8: time hold-out split, thresholds from the training part only
src/tracking.py     run records: settings, environment, harvested numbers, summary.md, index, diff, optional MLflow
src/design/         step 5: graph.py (BFS/Dijkstra/MST/bridges/max-flow) + grid synthesis
                    step 7: cases.py (standard grids), baseline.py (default grids), compare.py (scenarios, metrics, verdict)
run_pipeline.py     one command for steps 1-8 + tests; logs every step and records the run in runs/
runs.py             list / show / diff / watch the recorded runs
.vscode/            tasks, debug launchers, test discovery, recommended extensions
tools/              export_cases.py: pandapower -> dataset/grids CSV (needs pandapower; not needed to run the project)
models/             tier detectors fitted by step 6 (not committed)
tests/              153 tests: data layer, splits, audit, provenance, graph routines, design invariants,
                    detectors, case loader, default builders, failure scenarios, verdict rules,
                    forward split, run tracker
explaination.txt    plain-text walkthrough of the dataset, every step and every algorithm (read this first)
outputs/            everything the steps write; step 5 reads it
outputs_smoke/      the same from a --smoke run (never reportable; git-ignored)
runs/               one folder per pipeline run: summary.md, run.json, per-step logs, copy of the results
```

## What the code asserts and stops on

- T1 Train and Test have the same columns; T2 only the five known classes; T3 numeric features, no NaN;
  T4 a usable time axis.
- A blocked fold never has a time block on both sides and always contains all five classes.
- `time` / `t_ms` / labels / block ids are never features.
- Every feature column maps to a known layer (unknown columns are reported, not guessed).
- Step 5 inputs are tagged measured / FALLBACK / STALE (older than Train.csv); `--require-measured` exits 2
  on any that is not fresh. `outputs/step5_inputs.json` records the dataset fingerprint they came from.

## Step 5 stages

| stage | question | method | module |
|---|---|---|---|
| A | which lines? | Kruskal MST + augmentation until no bridge / cut vertex (DFS low-link) | `power.py` |
| B | where do PMUs go? | minimum dominating set (scipy milp, greedy fallback) + zero-injection rule | `observability.py` |
| C | how many levels, where are hubs and the CC? | all-pairs Dijkstra, k-medoid zoning, smallest feasible k | `hierarchy.py` |
| D | does it survive failure / attack? | BFS failure sweeps, lateral movement, max-flow (Menger) | `security.py` |
| E | what does it buy vs centralised? | shortest-path traffic routing, A/B/B+/C comparison, role -> model map | `evaluate.py` |

Naming: the design code keeps the "PMU" vocabulary; a PMU / MU sensor here is a process-bus merging-unit feed.
A station emits at most one verdict per scored row, so verdict rate is capped at the measured row rate
(`evaluate.verdict_rate_hz`).

```powershell
.\.venv\Scripts\python step5_design_grid.py --require-measured --n 80 --seed 7
.\.venv\Scripts\python step5_design_grid.py --sites region.csv      # id,x_km,y_km,kind,mw
.\.venv\Scripts\python step5_design_grid.py --sensitivity --scaling
```

## Steps 6-7: default vs optimized

Step 6 turns the data into three numbers; step 7 lets the graph decide who gets which.

| step | question | method | module |
|---|---|---|---|
| 6 | how well does each tier of evidence catch an attack, and how sure are we? | out-of-fold attack score, threshold at a 1% false-alert budget, 20-block bootstrap | `detectors.py` |
| 7a | what would a builder do without optimizing? | radial MST + PMU everywhere + one control centre; same-spend untargeted lines + standby; as-built | `baseline.py` |
| 7b | which substations are still watched after a failure? | BFS on the surviving graph; fused recall where a PMU observes, station recall elsewhere | `compare.py` |
| 7c | is the optimized grid better, and can we tell? | paired two-level bootstrap, per-metric tolerance, pre-declared overall rule | `compare.py` |

What is measured: detector recall (step 6). What is simulated: every grid. The verdict rules are in the
`compare.py` docstring and printed with the results; `step7` also runs a self-check (optimized vs itself, and
with the roles swapped) to show the rule returns "not established" when it should.

```powershell
.\.venv\Scripts\python step6_train_detectors.py
.\.venv\Scripts\python step7_compare_grids.py --require-measured --sensitivity
.\.venv\Scripts\python step7_compare_grids.py --sizes 40 --seeds 10       # quick look
```

## Smoke runs and overrides

`python run_pipeline.py --smoke` sets `PP_N_SPLITS=3 PP_N_ESTIMATORS=40 PP_BOOTSTRAP_B=60 PP_ENSEMBLE_SEEDS=3` and
redirects results to `outputs_smoke/` and `models_smoke/` (`PP_OUTPUT_DIR`, `PP_MODELS_DIR`). Never set them for
numbers you report; any override marks the run as not reportable. See RUNNING.md.
