# Steps 1-4 - code

Implements steps 1-4 of the plan in `claude/project-brief.md`: acquire+profile,
align+features, split safely+baselines, run E1-E8.

## Run

```bash
./.venv/bin/python step1_profile.py            # Phase 0 verification -> Appendix A
./.venv/bin/python step2_features.py           # modality views + separability pre-check
./.venv/bin/python step3_split_and_baseline.py # leakage control + the three baselines
./.venv/bin/python step4_run_experiments.py    # E1..E8   (--only e1 e2 / --skip e8)
```

Results land in `outputs/`. Step 1 writes a parquet cache to `outputs/cache/`;
delete it after changing anything in `src/loading.py` or `src/config.py`.

## Install first

`xgboost` and `shap` are NOT in the venv, and the venv has no pip:

```bash
python3.11 -m venv --upgrade-deps .venv     # or: uv pip install -r requirements.txt
./.venv/bin/pip install -r requirements.txt
```

Without xgboost the code falls back to sklearn's HistGradientBoostingClassifier.
It runs and the numbers are valid, but: it is ~10x slower to rank features
(permutation importance instead of split gain), and E7 falls back from SHAP to
per-class permutation importance. Install both before generating reported
numbers.

## Layout

```
src/config.py       every choice in one place; print it for Appendix A
src/modalities.py   column -> Purdue layer. The thesis, as code.
src/loading.py      loads + asserts the 3 structural facts; Inf policy
src/splits.py       by_event (honest) / by_scenario (E8) / naive_rows (the trap)
src/models.py       one frozen model budget shared by every view
src/metrics.py      macro-F1, per-class recall, PR-AUC, alerts/day
src/experiments/    e1..e8, one file each
```

## Env overrides (smoke testing only)

`PP_N_SPLITS` and `PP_N_ESTIMATORS` shrink the run so the whole pipeline
finishes in ~2 minutes. Never set them for a run whose numbers get reported.

```bash
PP_N_SPLITS=2 PP_N_ESTIMATORS=25 ./.venv/bin/python step4_run_experiments.py
```

## Things the code asserts, and will stop on

- multiclass/ and triple/ are row-aligned (the positional label join)
- each scenario is ONE contiguous block per file (the definition of an event)
- no event instance appears on both sides of a split
- every feature column maps to a known modality
