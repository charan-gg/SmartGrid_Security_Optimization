# Run 20261018-222624_smoke

- status: **ok**  |  mode: SMOKE (not reportable)  |  2.4 min  |  started 2026-10-18T22:26:24
- name/note: - / -
- git: 9b3f9fa (uncommitted changes)  |  python 3.14.7  |  dataset fb3e4aae715163c5

> SMOKE run: fewer folds/trees/replicates, written to outputs_smoke/. Checks that the pipeline runs; never quote these numbers.

## Steps

| step | status | seconds | log |
|---|---|---|---|
| profile | ok | 48.5 | 01_profile.log |
| views | ok | 0.6 | 02_views.log |
| baselines | ok | 14.2 | 03_baselines.log |
| experiments | ok | 31.8 | 04_experiments.log |
| design | ok | 0.9 | 05_design.log |
| detectors | ok | 4.7 | 06_detectors.log |
| compare | ok | 3.5 | 07_compare.log |
| forward | ok | 6.7 | 08_forward.log |
| tests | ok | 31.2 | 09_tests.log |

## Headline

| | |
|---|---|
| **TRAIN + TEST, blocked CV** |  |
| macro-F1, all 52 features | 0.765 +/- 0.014 |
| macro-F1, station layers (20) | 0.764 +/- 0.020 |
| attack coverage, all features | 0.607 |
| same data, shuffled CV (leaks neighbours) | 0.768 |
| same data, published Train->Test (same recording) | 0.944 |
| fold-stable features / smallest sufficient k | 7 / 12 |
| **TEST ON LATER DATA, forward hold-out** |  |
| macro-F1, all features | 0.590 |
| fused detector: attack coverage / false-alert rate | 0.482 / 0.016  (target 0.010) |
| sentinel detector: attack coverage / false-alert rate | 0.556 / 0.008 |
| **DETECTOR TIERS, attack coverage at 1% false alerts** |  |
| sentinel (6 features) | 0.576  (95% CI 0.506 to 0.673) |
| station (20) | 0.607  (95% CI 0.588 to 0.670) |
| fused (52) | 0.625  (95% CI 0.593 to 0.698) |
| **GRID: DEFAULT vs OPTIMIZED** |  |
| verdict vs naive default | OPTIMIZED BETTER |
| verdict vs budget-matched default | OPTIMIZED BETTER |
| verdict vs same-lines control | OPTIMIZED BETTER (narrow) |
| coverage, 2 targeted losses: naive -> optimized | 0.144 -> 0.598 |
| **CHECKS** |  |
| unit tests | OK (153 ran) |

## Settings that change a number

- CV: 3 folds over 20 time blocks, seed 42
- model: {"n_estimators": 40, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0}
- detectors: false-alert budget 1%, bootstrap B=60
- grid assumptions hash: 13ccc446d3
- environment overrides: {'PP_OUTPUT_DIR': 'outputs_smoke', 'PP_MODELS_DIR': 'models_smoke', 'PP_N_SPLITS': '3', 'PP_N_ESTIMATORS': '40', 'PP_BOOTSTRAP_B': '60', 'PP_ENSEMBLE_SEEDS': '3'}
- packages: {"numpy": "2.5.4", "pandas": "3.0.6", "scikit-learn": "1.9.1", "xgboost": "3.4.1", "scipy": "1.18.1", "matplotlib": "3.11.2", "joblib": "1.6.0"}

## Every recorded number

<details><summary>show</summary>

| metric | value |
|---|---|
| compare.vs_budget-matched.cost_total_kusd.default | 2,365,280 |
| compare.vs_budget-matched.cost_total_kusd.optimized | 2,404,489 |
| compare.vs_budget-matched.cov_delivered_n2_targeted.default | 0.3962 |
| compare.vs_budget-matched.cov_delivered_n2_targeted.optimized | 0.5978 |
| compare.vs_budget-matched.cov_delivered_single.default | 0.594 |
| compare.vs_budget-matched.cov_delivered_single.optimized | 0.6363 |
| compare.vs_budget-matched.cov_delivered_worst.default | 0 |
| compare.vs_budget-matched.cov_delivered_worst.optimized | 0.487 |
| compare.vs_budget-matched.spof_total.default | 31.17 |
| compare.vs_budget-matched.spof_total.optimized | 0 |
| compare.vs_naive.cost_total_kusd.default | 1,770,017 |
| compare.vs_naive.cost_total_kusd.optimized | 2,404,489 |
| compare.vs_naive.cov_delivered_n2_targeted.default | 0.144 |
| compare.vs_naive.cov_delivered_n2_targeted.optimized | 0.5978 |
| compare.vs_naive.cov_delivered_single.default | 0.4857 |
| compare.vs_naive.cov_delivered_single.optimized | 0.6363 |
| compare.vs_naive.cov_delivered_worst.default | 0 |
| compare.vs_naive.cov_delivered_worst.optimized | 0.487 |
| compare.vs_naive.spof_total.default | 50 |
| compare.vs_naive.spof_total.optimized | 0 |
| data.merging_units | 4 |
| data.row_rate_hz | 2.294 |
| data.rows | 1094 |
| design.levels | 4 |
| design.lines | 47 |
| design.n1_premium_pct | 30.1 |
| design.pmus | 12 |
| design.total_cost_kusd | 3,081,620 |
| design.zones | 6 |
| detectors.fused.coverage | 0.6245 |
| detectors.fused.coverage_hi | 0.6977 |
| detectors.fused.coverage_lo | 0.5927 |
| detectors.sentinel.coverage | 0.5757 |
| detectors.sentinel.coverage_hi | 0.6734 |
| detectors.sentinel.coverage_lo | 0.506 |
| detectors.station.coverage | 0.6072 |
| detectors.station.coverage_hi | 0.6703 |
| detectors.station.coverage_lo | 0.5882 |
| e1.all.attack_coverage | 0.6065 |
| e1.all.macro_f1 | 0.7647 |
| e1.all.macro_f1_sd | 0.01435 |
| e1.all.recall_Fault | 0.9512 |
| e1.all.recall_Injection | 0.491 |
| e1.all.recall_Masquerade | 0.9673 |
| e1.all.recall_Normal | 0.9589 |
| e1.all.recall_Replay | 0.3613 |
| e1.consistency.attack_coverage | 0.3148 |
| e1.consistency.macro_f1 | 0.4987 |
| e1.consistency.macro_f1_sd | 0.01667 |
| e1.control_counters_only.attack_coverage | 0.2905 |
| e1.control_counters_only.macro_f1 | 0.2779 |
| e1.control_counters_only.macro_f1_sd | 0.01336 |
| e1.control_time_only.attack_coverage | 0.2739 |
| e1.control_time_only.macro_f1 | 0.2086 |
| e1.control_time_only.macro_f1_sd | 0.01308 |
| e1.goose.attack_coverage | 0.6255 |
| e1.goose.macro_f1 | 0.6815 |
| e1.goose.macro_f1_sd | 0.01498 |
| e1.process.attack_coverage | 0.1359 |
| e1.process.macro_f1 | 0.2772 |
| e1.process.macro_f1_sd | 0.01276 |
| e1.relay.attack_coverage | 0.01622 |
| e1.relay.macro_f1 | 0.263 |
| e1.relay.macro_f1_sd | 0.003313 |
| e1.station.attack_coverage | 0.6251 |
| e1.station.macro_f1 | 0.7643 |
| e1.station.macro_f1_sd | 0.02002 |
| e3.k_star | 12 |
| e3.stable_features | 7 |
| e4.tier1_edge_kstar.p95_ms | 3.088 |
| e4.tier1_edge_process_only.p95_ms | 5.342 |
| e4.tier2_zone_fused.p95_ms | 7.535 |
| e4.tier3_widearea_fused.p95_ms | 8.204 |
| e6.recall_at_2000_false_alerts_per_day | 0.8194 |
| e8.mean_gap_all | 0.2317 |
| forward.all.macro_f1 | 0.5896 |
| forward.fused.test_coverage | 0.4823 |
| forward.fused.test_fpr | 0.01639 |
| forward.process.macro_f1 | 0.2352 |
| forward.sentinel.test_coverage | 0.5564 |
| forward.sentinel.test_fpr | 0.008197 |
| forward.station.macro_f1 | 0.6428 |
| forward.station.test_coverage | 0.1335 |
| forward.station.test_fpr | 0.008197 |
| split.all.blocked_cv_macro_f1 | 0.7647 |
| split.all.published_train_to_test_macro_f1 | 0.944 |
| split.all.shuffled_cv_macro_f1 | 0.7683 |
| split.process.blocked_cv_macro_f1 | 0.2772 |
| split.process.published_train_to_test_macro_f1 | 0.6792 |
| split.process.shuffled_cv_macro_f1 | 0.2908 |
| split.station.blocked_cv_macro_f1 | 0.7643 |
| split.station.published_train_to_test_macro_f1 | 0.9416 |
| split.station.shuffled_cv_macro_f1 | 0.7851 |
| tests.ran | 153 |
| compare.vs_bm-lines+tiered | OPTIMIZED BETTER (narrow) |
| compare.vs_budget-matched | OPTIMIZED BETTER |
| compare.vs_naive | OPTIMIZED BETTER |
| design.feasible | True |
| tests.result | OK |

</details>
