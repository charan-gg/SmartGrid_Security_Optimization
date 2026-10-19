# Run 20261018-223917

- status: **failed**  |  mode: full  |  12.8 min  |  started 2026-10-18T22:39:17
- name/note: - / -
- git: 9b3f9fa (uncommitted changes)  |  python 3.14.7  |  dataset fb3e4aae715163c5

## Steps

| step | status | seconds | log |
|---|---|---|---|
| profile | ok | 2.5 | 01_profile.log |
| views | ok | 0.6 | 02_views.log |
| baselines | ok | 106.8 | 03_baselines.log |
| experiments | ok | 377.9 | 04_experiments.log |
| design | ok | 15.3 | 05_design.log |
| detectors | ok | 18.6 | 06_detectors.log |
| compare | ok | 127.6 | 07_compare.log |
| forward | ok | 50.5 | 08_forward.log |
| tests | FAILED (exit 1) | 70.5 | 09_tests.log |

## Headline

| | |
|---|---|
| **TRAIN + TEST, blocked CV** |  |
| macro-F1, all 52 features | 0.714 +/- 0.029 |
| macro-F1, station layers (20) | 0.723 +/- 0.031 |
| attack coverage, all features | 0.576 |
| same data, shuffled CV (leaks neighbours) | 0.789 |
| same data, published Train->Test (same recording) | 0.987 |
| fold-stable features / smallest sufficient k | 6 / 8 |
| **TEST ON LATER DATA, forward hold-out** |  |
| macro-F1, all features | 0.536 |
| fused detector: attack coverage / false-alert rate | 0.503 / 0.033  (target 0.010) |
| sentinel detector: attack coverage / false-alert rate | 0.528 / 0.016 |
| **DETECTOR TIERS, attack coverage at 1% false alerts** |  |
| sentinel (6 features) | 0.625  (95% CI 0.565 to 0.690) |
| station (20) | 0.528  (95% CI 0.488 to 0.624) |
| fused (52) | 0.534  (95% CI 0.492 to 0.658) |
| **GRID: DEFAULT vs OPTIMIZED** |  |
| verdict vs naive default | NOT ESTABLISHED |
| verdict vs budget-matched default | NOT ESTABLISHED |
| verdict vs same-lines control | OPTIMIZED BETTER (narrow) |
| coverage, 2 targeted losses: naive -> optimized | 0.118 -> 0.533 |
| **CHECKS** |  |
| unit tests | FAILED (failures=1, skipped=6) (153 ran) |

## Settings that change a number

- CV: 5 folds over 20 time blocks, seed 42
- model: {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0}
- detectors: false-alert budget 1%, bootstrap B=300
- grid assumptions hash: 13ccc446d3
- environment overrides: none
- packages: {"numpy": "2.5.4", "pandas": "3.0.6", "scikit-learn": "1.9.1", "xgboost": "3.4.1", "scipy": "1.18.1", "matplotlib": "3.11.2", "joblib": "1.6.0"}

## Every recorded number

<details><summary>show</summary>

| metric | value |
|---|---|
| compare.vs_budget-matched.cost_total_kusd.default | 3,579,907 |
| compare.vs_budget-matched.cost_total_kusd.optimized | 3,616,091 |
| compare.vs_budget-matched.cov_delivered_n2_targeted.default | 0.3575 |
| compare.vs_budget-matched.cov_delivered_n2_targeted.optimized | 0.5329 |
| compare.vs_budget-matched.cov_delivered_single.default | 0.5341 |
| compare.vs_budget-matched.cov_delivered_single.optimized | 0.5642 |
| compare.vs_budget-matched.cov_delivered_worst.default | 0 |
| compare.vs_budget-matched.cov_delivered_worst.optimized | 0.4404 |
| compare.vs_budget-matched.spof_total.default | 48.32 |
| compare.vs_budget-matched.spof_total.optimized | 0 |
| compare.vs_naive.cost_total_kusd.default | 2,697,853 |
| compare.vs_naive.cost_total_kusd.optimized | 3,616,091 |
| compare.vs_naive.cov_delivered_n2_targeted.default | 0.1179 |
| compare.vs_naive.cov_delivered_n2_targeted.optimized | 0.5329 |
| compare.vs_naive.cov_delivered_single.default | 0.4502 |
| compare.vs_naive.cov_delivered_single.optimized | 0.5642 |
| compare.vs_naive.cov_delivered_worst.default | 0 |
| compare.vs_naive.cov_delivered_worst.optimized | 0.4404 |
| compare.vs_naive.spof_total.default | 79.28 |
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
| detectors.fused.coverage | 0.5339 |
| detectors.fused.coverage_hi | 0.6583 |
| detectors.fused.coverage_lo | 0.4916 |
| detectors.sentinel.coverage | 0.6246 |
| detectors.sentinel.coverage_hi | 0.6904 |
| detectors.sentinel.coverage_lo | 0.5655 |
| detectors.station.coverage | 0.5281 |
| detectors.station.coverage_hi | 0.6237 |
| detectors.station.coverage_lo | 0.4879 |
| e1.all.attack_coverage | 0.5757 |
| e1.all.macro_f1 | 0.7144 |
| e1.all.macro_f1_sd | 0.02907 |
| e1.all.recall_Fault | 0.9429 |
| e1.all.recall_Injection | 0.3956 |
| e1.all.recall_Masquerade | 0.9291 |
| e1.all.recall_Normal | 0.882 |
| e1.all.recall_Replay | 0.4023 |
| e1.consistency.attack_coverage | 0.3314 |
| e1.consistency.macro_f1 | 0.5033 |
| e1.consistency.macro_f1_sd | 0.02874 |
| e1.control_counters_only.attack_coverage | 0.2836 |
| e1.control_counters_only.macro_f1 | 0.2285 |
| e1.control_counters_only.macro_f1_sd | 0.09842 |
| e1.control_time_only.attack_coverage | 0.3585 |
| e1.control_time_only.macro_f1 | 0.1734 |
| e1.control_time_only.macro_f1_sd | 0.03693 |
| e1.goose.attack_coverage | 0.6287 |
| e1.goose.macro_f1 | 0.6288 |
| e1.goose.macro_f1_sd | 0.04291 |
| e1.process.attack_coverage | 0.1563 |
| e1.process.macro_f1 | 0.3081 |
| e1.process.macro_f1_sd | 0.0186 |
| e1.relay.attack_coverage | 0.01614 |
| e1.relay.macro_f1 | 0.2531 |
| e1.relay.macro_f1_sd | 0.03816 |
| e1.station.attack_coverage | 0.6153 |
| e1.station.macro_f1 | 0.723 |
| e1.station.macro_f1_sd | 0.03145 |
| e3.k_star | 8 |
| e3.stable_features | 6 |
| e4.tier1_edge_kstar.p95_ms | 4.899 |
| e4.tier1_edge_process_only.p95_ms | 7.023 |
| e4.tier2_zone_fused.p95_ms | 10.5 |
| e4.tier3_widearea_fused.p95_ms | 10.17 |
| e6.recall_at_2000_false_alerts_per_day | 0.7826 |
| e8.mean_gap_all | 0.2914 |
| forward.all.macro_f1 | 0.5356 |
| forward.fused.test_coverage | 0.503 |
| forward.fused.test_fpr | 0.03279 |
| forward.process.macro_f1 | 0.2624 |
| forward.sentinel.test_coverage | 0.5281 |
| forward.sentinel.test_fpr | 0.01639 |
| forward.station.macro_f1 | 0.5714 |
| forward.station.test_coverage | 0.3192 |
| forward.station.test_fpr | 0.008197 |
| split.all.blocked_cv_macro_f1 | 0.7144 |
| split.all.published_train_to_test_macro_f1 | 0.9873 |
| split.all.shuffled_cv_macro_f1 | 0.7887 |
| split.process.blocked_cv_macro_f1 | 0.3081 |
| split.process.published_train_to_test_macro_f1 | 1 |
| split.process.shuffled_cv_macro_f1 | 0.284 |
| split.station.blocked_cv_macro_f1 | 0.723 |
| split.station.published_train_to_test_macro_f1 | 0.9819 |
| split.station.shuffled_cv_macro_f1 | 0.8114 |
| tests.ran | 153 |
| compare.vs_bm-lines+tiered | OPTIMIZED BETTER (narrow) |
| compare.vs_budget-matched | NOT ESTABLISHED |
| compare.vs_naive | NOT ESTABLISHED |
| design.feasible | True |
| tests.result | FAILED (failures=1, skipped=6) |

</details>
