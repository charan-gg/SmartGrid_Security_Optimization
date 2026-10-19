# Smart-grid attack detection - Predictive Learning project

Detect cyber-physical attacks on a substation from IEC 61850 GOOSE traffic, then use what
the measurements say to **design** the detection hierarchy of a grid (graph algorithms,
step 5) instead of assuming one.

## Data

Kaggle *Power System Intrusion Dataset* (sequincozes). Put the two files here:

```
dataset/Train.csv
dataset/Test.csv
```

One row = one GOOSE message seen by a 4-merging-unit (MU1..MU4) substation testbed.
52 features + `time` + `class`. Classes: Normal 455, Masquerade 397, Injection 101,
Replay 100, Fault 41 (Normal / Natural fault / Attack).

| layer | columns | what it is |
|---|---|---|
| process (32) | threePhaseV/C*, MU*VoltageAngle*, MU*CurrentAngle*, IED4_i* | raw MU phasors |
| goose (7) | sqNum, stnum, state_cb, sqDiff, stDiff, timeLastMsg, recentChange | protocol state |
| consistency (8) | MU*Cs, Cs, consistency, threePhase*Sum | cross-MU checks at the station |
| relay (5) | MU*Log, any_relay | relay logs |
| station (20) | goose + consistency + relay | everything a station correlator sees without raw phasors |

## The one thing to know about this data

**Test.csv is the same recording as Train.csv.** All 1,094 (time, class) messages appear in
both files; 48 of 52 feature columns are identical and only the four timing columns
(`timeLastMsg`, `stDiff`, `sqDiff`, `recentChange`) differ. Train's timing columns can be
rebuilt from its own time-ordered rows (97-99%), Test's cannot (31-35%).
A Train->Test score therefore measures memorisation, not generalisation, so **no reported
number uses Test.csv**. Step 1 prints this audit; step 3 sizes the inflation.

Protocol used everywhere: 20 contiguous time blocks of Train.csv, 5-fold
`StratifiedGroupKFold` grouped on the block. `time` is never a feature.

| macro-F1 | all 52 | station 20 | process 32 |
|---|---|---|---|
| shuffled rows (leaky) | 0.788 | 0.810 | 0.271 |
| **blocked time CV (reported)** | **0.711 +/- 0.032** | 0.719 | 0.298 |
| published Train->Test | 0.989 | 0.982 | 1.000 |

## Results (blocked CV, measured)

- **Detection lives on the station bus, not in the raw phasors.** Macro-F1 by layer: goose 0.627,
  consistency 0.498, relay 0.253, process 0.298, station 0.719, all 0.711. Adding the phasors to the
  station layers changes nothing beyond noise (-0.007).
- **Per-class recall (all features):** Normal 0.87, Fault 0.94, Masquerade 0.93, **Injection 0.40, Replay 0.39**.
  The weak classes are also the rare ones (101 and 100 rows).
- **Attack coverage** (mean recall over the three attack classes): all 0.577, station 0.606 (not
  distinguishable from all), phasors alone 0.156.
- **Controls:** a model that sees only the clock scores 0.18 macro-F1, only the counters 0.24.
- **Feature budget (E3):** k* = 8 features reach the full model within 0.01 macro-F1 (84.6% less upstream
  traffic). Six features are selected in every fold and none is a phasor:
  `consistency, state_cb, any_relay, stDiff, recentChange, MU3Log`.
- **Operating point (E6):** the data resolves ~181 false alerts/day at best; 0.79 attack recall at
  <= 2,000 false alerts/day, 0.88 at <= 10,000/day.
- **Unseen attack class (E8, both detectors held to 5% false alerts):** recall on Masquerade falls from
  0.95 to 0.34 when it was never in training (gap +0.61); Injection 0.60 -> 0.53 (+0.08); Replay
  0.35 -> 0.20 (+0.15, and Replay is hard even when seen). Mean gap +0.28: a central model does not
  transfer to a new Masquerade-type attack, so sites need local or federated training. The phasors alone
  detect ~5% at that budget, seen or not.
- Capture: 477 s, 2.29 messages/s, median gap 371 ms, 4 MUs per row.

## Step 5 - the designed grid (synthetic 40-substation region)

Built from the measured inputs above (`outputs/step5_inputs.json`; every input is tagged measured /
FALLBACK / STALE, and `--require-measured` refuses anything that is not fresh).

- Network: 47 lines, N-1 secure, +30.1% over the radial MST.
- Observability: 12 PMUs (30% of substations).
- Hierarchy: 6 zone hubs -> 4 levels (bay sentinel > station correlator > zone hub > control centre);
  the zone count is forced by the 25% blast-radius cap.
- **At 2.3 rows/s neither bandwidth nor compute forces a tier.** Raw centralised fusion peaks at
  0.06% of a link and every node needs < 0.01 cores. What tiering buys is blast radius (one fusion node
  serves <= 25% of bays instead of 100%) and survival of a single-node loss.
- **PMU placement buys no detection** on this data (E1 phasor gain is within noise); it is a physical
  observability decision only.

## Step 6 - the three detector tiers (measured)

Out-of-fold attack score (P(Masquerade)+P(Injection)+P(Replay)), one alert threshold per tier set so that
**1% of non-attack messages (Normal + Fault) are flagged**, attack coverage = mean recall of the three attack
classes at that threshold, 95% intervals from a bootstrap over the 20 time blocks (threshold re-derived in
every replicate).

| tier | features | false alerts/day per substation | attack coverage [95%] | Masq / Inj / Replay |
|---|---|---|---|---|
| sentinel (E3 stable set) | 6 | 543 | 0.538 [0.501, 0.693] | 0.96 / 0.49 / 0.17 |
| station (no phasors) | 20 | 725 | 0.524 [0.452, 0.621] | 0.85 / 0.50 / 0.23 |
| fused (everything) | 52 | 725 | 0.558 [0.518, 0.648] | 0.91 / 0.54 / 0.22 |

**The three tiers are not distinguishable at this budget** (paired: station - sentinel -0.037 [-0.178, +0.050];
fused - station +0.035 [-0.024, +0.111]). So any difference between grids cannot come from which detector runs
where; it has to come from which substations are still watched after something fails.

## Step 7 - default vs optimized grid (grids simulated, detector recalls measured)

The Kaggle data has no topology, so the grids are synthetic layouts (20/40/80 substations x 30 seeds) and the
7 standard test grids in `dataset/grids/`. Each failure scenario is a graph traversal; the measured recall of
the tier a substation still has says how much attack it catches.

| | lines | sensors | fusion | detection |
|---|---|---|---|---|
| **naive default** | radial MST | PMU in every substation | 1 control centre, largest site | everything streams to it |
| **budget-matched default** | MST + shortest extra corridors, same total spend | PMU everywhere | control centre + standby | centralised |
| **optimized** | N-1 augmentation (no bridge, no cut vertex) | minimum dominating set + zero-injection | k-medoid zones, 4 levels | tiered, every station detects locally |
| *control* | untargeted lines, same spend | minimum set | zones | tiered (the optimized architecture) |

Scenarios: every single substation, a 50 km regional outage around every site, 100 random double losses, and an
adversary's 40 best double losses among the fusion points and the 10 most critical substations.
"Delivered" coverage counts an alert only if it can still reach a hub or control centre (strict reading);
"local" coverage (secondary) counts it wherever it fires.

**Rule, fixed before looking at results.** A metric is *better/worse* only if the whole 95% interval of
(optimized - default) is beyond its tolerance, *tie* if it is inside, otherwise *inconclusive*.
Objectives (bridges, zone share, routes: what the algorithms minimise) show the algorithm worked; only
**outcomes** (attack detection that survives a failure, never optimised) can establish "better".
"OPTIMIZED BETTER" needs >= 1 outcome win, 0 security losses, 0 performance losses ("narrow" unless most
outcomes are wins); otherwise "NOT ESTABLISHED". Price is reported, never netted against security.

Result over 90 synthetic layouts (means; every outcome below is a clear win unless marked):

| | naive | budget-matched | control | **optimized** |
|---|---|---|---|---|
| coverage, nothing failed | 0.571 | 0.571 | 0.571 | 0.571 (tie) |
| delivered, one substation lost | 0.455 | 0.540 | 0.564 | 0.570 |
| delivered, 50 km regional outage | 0.394 | 0.493 | 0.552 | 0.562 |
| delivered, random double loss | 0.368 | 0.509 | 0.555 | 0.565 |
| delivered, adversary's double loss | 0.119 | 0.362 | 0.509 | 0.538 |
| delivered, worst scenario of all | 0.000 | 0.000 | 0.432 | 0.445 |
| single points of failure (bridges + cut vertices) | 79 | 48 | 48 | 0 |
| bays one compromised hub exposes | 100% | 100% | 22% | 22% |
| worst-latency bay -> first fused verdict | 447 ms | 446 ms | 452 ms | 452 ms (tie, +5 ms) |
| total cost (M kUSD, relative) | 2.70 | 3.58 | 3.58 | 3.62 (+37% vs naive; +1.4% vs budget-matched = tie) |
| PMUs | 47 | 47 | 13 | 13 |

- **Verdict:** optimized is better than both defaults on security (5 of 5 outcomes, 5 of 5 objectives), ties
  on performance, and costs +37% over the radial default and the same as the budget-matched one. Holds at
  20, 40 and 80 substations and on all 7 standard grids.
- **Which half does the work?** `step7_attribution.csv` crosses lines x deployment. For a single-substation
  loss the optimized lines and the tiered detection each recover about the same amount (+0.10 over naive);
  for the adversary's double loss and the worst scenario, tiered detection (local detectors, zones, no single
  control centre) does nearly all of it (adversary 0.12 -> 0.49 from tiering alone, 0.12 -> 0.40 from lines
  alone; worst case 0 -> 0.41 vs 0 -> 0). Graph-designed lines remove every single point of failure and add
  redundant routes, but against untargeted lines at the same spend (the control) they move coverage by only
  +0.7 to +2.9 points - only the adversary scenario is a clear win, so that verdict is **narrow**.
- **It can say no.** Optimized vs itself: all ties, not established. Roles swapped: not established. Under
  **5x slower substation hardware** the tiered design's three serial compute stages cost +39 ms against a 10 ms
  tolerance and the verdict flips to **not established** (latency stays far inside the 1 s budget; the
  pre-declared rule is what flips). If a PMU's inferred neighbours get no fused recall, intact coverage
  drops 2.6 points (interval includes 0, so inconclusive) and the lead over budget-matched narrows.
- Every other sensitivity (outage radius 25/100 km, line cost 0.5x/2x, fibre 4x, detector recall at its lower
  95% bound) keeps the verdict. `step7_sensitivity.csv`.
- **Standard grids (IEEE 14-300, Illinois 200):** optimized beats the as-built corridors on delivered coverage
  and single points of failure on every case. **Do not read the cost column against as-built**: the optimized
  lines are not checked against loads or ratings and the coordinates are schematic.

## Step 8 - forward test (train early, test late)

Train.csv and Test.csv are one recording, so Test.csv is not a test set. The honest hold-out is time: fit on
blocks 0-13, drop block 14, score blocks 15-19 once (330 messages). Alert thresholds come from the training
part's out-of-fold scores only.

| macro-F1 | blocked CV (reported) | forward hold-out |
|---|---|---|
| all 52 | 0.711 | 0.551 |
| station 20 | 0.719 | 0.573 |
| process 32 | 0.298 | 0.262 |

- **Detectors, attack coverage / false alerts on the test part** (target 1% of 122 negatives = about 1):
  sentinel 0.536 / 2, station 0.375 / 1, fused 0.524 / 3. One split cannot rank the tiers; the ordering differs from step 6.
- **Where it fails:** 53 of 148 Masquerade messages are called Injection; Replay recall 0.31 (32 messages);
  Fault has 8 test messages and is not a measurement. `step8_forward_confusion_all.csv`.
- **Reading it:** expect about 0.15 lower macro-F1 on a later capture than blocked CV suggests. Blocked CV
  remains the better way to compare views; the forward test is the better guess at deployment.

## Run (Windows)

Full guide, VS Code setup and run tracking: **RUNNING.md**.

```powershell
py -m venv .venv ; .\.venv\Scripts\Activate.ps1 ; pip install -r requirements.txt     # once
python run_pipeline.py --smoke     # ~1 min: does everything run? numbers NOT reportable
python run_pipeline.py             # ~5-8 min: steps 1-8 + 152 unit tests; the run is recorded in runs/
python runs.py                     # list runs;  runs.py show last | diff prev last | watch
```

Single steps (what `run_pipeline.py` calls):

```powershell
python step1_profile.py                    # profile + Train/Test audit
python step2_views.py                      # layer membership
python step3_split_and_baseline.py         # protocols compared
python step4_run_experiments.py            # E1 E3 E4 E6 E8  (--only e4, --skip e8)
python step5_design_grid.py --require-measured
python step6_train_detectors.py            # 3 tier detectors + recall error bars (-> models/)
python step7_compare_grids.py --require-measured --sensitivity   # default vs optimized
python step8_forward_test.py               # train early, test late
python -m unittest discover -s tests -v    # 152 tests
```

E4 latencies depend on the CPU: a full run on your machine re-measures them.

## Limits

- 1,094 rows from one 8-minute capture of one testbed; Injection and Replay have ~100 rows each,
  so their recalls carry +/-0.12-0.16 fold SD.
- Step 5 costs, line costs, latencies and the blast-radius cap are assumptions (`config.GRID_ASSUMPTIONS`).
- N-1 is topological; no power-flow check. The data cannot say whether detection holds on buses
  observed only by inference.
- Step 7 grids are simulated. Detector recall is per message from one testbed substation: cross-substation
  fusion gain is not credited (it would favour the tiered design), and the capture has no attack episodes, so
  there is no time-to-first-alert. The adversary is a scenario set, not a threat model. The default builders
  are one reasonable reading of "not optimized"; the budget-matched default and the control exist so the
  result is not an artefact of a weak baseline.
