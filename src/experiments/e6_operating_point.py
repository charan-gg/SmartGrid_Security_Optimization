"""
E6 - OPERATING POINT AND ALARM BUDGET.
Architectural parameter it fixes: the alert threshold each tier ships with.

A detector is not deployed at "the threshold that maximised F1 on my laptop".
It is deployed at the threshold an operator will tolerate, expressed in alerts
per 24 hours. This experiment converts the model's probability output into that
number and reports what recall you can buy at each alarm budget.

Watch the base-rate assumption. This dataset is ~71% attack rows; a real
substation is overwhelmingly quiet. config.BENIGN_FRACTION_REAL carries that
assumption explicitly and every alerts/day figure scales with it, which is why
the result is reported as a CURVE over thresholds rather than a single number.
"""
import numpy as np
import pandas as pd

from .. import config as C
from ..metrics import false_alerts_per_day
from ..models import EncodedModel
from ..splits import by_event_single

# Alarm budgets a control room might actually accept, alerts per 24h.
ALARM_BUDGETS = [0.1, 1.0, 5.0, 25.0, 100.0]


def run(df, views, view_name="all"):
    tr, te = by_event_single(df)
    train_df, test_df = df.iloc[tr], df.iloc[te]
    cols = views[view_name]

    model = EncodedModel(len(C.CLASS_ORDER)).fit(train_df[cols], train_df[C.TARGET])
    proba = model.predict_proba(test_df[cols])
    attack_col = C.CLASS_ORDER.index(C.ATTACK_CLASS)
    score = proba[:, attack_col]

    y_true = (test_df[C.TARGET] == C.ATTACK_CLASS).to_numpy()
    n_pos, n_neg = int(y_true.sum()), int((~y_true).sum())

    rows = []
    for thr in np.unique(np.round(np.linspace(0.01, 0.999, 250), 4)):
        pred = score >= thr
        tp = int((pred & y_true).sum())
        fp = int((pred & ~y_true).sum())
        fpr = fp / n_neg if n_neg else 0.0
        rows.append({
            "threshold": float(thr),
            "recall_attack": tp / n_pos if n_pos else 0.0,
            "precision_attack": tp / (tp + fp) if (tp + fp) else 1.0,
            "fpr": fpr,
            "false_alerts_per_day": false_alerts_per_day(fpr),
        })
    curve = pd.DataFrame(rows)
    curve.to_csv(C.OUTPUT_DIR / "e6_operating_curve.csv", index=False)

    print(f"  test rows: {len(y_true):,}  (attack {n_pos:,} / other {n_neg:,})")
    print(f"  assumed benign duty cycle: {C.BENIGN_FRACTION_REAL} at "
          f"{C.SAMPLE_RATE_HZ} Hz -> "
          f"{C.SAMPLE_RATE_HZ*86400*C.BENIGN_FRACTION_REAL:,.0f} benign samples/day")
    # ---------------------------------------------- measurement resolution
    # The finest non-zero false-positive rate this test set can express is one
    # single false positive. Below that, the only measurable value is exactly
    # zero, and a threshold reported there is an artefact of test-set size, not
    # evidence that the detector is that quiet. State the floor explicitly.
    floor_alerts = false_alerts_per_day(1.0 / n_neg) if n_neg else float("inf")
    print(f"\n  RESOLUTION FLOOR: {n_neg:,} negative test samples means the")
    print(f"  smallest measurable non-zero rate is 1 FP = "
          f"{floor_alerts:,.0f} alerts/day.")
    print("  Any budget below that can only be 'zero false positives observed',")
    print("  which is an upper bound set by test-set size. To measure a budget")
    print("  of N alerts/day you need roughly "
          f"{C.SAMPLE_RATE_HZ*86400*C.BENIGN_FRACTION_REAL:,.0f}/N negative samples.")
    print("  -> report E6 as a CURVE plus this floor; do not quote a single")
    print("     threshold as if the low-alarm region were measured.")

    print("\n  RECALL PURCHASABLE AT EACH ALARM BUDGET")
    picks = []
    for budget in ALARM_BUDGETS:
        below_floor = budget < floor_alerts
        ok = curve[curve["false_alerts_per_day"] <= budget]
        if ok.empty:
            print(f"    <= {budget:>6.1f} alerts/day : unreachable at any threshold")
            continue
        best = ok.loc[ok["recall_attack"].idxmax()]
        picks.append({"budget": budget, "below_resolution_floor": below_floor,
                      **best.to_dict()})
        note = "   [BELOW RESOLUTION FLOOR - not measured, upper bound only]" if below_floor else ""
        print(f"    <= {budget:>6.1f} alerts/day : threshold={best['threshold']:.3f}  "
              f"recall={best['recall_attack']:.4f}  "
              f"precision={best['precision_attack']:.4f}{note}")
    pd.DataFrame(picks).to_csv(C.OUTPUT_DIR / "e6_operating_points.csv", index=False)

    print("\n  -> the threshold chosen here is a CONFIGURATION CONSTANT of the")
    print("     architecture. Quote it in the tier spec alongside its budget.")
    print("     A per-sample alarm also needs a hold-down/debounce rule, since")
    print("     one event spans many samples: state that rule in the report.")
    return curve
