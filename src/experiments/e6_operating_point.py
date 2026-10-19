"""
E6 - OPERATING POINT AND ALARM BUDGET.
Architectural parameter it fixes: the alert threshold each tier ships with.

A detector is deployed at the threshold an operator will tolerate, expressed in
false alerts per 24 hours, not at the threshold that maximised F1 on a laptop. This
converts the model's probabilities into that number and reports what attack recall
each alarm budget buys.

Score = P(Masquerade) + P(Injection) + P(Replay) from the 5-class model, taken out-of-fold
over the whole capture (every message scored by a model that never saw its time block).
"Negative" = Normal + Fault: a Fault is a natural event, so flagging it as an attack is a
false alert. The negative message rate is measured from the capture.
"""
import numpy as np
import pandas as pd

from .. import config as C
from ..data import rates
from ..metrics import false_alerts_per_day
from ..models import EncodedModel
from ..splits import SPLITTERS

ALARM_BUDGETS = [100.0, 500.0, 2000.0, 10000.0]       # false alerts per 24 h


def oof_attack_score(df, cols):
    score = np.zeros(len(df))
    att = [C.CLASS_ORDER.index(c) for c in C.ATTACK_CLASSES]
    for tr, te in SPLITTERS["blocked"](df):
        m = EncodedModel().fit(df.iloc[tr][cols], df.iloc[tr][C.TARGET])
        score[te] = m.predict_proba(df.iloc[te][cols])[:, att].sum(axis=1)
    return score


def run(df, views, view_name="all"):
    cols = views[view_name]
    r = rates(df)
    score = oof_attack_score(df, cols)
    is_attack = (df[C.FAMILY_COL] == "Attack").to_numpy()
    n_pos, n_neg = int(is_attack.sum()), int((~is_attack).sum())

    rows = []
    for thr in np.unique(np.round(np.linspace(0.01, 0.999, 250), 4)):
        pred = score >= thr
        tp, fp = int((pred & is_attack).sum()), int((pred & ~is_attack).sum())
        fpr = fp / n_neg
        rows.append({"threshold": float(thr), "recall_attack": tp / n_pos,
                     "precision_attack": tp / (tp + fp) if (tp + fp) else 1.0, "fpr": fpr,
                     "false_alerts_per_day": false_alerts_per_day(fpr, r["negative_rate_hz"])})
    curve = pd.DataFrame(rows)
    curve.to_csv(C.OUTPUT_DIR / "e6_operating_curve.csv", index=False)

    per_day = r["negative_rate_hz"] * 86400
    floor = false_alerts_per_day(1.0 / n_neg, r["negative_rate_hz"])
    print(f"  out-of-fold scores for {len(df):,} messages (attack {n_pos:,} / other {n_neg:,}), view '{view_name}'")
    print(f"  measured non-attack rate {r['negative_rate_hz']:.2f} msg/s -> {per_day:,.0f} per day per unit")
    print(f"\n  RESOLUTION FLOOR: {n_neg:,} negatives => one false positive = {floor:,.0f} alerts/day.")
    print("  A budget below that is 'no false positive observed', an upper bound set by the size")
    print("  of the capture, not a measured rate. Report E6 as a CURVE plus this floor.")

    print("\n  ATTACK RECALL PURCHASABLE AT EACH ALARM BUDGET")
    picks = []
    for budget in ALARM_BUDGETS:
        ok = curve[curve["false_alerts_per_day"] <= budget]
        if ok.empty:
            print(f"    <= {budget:>7,.0f} alerts/day : unreachable at any threshold")
            continue
        best = ok.loc[ok["recall_attack"].idxmax()]
        below = budget < floor
        picks.append({"budget": budget, "below_resolution_floor": below, **best.to_dict()})
        print(f"    <= {budget:>7,.0f} alerts/day : threshold={best['threshold']:.3f}  "
              f"recall={best['recall_attack']:.3f}  precision={best['precision_attack']:.3f}"
              f"{'   [below floor: upper bound only]' if below else ''}")
    pd.DataFrame(picks).to_csv(C.OUTPUT_DIR / "e6_operating_points.csv", index=False)
    print("\n  -> the chosen threshold is a CONFIGURATION CONSTANT of the tier spec. A per-message")
    print("     alarm also needs a hold-down rule: one attack spans many messages.")
    return curve
