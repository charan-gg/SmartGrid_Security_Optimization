"""
E8 - UNSEEN-ATTACK GENERALISATION.
Architectural parameter it fixes: central model shipping vs per-site / federated
training, and who owns model updates.

The question a utility asks: if the model is trained where attack type A was observed
and then deployed where a new variant B shows up, does it still raise the alarm?
Simulated by leave-one-attack-class-out. For each held-out attack class the detector
(attack vs not-attack) is trained WITHOUT any message of that class, inside the same
blocked-time folds, and then asked to flag it. Compare with the same detector trained
WITH the class.

  seen recall     attack class was in the training data
  unseen recall   attack class was never shown to the model
  gap             seen - unseen: the transferability cost of that attack type

Both detectors are compared at the SAME false-alert budget (MATCHED_FPR of the non-attack
messages, threshold taken from the pooled out-of-fold scores). Comparing at each model's own
0.5 cut would mix the transfer cost with a shift in operating point: a model that has never seen
the attack also raises fewer false alarms. The 0.5-cut numbers stay in the CSV as *_at_0.5.

A small gap says one central model can be shipped to every site. A large gap says each
site must train on its own traffic, which is a different system to build.
"""
import numpy as np
import pandas as pd

from .. import config as C
from ..models import EncodedModel
from ..splits import SPLITTERS

BINARY = ["Other", "Attack"]
MATCHED_FPR = 0.05          # false-alert budget at which seen and unseen detectors are compared


def _label(d):
    return np.where(d[C.FAMILY_COL] == "Attack", "Attack", "Other")


def run(df, views, view_names=("all", "station", "process")):
    rows = []
    for view in [v for v in view_names if views.get(v)]:
        cols = views[view]
        for held in C.ATTACK_CLASSES:
            sc = {"seen": [], "unseen": []}        # out-of-fold P(attack), pooled over folds
            is_held_all, is_other_all = [], []
            for tr, te in SPLITTERS["blocked"](df):
                train, test = df.iloc[tr], df.iloc[te]
                is_held_all.append((test[C.TARGET] == held).to_numpy())
                is_other_all.append((test[C.FAMILY_COL] != "Attack").to_numpy())
                for mode in ("seen", "unseen"):
                    sub = train if mode == "seen" else train[train[C.TARGET] != held]
                    m = EncodedModel(BINARY).fit(sub[cols], _label(sub))
                    sc[mode].append(m.predict_proba(test[cols])[:, BINARY.index("Attack")])
            held_, other_ = np.concatenate(is_held_all), np.concatenate(is_other_all)
            row = dict(view=view, held_out=held, n_held_out=int(held_.sum()))
            for mode in ("seen", "unseen"):
                score = np.concatenate(sc[mode])
                # (a) the detector's own 0.5 cut, as deployed
                row[f"recall_{mode}_at_0.5"] = float((score[held_] > 0.5).mean())
                row[f"fpr_{mode}_at_0.5"] = float((score[other_] > 0.5).mean())
                # (b) the same false-alert budget for both detectors, on the same negatives
                thr = np.quantile(score[other_], 1.0 - MATCHED_FPR)
                row[f"recall_{mode}"] = float((score[held_] > thr).mean())
                row[f"fpr_{mode}"] = float((score[other_] > thr).mean())
            row["gap"] = row["recall_seen"] - row["recall_unseen"]
            row["gap_at_0.5"] = row["recall_seen_at_0.5"] - row["recall_unseen_at_0.5"]
            rows.append(row)
            print(f"  {view:<8} {held:<11} @FPR<={MATCHED_FPR:.0%}: seen={row['recall_seen']:.3f}  "
                  f"unseen={row['recall_unseen']:.3f}  gap={row['gap']:+.3f}   "
                  f"(own 0.5 cut: {row['recall_seen_at_0.5']:.3f}/{row['recall_unseen_at_0.5']:.3f}, "
                  f"fpr {row['fpr_seen_at_0.5']:.3f}/{row['fpr_unseen_at_0.5']:.3f})")
    res = pd.DataFrame(rows)
    res.to_csv(C.OUTPUT_DIR / "e8_unseen_attack.csv", index=False)

    fused = res[res["view"] == "all"]
    gap = float(fused["gap"].mean())
    worst = fused.loc[fused["gap"].idxmax()]
    print(f"\n  mean transferability gap, all features: {gap:+.3f}  (worst: {worst['held_out']} "
          f"{worst['gap']:+.3f})")
    if gap > 0.10:
        print("  -> LARGE. A model trained without an attack type misses it. Sites need to train on")
        print("     their own traffic (or federate), and Tier 3 should own model updates.")
    else:
        print("  -> SMALL. One centrally trained model transfers to unseen attack types.")
    return res
