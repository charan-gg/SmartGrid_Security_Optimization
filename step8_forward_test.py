"""
STEP 8 - forward test: train on the early part of the capture, test once on the late part.

The held-out test this data allows (Test.csv is the same recording as Train.csv, so it cannot be
one). See src/forward_test.py for the split.

    python step8_forward_test.py
"""
import json

import pandas as pd

from src import config as C
from src import detectors as DT
from src import forward_test as FT
from src.data import load_dataset
from src.models import backend_name


def main():
    df, _, views = load_dataset()
    stable = DT.stable_features()
    tiers, classes, cms, info = FT.run(df, views, stable)

    print(f"backend {backend_name()} | train blocks {info['train_blocks'][0]}-{info['train_blocks'][-1]} "
          f"({info['n_train']:,} msgs) | purged block | test blocks {info['test_blocks'][0]}-{info['test_blocks'][-1]} "
          f"({info['n_test']:,} msgs)")
    print(f"test class counts: {info['test_class_counts']}\n")

    print(f"ATTACK DETECTORS at a {C.OPERATING_FPR:.0%} false-alert budget (threshold fixed on the train part only)")
    print("  tier       thr    false alerts on test    coverage train-OOF -> test     Masq          Inj          Replay")
    for _, t in tiers.iterrows():
        print(f"  {t['tier']:<9} {t['threshold']:.3f}   {t['test_false_alerts']:>3}/{t['test_negatives']:<3} = {t['test_fpr']:.3f}"
              f"        {t['train_oof_coverage']:.3f} -> {t['test_coverage']:.3f}"
              f"        {t['recall_Masquerade']:.2f} (n={t['n_Masquerade']})  {t['recall_Injection']:.2f} (n={t['n_Injection']})"
              f"  {t['recall_Replay']:.2f} (n={t['n_Replay']})")

    print("\n5-CLASS HOLD-OUT (macro-F1; blocked-CV figure from step 4 in brackets)")
    try:
        cv = pd.read_csv(C.OUTPUT_DIR / "e1_views_blocked.csv").set_index("view")["macro_f1"]
    except (OSError, KeyError):
        cv = {}
    for _, c in classes.iterrows():
        ref = f"   [blocked CV {cv[c['view']]:.3f}]" if c["view"] in cv else ""
        print(f"  {c['view']:<8} macro-F1 {c['macro_f1']:.3f}   recall: " + "  ".join(
            f"{k} {c[f'recall__{k}']:.2f}" for k in C.CLASS_ORDER) + ref)
    print("\n  confusion (rows = truth, cols = predicted), view 'all':")
    print(cms["all"].to_string())

    tiers.to_csv(C.OUTPUT_DIR / "step8_forward_tiers.csv", index=False)
    classes.to_csv(C.OUTPUT_DIR / "step8_forward_classes.csv", index=False)
    cms["all"].to_csv(C.OUTPUT_DIR / "step8_forward_confusion_all.csv")
    (C.OUTPUT_DIR / "step8_forward_test.json").write_text(json.dumps(info, indent=2, default=int))
    print("\n  ONE split, ~1,000 messages: read the counts, not just the rates. A different cut of the")
    print("  capture gives different numbers; the blocked CV in step 4 averages over all of them.")


if __name__ == "__main__":
    main()
