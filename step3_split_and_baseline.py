"""
STEP 3 - evaluation protocols: how much does the way you split change the score?

Four ways to score the SAME model on the SAME five-class problem:

  shuffled rows CV     random rows of Train.csv      near-duplicate neighbours on both sides
  blocked time CV      whole time blocks of Train    THE honest number; every later step uses it
  published split      fit Train.csv, score Test.csv Test is the same recording as Train
  reverse              fit Test.csv, score Train.csv Test's timing columns were featurised differently

The differences between rows ARE the finding: they size how much of an apparently great
result is memorisation. Reported for three views so the effect is visible per layer.

Run:  python step3_split_and_baseline.py
"""
import json

import numpy as np
import pandas as pd

from src import config as C
from src.data import load_published
from src.metrics import evaluate
from src.models import EncodedModel, backend_name
from src.modalities import build_views
from src.splits import SPLITTERS

VIEWS = ("all", "station", "process")


def cv(df, cols, splitter):
    res = []
    for tr, te in SPLITTERS[splitter](df):
        m = EncodedModel().fit(df.iloc[tr][cols], df.iloc[tr][C.TARGET])
        res.append(evaluate(df.iloc[te][C.TARGET], m.predict(df.iloc[te][cols]),
                            m.predict_proba(df.iloc[te][cols])))
    return res


def fit_score(fit_df, score_df, cols):
    m = EncodedModel().fit(fit_df[cols], fit_df[C.TARGET])
    return evaluate(score_df[C.TARGET], m.predict(score_df[cols]), m.predict_proba(score_df[cols]))


def main():
    print("=" * 78)
    print(f"STEP 3 - EVALUATION PROTOCOLS   (backend: {backend_name()})")
    print("=" * 78)
    train, test, feature_cols = load_published()
    views = build_views(feature_cols)

    base = evaluate(train[C.TARGET], np.full(len(train), "Normal"))
    print(f"\n[0] constant 'Normal' predictor: macro-F1 {base['macro_f1']:.4f} "
          f"(accuracy {(train[C.TARGET] == 'Normal').mean():.4f})")

    table, per_class = {}, {}
    for v in VIEWS:
        cols = views[v]
        folds_naive, folds_blocked = cv(train, cols, "naive_rows"), cv(train, cols, "blocked")
        f = lambda rs: float(np.mean([r["macro_f1"] for r in rs]))
        pub = fit_score(train, test, cols)
        rev = fit_score(test, train, cols)
        table[v] = {"shuffled_rows_cv": f(folds_naive), "blocked_time_cv": f(folds_blocked),
                    "blocked_sd": float(np.std([r["macro_f1"] for r in folds_blocked])),
                    "published_train_to_test": pub["macro_f1"], "reverse_test_to_train": rev["macro_f1"]}
        per_class[v] = {"blocked": {c: float(np.mean([r[f"recall__{c}"] for r in folds_blocked]))
                                    for c in C.CLASS_ORDER},
                        "published": {c: pub[f"recall__{c}"] for c in C.CLASS_ORDER}}

    res = pd.DataFrame(table).T
    print("\n[1] MACRO-F1 BY PROTOCOL")
    print(res.to_string(float_format=lambda x: f"{x:.4f}"))
    res["memorisation_gap"] = res["shuffled_rows_cv"] - res["blocked_time_cv"]
    res["published_inflation"] = res["published_train_to_test"] - res["blocked_time_cv"]
    print("\n[2] GAPS (how much each shortcut flatters the model)")
    print(res[["memorisation_gap", "published_inflation"]].to_string(float_format=lambda x: f"{x:+.4f}"))

    print("\n[3] PER-CLASS RECALL, all features: honest vs published")
    pc = pd.DataFrame({"blocked_time_cv": per_class["all"]["blocked"],
                       "published_split": per_class["all"]["published"]}).reindex(C.CLASS_ORDER)
    print(pc.to_string(float_format=lambda x: f"{x:.3f}"))

    # how much of Test is literally in Train, layer by layer
    print("\n[4] TEST MESSAGES WHOSE COLUMNS APPEAR VERBATIM IN Train.csv")
    seen = {}
    for v in ("process", "station", "all"):
        key = lambda d: d[views[v]].round(6).astype(str).agg("|".join, axis=1)
        seen[v] = float(key(test).isin(set(key(train))).mean())
        print(f"  {v:<8} {seen[v]:.1%}")
    print("  -> a model scored on Test sees messages it was fitted on; the MU-angle columns are")
    print("     almost entirely identical, which is why a process-only model looks perfect there.")

    print("\n  HEADLINE for the report: blocked time CV, all features: macro-F1 "
          f"{res.loc['all', 'blocked_time_cv']:.4f} +/- {res.loc['all', 'blocked_sd']:.4f}.")
    print(f"  The published split would have reported {res.loc['all', 'published_train_to_test']:.4f}.")

    out = {"table": json.loads(res.to_json(orient="index")), "per_class": per_class,
           "test_rows_seen_in_train": seen, "constant_normal_macro_f1": base["macro_f1"],
           "backend": backend_name()}
    with open(C.OUTPUT_DIR / "step3_baselines.json", "w") as fh:
        json.dump(out, fh, indent=2)
    res.to_csv(C.OUTPUT_DIR / "step3_protocols.csv")


if __name__ == "__main__":
    main()
