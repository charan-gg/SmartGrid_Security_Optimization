"""
E2 - OBSERVABILITY AND DETECTION LEAD TIME.  The headline experiment.
Architectural parameter it fixes: which levels to instrument, and each tier's
latency budget.

No synthetic splicing is needed, which was the whole reason for the dataset
pivot. Every event instance is a contiguous, time-ordered block of samples, so
we can ask, per modality: how far into the event does this layer first produce
a confident, correct verdict?

Definition (config.E2_CONFIDENCE, E2_SUSTAIN_ROWS): the verdict time is the
first sample index t at which the model has predicted the true class with
probability >= confidence for `sustain` consecutive samples. The sustain
requirement is what stops one lucky frame being reported as a detection.

CAVEAT to state in the report: t is measured from the START OF THE CAPTURED
BLOCK, not from the physical instant of attack onset, which this dataset does
not label. Absolute times are therefore not meaningful. The DIFFERENCE between
two modalities on the same event is meaningful, because both are measured from
the same origin - and that difference is the lead time we are claiming.
"""
import numpy as np
import pandas as pd

from .. import config as C
from ..models import EncodedModel
from ..splits import by_event_single


def _verdict_index(proba_true_class, confident, sustain):
    """First index where prob >= confident for `sustain` consecutive samples."""
    ok = proba_true_class >= confident
    if len(ok) < sustain:
        return None
    # rolling AND over a window of length `sustain`
    run = np.convolve(ok.astype(int), np.ones(sustain, dtype=int), mode="valid")
    hits = np.flatnonzero(run == sustain)
    return int(hits[0]) if hits.size else None


def run(df, views):
    tr, te = by_event_single(df)
    train_df, test_df = df.iloc[tr], df.iloc[te]

    # one model per modality view, all trained on the same events
    models = {}
    for name, cols in views.items():
        if not cols:
            continue
        models[name] = EncodedModel(len(C.CLASS_ORDER)).fit(
            train_df[cols], train_df[C.TARGET])

    class_index = {c: i for i, c in enumerate(C.CLASS_ORDER)}
    records = []

    for event_id, block in test_df.groupby(C.GROUP_KEY, sort=False):
        block = block.sort_values("t_in_event")
        true_class = block[C.TARGET].iloc[0]
        # an event is one scenario, so the label is constant within it
        assert block[C.TARGET].nunique() == 1, f"{event_id} has mixed labels"
        col_true = class_index[true_class]

        for name, model in models.items():
            proba = model.predict_proba(block[views[name]])[:, col_true]
            idx = _verdict_index(proba, C.E2_CONFIDENCE, C.E2_SUSTAIN_ROWS)
            records.append({
                "event_id": event_id,
                "scenario_id": block["scenario_id"].iloc[0],
                "true_class": true_class,
                "view": name,
                "event_len": len(block),
                "verdict_sample": idx,                    # None = never detected
                "verdict_ms": None if idx is None else 1000.0 * idx / C.SAMPLE_RATE_HZ,
                "detected": idx is not None,
                "mean_confidence": float(proba.mean()),
            })

    per_event = pd.DataFrame(records)
    per_event.to_csv(C.OUTPUT_DIR / "e2_per_event.csv", index=False)

    # ---------------------------------------------------------- summary table
    print("\n  DETECTION RATE AND TIME-TO-VERDICT, per modality")
    summary = (per_event
               .groupby("view")
               .agg(detection_rate=("detected", "mean"),
                    median_verdict_sample=("verdict_sample", "median"),
                    median_verdict_ms=("verdict_ms", "median"),
                    p90_verdict_ms=("verdict_ms", lambda s: s.quantile(0.90)))
               .sort_values("median_verdict_ms"))
    print(summary.to_string(float_format=lambda v: f"{v:8.3f}"))

    # ------------------------------------------------- the lead-time claim
    # Pairwise, on events where BOTH views eventually decide. Restricting to
    # jointly-detected events is what makes the comparison fair; otherwise a
    # view looks fast simply by never detecting the hard events.
    print("\n  PAIRWISE LEAD TIME (ms), positive = row view decides EARLIER")
    print("  (computed only over events BOTH views eventually decide; a blank")
    print("   cell means the two views never decided on the same event)")
    wide = per_event.pivot(index="event_id", columns="view", values="verdict_ms")
    names = [v for v in wide.columns if v != "all"]
    lead = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            if a == b:
                lead.loc[a, b] = 0.0          # selecting [a, a] would duplicate the column
                continue
            both = wide[[a, b]].dropna()
            # Restricting to jointly-detected events is deliberate: a modality
            # that simply never decides on the hard events would otherwise look
            # fast by omission.
            lead.loc[a, b] = float((both[b] - both[a]).median()) if len(both) else np.nan
    print(lead.to_string(float_format=lambda v: f"{v:8.2f}"))
    lead.to_csv(C.OUTPUT_DIR / "e2_lead_time_matrix.csv")

    # ------------------------------------------- per-class, which layer first
    print("\n  EARLIEST-DECIDING MODALITY PER CLASS (this is the E2 result)")
    for cls in C.CLASS_ORDER:
        sub = per_event[(per_event["true_class"] == cls) & per_event["detected"]]
        if sub.empty:
            print(f"    {cls:<10} never detected by any modality")
            continue
        med = sub.groupby("view")["verdict_ms"].median().drop(index="all", errors="ignore")
        if med.empty:
            continue
        first = med.idxmin()
        print(f"    {cls:<10} earliest = {first:<8} ({med.min():7.2f} ms)   "
              + "  ".join(f"{v}={m:.0f}ms" for v, m in med.items()))

    summary.to_csv(C.OUTPUT_DIR / "e2_summary.csv")
    print("\n  NOTE: ms figures assume "
          f"{C.SAMPLE_RATE_HZ} Hz (config.SAMPLE_RATE_HZ, marked CONFIRM).")
    return per_event, summary, lead
