"""
Modality views: which of the 128 feature columns belong to which layer of the
control hierarchy.

This file is where the core thesis becomes code. The claim is that the four
data sources sit at different Purdue levels and therefore see different attacks
at different times. Slicing the columns is how that claim gets tested.

    PMU synchrophasor measurements  -> L1  physical
    Protective relay logs           -> L2  control layer
    Control panel logs              -> L2/L3 operator
    Snort IDS alerts                -> L3.5 network / cyber
"""
import re

# --------------------------------------------------------------------------
# Column naming in this dataset
# --------------------------------------------------------------------------
# Each of the 4 PMUs contributes 29 columns (4 x 29 = 116):
#   R{n}-PA{k}:VH   phase angle, voltage        (k = 1..3  three phases)
#   R{n}-PM{k}:V    phase magnitude, voltage
#   R{n}-PA{k}:IH   phase angle, current        (k = 4..6)
#   R{n}-PM{k}:I    phase magnitude, current
#   ... repeated for the second three-phase set (k = 7..12)
#   R{n}:F          frequency
#   R{n}:DF         frequency delta (df/dt)
#   R{n}-PA:Z       apparent impedance  <- the Inf-carrying columns
#   R{n}-PA:ZH      apparent impedance angle
#   R{n}:S          status flag
_PMU_MEASUREMENT = re.compile(r"^R\d+-P[AM]\d+:(VH|V|IH|I)$")
_PMU_DERIVED     = re.compile(r"^R\d+(:F|:DF|-PA:Z|-PA:ZH)$")
_PMU_STATUS      = re.compile(r"^R\d+:S$")

_RELAY_LOG   = re.compile(r"^relay\d+_log$")
_SNORT_LOG   = re.compile(r"^snort_log\d+$")
_CONTROL_LOG = re.compile(r"^control_panel_log\d+$")

# R{n}:S is the relay status flag. The published schema counts it among the
# PMU's 29 columns, so it is PMU by default. But it is arguably a control-layer
# signal, and if the report wants to argue that, flipping this one switch moves
# it and every downstream experiment follows. Record whichever you choose.
STATUS_FLAG_IS_CONTROL_LAYER = False


def classify_column(col: str) -> str:
    """Map a single column name to its modality. Returns one of:
    'pmu', 'relay', 'snort', 'control', 'meta', 'unknown'."""
    if _PMU_MEASUREMENT.match(col) or _PMU_DERIVED.match(col):
        return "pmu"
    if _PMU_STATUS.match(col):
        return "relay" if STATUS_FLAG_IS_CONTROL_LAYER else "pmu"
    if _RELAY_LOG.match(col):
        return "relay"
    if _SNORT_LOG.match(col):
        return "snort"
    if _CONTROL_LOG.match(col):
        return "control"
    # engineered companions inherit the modality of their parent column
    if col.endswith("__was_inf"):
        return classify_column(col[: -len("__was_inf")])
    return "unknown"


def modality_map(columns) -> dict:
    """{column_name: modality} for every feature column given."""
    return {c: classify_column(c) for c in columns}


def columns_for(columns, modality: str) -> list:
    """All feature columns belonging to one modality, order preserved."""
    return [c for c in columns if classify_column(c) == modality]


# The views E1 compares. Single modalities test "can this layer alone see the
# attack"; 'all' is the fusion point. Pairs are optional extras for the report.
SINGLE_MODALITIES = ["pmu", "relay", "snort", "control"]


def build_views(feature_columns) -> dict:
    """{view_name: [columns]} for every view E1/E5 needs."""
    views = {m: columns_for(feature_columns, m) for m in SINGLE_MODALITIES}
    views["all"] = list(feature_columns)
    # 'cyber_only' = everything that is NOT the physical layer. This is the
    # single most useful extra view: it answers "could a purely IT-side
    # monitoring deployment, with no PMU access at all, do the job?"
    views["cyber_only"] = (
        views["relay"] + views["snort"] + views["control"]
    )
    return views
