"""
Views: which feature columns belong to which layer of the substation.

This file is where the architecture argument becomes code. A substation is
instrumented at different points and each sees different evidence:

    process    MU voltage/current angles and magnitudes (IEC 61850-9-2 process bus,
               the physical layer: what a PMU-class sensor sees)
    goose      GOOSE protocol state on the station bus: sequence/state numbers,
               their deltas, time since the previous message, circuit-breaker state
    consistency cross-MU checks computed at the station: per-MU consistency flags
               and the three-phase voltage/current sums (Kirchhoff residuals)
    relay      protection-relay log flags
    station    goose + consistency + relay: everything a station correlator has
               WITHOUT any raw phasors

E1 tests which of these layers carries the evidence for which class. The names
are hypotheses about the layers, not findings; E1 measures them.

The raw `time` column is never a feature (it is the capture clock, so a model could
learn WHEN attacks were launched instead of what they look like); data.py keeps it
as metadata only.
"""
import re

_PROCESS = re.compile(r"^(threePhase(V|C|Current)MU\d|MU\d(Voltage|Current)Angle[ABC]|IED\d_i[ABC])$")
_GOOSE = {"sqNum", "stnum", "state_cb", "sqDiff", "stDiff", "timeLastMsg", "recentChange"}
_CONSISTENCY = re.compile(r"^(MU\dCs|Cs|consistency|threePhase[VC]Sum)$")
_RELAY = re.compile(r"^(MU\dLog|any_relay)$")

SINGLE_MODALITIES = ["process", "goose", "consistency", "relay"]


def classify_column(col: str) -> str:
    """'process' | 'goose' | 'consistency' | 'relay' | 'unknown'."""
    if _PROCESS.match(col):
        return "process"
    if col in _GOOSE:
        return "goose"
    if _CONSISTENCY.match(col):
        return "consistency"
    if _RELAY.match(col):
        return "relay"
    return "unknown"


def modality_map(columns) -> dict:
    """{column_name: modality} for every feature column given."""
    return {c: classify_column(c) for c in columns}


def columns_for(columns, modality: str) -> list:
    """All feature columns belonging to one modality, order preserved."""
    return [c for c in columns if classify_column(c) == modality]


def n_units(columns) -> int:
    """Merging units one row spans: the distinct MU<n> prefixes in the column names."""
    found = {m.group(1) for c in columns for m in [re.search(r"MU(\d)", c)] if m}
    return len(found)


def build_views(feature_columns) -> dict:
    """{view_name: [columns]} for every view E1/E4/E8 need."""
    views = {m: columns_for(feature_columns, m) for m in SINGLE_MODALITIES}
    views["station"] = views["goose"] + views["consistency"] + views["relay"]
    views["all"] = list(feature_columns)
    return views
