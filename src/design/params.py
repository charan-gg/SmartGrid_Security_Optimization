"""
Inputs to grid synthesis: what Train.csv MEASURED, and what is merely ASSUMED.

`Measured` is read from outputs/*.csv|json so that re-running a step changes the
design automatically. If a file is missing the loader falls back to a placeholder and
says so in `provenance`; if a file is OLDER than dataset/Train.csv it is marked STALE
(it was produced from a previous dataset). A design number should never be quoted
without knowing which of the three it is: `provenance_problems()` lists the ones that
are not fresh measurements, and `step5_design_grid.py --require-measured` refuses them.

Sensor naming: the design code keeps the PMU vocabulary of power-system sensing. Here
a "PMU / MU sensor" is a process-bus merging-unit feed (the `process` view).

`Assumptions` is built from config.GRID_ASSUMPTIONS. It is a frozen dataclass so the
sensitivity sweep can derive variants with dataclasses.replace().
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path

from .. import config as C
from ..modalities import classify_column

# Placeholders used only if an output file is absent. They are NOT measurements.
FALLBACK = dict(
    ieds_per_unit=4, sample_rate_hz=2.3, n_features_full=52, n_features_process=32,
    k_stable=12, k_stable_process=0,
    edge_p50_ms=1.3, edge_p95_ms=1.8,  edge_batch_ms=0.02,
    station_p50_ms=5.0, station_p95_ms=5.5, station_batch_ms=0.06,
    wide_p50_ms=5.0, wide_p95_ms=5.5, wide_batch_ms=0.06,
    det_all=0.6, det_station=0.6, det_process=0.2,
)


@dataclass(frozen=True)
class Measured:
    sample_rate_hz: float       # messages/s one measured unit emits (step 1)
    bytes_per_feature: int
    ieds_per_unit: int          # merging units spanned by one feature row (step 1)
    n_features_full: int        # fused vector (E1 'all')
    n_features_process: int     # process-bus-only vector (E4 tier1_edge_process_only)
    k_stable: int               # E3: features selected in every fold
    k_stable_process: int       # ...of which are MU phasor columns
    edge_p50_ms: float          # E4 tier1_edge_kstar
    edge_p95_ms: float
    edge_batch_ms: float
    station_p50_ms: float       # E4 tier2_zone_fused
    station_p95_ms: float
    station_batch_ms: float     # per-sample cost when scoring in batches of 128
    wide_p50_ms: float          # E4 tier3_widearea_fused
    wide_p95_ms: float
    wide_batch_ms: float
    det_all: float              # E1: attack coverage with every layer (a unit that has MU sensing)
    det_station: float          # E1: ...with the station-bus layers only (no raw phasors)
    det_process: float          # E1: ...with the process-bus phasors only
    provenance: dict

    # ---- derived per-bay figures (the unit the grid is built from) --------
    @property
    def evidence_Bps_per_bay(self):
        """Tier-1 evidence vector, bytes/s per bay (E3 stable feature set)."""
        return self.k_stable / self.ieds_per_unit * self.bytes_per_feature * self.sample_rate_hz

    @property
    def raw_Bps_per_bay(self):
        """Full feature stream, bytes/s per bay (config B: centralised raw)."""
        return self.n_features_full / self.ieds_per_unit * self.bytes_per_feature * self.sample_rate_hz

    @property
    def process_Bps_per_bay(self):
        """Process-bus-only stream, bytes/s per bay (config A: physical-only centralised)."""
        return self.n_features_process / self.ieds_per_unit * self.bytes_per_feature * self.sample_rate_hz

    @property
    def sample_period_ms(self):
        return 1000.0 / self.sample_rate_hz

    def table(self):
        """[(name, value, provenance)] for printing."""
        rows = []
        for f in fields(self):
            if f.name != "provenance":
                rows.append((f.name, getattr(self, f.name), self.provenance.get(f.name, "config")))
        for name in ("evidence_Bps_per_bay", "raw_Bps_per_bay", "process_Bps_per_bay"):
            rows.append((name, getattr(self, name), "derived"))
        return rows


def _read(path):
    with open(path, newline="") as fh:
        return list(csv.reader(fh))


def dataset_fingerprint():
    """Short hash of Train.csv + Test.csv, so an output can name the data it came from."""
    try:
        h = hashlib.sha256()
        for p in (C.TRAIN_CSV, C.TEST_CSV):
            h.update(Path(p).read_bytes())
        return h.hexdigest()[:16]
    except OSError:
        return None


def provenance_problems(M):
    """[(input, provenance)] for every input that is NOT a fresh measurement from the
    current dataset: a placeholder, or an output older than Train.csv."""
    return [(n, p) for n, _, p in M.table()
            if str(p).startswith("FALLBACK") or "STALE" in str(p)]


def load_measured(output_dir=None, ieds_per_unit=None):
    out = Path(output_dir or C.OUTPUT_DIR)
    v = dict(FALLBACK)
    prov = {k: "FALLBACK (placeholder)" for k in v}
    try:
        t_data = max(Path(p).stat().st_mtime for p in (C.TRAIN_CSV, C.TEST_CSV))
    except OSError:
        t_data = None

    def take(values, src):
        tag = src
        try:
            if t_data is not None and (out / src).stat().st_mtime < t_data:
                tag = f"{src} [STALE: older than the dataset files]"
        except OSError:
            pass
        v.update(values)
        prov.update({k: tag for k in values})

    # step 1: merging units per row, and the measured message rate
    try:
        prof = json.loads((out / "step1_profile.json").read_text())
        take(dict(ieds_per_unit=int(prof["n_units"]),
                  sample_rate_hz=float(prof["rates"]["row_rate_hz"])), "step1_profile.json")
    except (OSError, KeyError, ValueError):
        pass
    override = ieds_per_unit if ieds_per_unit is not None else C.MEASURED_UNIT_IEDS
    if override is not None:
        take(dict(ieds_per_unit=int(override)), "config.MEASURED_UNIT_IEDS override")

    # E3: stable feature set and full vector width
    try:
        rows = _read(out / "e3_feature_stability.csv")[1:]
        stable = [r[0] for r in rows if int(r[1]) == C.N_SPLITS]
        take(dict(k_stable=len(stable),
                  k_stable_process=sum(1 for n in stable if classify_column(n) == "process")),
             "e3_feature_stability.csv")
        ks = [int(r[0]) for r in _read(out / "e3_feature_budget.csv")[1:]]
        take(dict(n_features_full=max(ks)), "e3_feature_budget.csv")
    except (OSError, ValueError, IndexError):
        pass

    # E4: per-role inference cost
    try:
        rows = _read(out / "e4_compute_profile.csv")
        head = rows[0]
        by = {r[0]: dict(zip(head, r)) for r in rows[1:]}
        e, s, w = by["tier1_edge_kstar"], by["tier2_zone_fused"], by["tier3_widearea_fused"]
        take(dict(n_features_process=int(by["tier1_edge_process_only"]["n_features"])),
             "e4_compute_profile.csv")
        take(dict(edge_p50_ms=float(e["p50_ms"]), edge_p95_ms=float(e["p95_ms"]),
                  edge_batch_ms=float(e["batch128_per_sample_ms"]),
                  station_p50_ms=float(s["p50_ms"]), station_p95_ms=float(s["p95_ms"]),
                  station_batch_ms=float(s["batch128_per_sample_ms"]),
                  wide_p50_ms=float(w["p50_ms"]), wide_p95_ms=float(w["p95_ms"]),
                  wide_batch_ms=float(w["batch128_per_sample_ms"])),
             "e4_compute_profile.csv")
    except (OSError, KeyError, ValueError):
        pass

    # E1: attack coverage per layer (mean recall over the three attack classes)
    try:
        rows = _read(out / "e1_views_blocked.csv")
        head = rows[0]
        by = {r[0]: dict(zip(head, r)) for r in rows[1:]}
        d_all, d_station = float(by["all"]["attack_coverage"]), float(by["station"]["attack_coverage"])
        take(dict(det_all=d_all, det_station=min(d_station, d_all),
                  det_process=float(by["process"]["attack_coverage"])), "e1_views_blocked.csv")
        if d_station > d_all:
            # Removing sensors cannot raise detection; the excess is fold noise (E1 prints the
            # paired SD). Cap it so the design never shows sensor placement HURTING detection.
            prov["det_station"] += f" [capped: E1 station {d_station:.3f} > all {d_all:.3f}, noise]"
    except (OSError, KeyError, ValueError):
        pass

    return Measured(bytes_per_feature=C.BYTES_PER_FEATURE, provenance=prov, **v)


@dataclass(frozen=True)
class Assumptions:
    n_sites: int
    region_km: float
    frac_gen: float
    frac_junction: float
    line_kusd_per_km: float
    candidate_knn: int
    n1_mode: str
    pmu_obs_k: int
    use_zib: bool
    pmu_kusd: float
    sentinel_kusd_per_bay: float
    station_correlator_kusd: float
    hub_kusd: float
    cc_kusd: float
    fibre_kusd_per_km: float
    fibre_us_per_km: float
    hop_ms: float
    lan_ms: float
    link_capacity_Bps: float
    verdict_bytes: int
    verdict_hz: float
    central_gateway_kusd_per_station: float
    max_zone_share: float
    max_zone_hops: int
    min_zone_stations: int
    max_cores_per_node: int
    util_cap: float
    zone_budget_ms: float
    wide_budget_ms: float
    hardware_slowdown: float
    failure_radius_km: float
    k_zone_max: int
    pam_restarts: int


def default_assumptions(**overrides):
    d = dict(C.GRID_ASSUMPTIONS)
    d.update(overrides)
    return Assumptions(**d)
