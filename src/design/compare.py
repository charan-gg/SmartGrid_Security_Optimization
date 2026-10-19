"""
Default vs optimized: the scoring engine.

WHAT IS MEASURED AND WHAT IS SIMULATED
  measured   detector recall of three evidence tiers, with block-bootstrap error bars
             (src/detectors.py, from Train.csv)
  simulated  everything about the grid: lines, sensors, zones, failures. The Kaggle data has
             no topology, so the grids are synthetic layouts (and 7 standard test cases).
The two meet in one place: for every failure scenario the GRAPH decides how many substations
are served by which detector tier, and the MEASURED recall of that tier says how much of an
attack it catches.

DETECTION MODEL (per surviving substation v, in one scenario)
  tiered   v always runs its own correlator (design rule R3), so local detection never needs
           the network. It gets the FUSED recall if a surviving PMU observes v (own PMU, a
           neighbour's, or the zero-injection rule), else the STATION recall.
           The alert is DELIVERED only if v can still reach its hub or the control centre.
  central  v is served only if it can still reach a control centre (primary, or the standby in
           the budget-matched default); otherwise nothing at v is watching. Every substation
           has a PMU in the defaults, so a served v gets the fused recall.
  fused_eff = max(fused, station) in every bootstrap replicate: removing sensors cannot
           improve detection, so a noisy draw where it appears to is not credited.
  Not credited anywhere: cross-substation fusion gain (one testbed substation cannot measure
  it) - which, if real, favours the tiered design, so leaving it out is conservative.

PRIMARY security metrics use DELIVERED coverage (the strict reading: an alert nobody can
receive does not count). LOCAL coverage is reported next to it as secondary.

VERDICT RULES (fixed before looking at results; printed with them)
  per metric  better / worse only if the whole 95% interval of (optimized - default) lies beyond
              a tolerance in that direction; tie if the whole interval lies inside +-tolerance;
              otherwise inconclusive.
  two kinds of security metric. OBJECTIVES are what the design algorithms minimise directly
              (bridges, cut vertices, zone share, routes to the control centre): beating the
              default on them shows the algorithm worked, not that the grid is safer. OUTCOMES
              are what the design never optimised: how much attack detection survives a failure.
              Only outcomes can establish that the optimized grid is better.
  per family  the counts of those outcomes. Price is never netted against security.
  overall     "OPTIMIZED BETTER" requires: >= 1 security OUTCOME win, 0 security losses (either
              kind), 0 performance losses. It is labelled "(narrow)" unless a majority of the
              outcome metrics are clear wins. Otherwise "NOT ESTABLISHED", with the reasons.
Intervals come from a paired two-level bootstrap: layouts resampled with replacement (the
spread across synthetic grids) x detector-recall replicates (the spread in the measured data).
"""
from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import baseline as BL
from . import evaluate as EV
from . import graph as G
from . import hierarchy as HI
from . import observability as OB
from .design import design_grid
from .sites import synth_sites

FAMILIES = ("single", "regional", "n2_random", "n2_targeted")
N2_RANDOM_PAIRS = 100
N2_CRITICAL = 10
N2_TARGETED_KEEP = 40       # the adversary's best pairs; same budget against every candidate


# --------------------------------------------------------------------- recalls
@dataclass
class Recalls:
    """Detector recall inputs, one entry per bootstrap replicate (paired across tiers)."""
    station: np.ndarray
    fused: np.ndarray            # already max(fused, station)

    @property
    def B(self):
        return len(self.station)

    @classmethod
    def from_draws(cls, draws):
        wide = draws.pivot(index="b", columns="tier", values="coverage").dropna(
            subset=["station", "fused"])
        st, fu = wide["station"].to_numpy(float), wide["fused"].to_numpy(float)
        return cls(station=st, fused=np.maximum(fu, st))

    @classmethod
    def point(cls, table):
        t = table.set_index("tier")["coverage"]
        s, f = float(t["station"]), float(t["fused"])
        return cls(station=np.array([s]), fused=np.array([max(f, s)]))


# ------------------------------------------------------------------ scenarios
def shares(c, failed, credit_inferred=True):
    """Shares (of SURVIVING bays) by how a failure scenario leaves the grid:
    det_f/det_s  detected locally by the fused / station-only tier
    del_f/del_s  ...and the alert can reach a fusion point (delivered)
    None if nothing survives."""
    cg, bays, h = c.comm, c.power.bays, c.h
    failed = set(failed)
    alive_nodes = [v for v in cg.nodes() if v not in failed]
    alive = sum(bays[v] for v in alive_nodes)
    if alive == 0:
        return None
    if c.deployment == "tiered":
        own = set(c.pmu.pmus) - failed
        if credit_inferred:
            sub = c.power.g.subgraph(alive_nodes)
            fused = OB.observed_set(sub, own, [z for z in c.zib if z not in failed])
        else:
            fused = own
        to_cc = G.bfs(cg, h.cc, skip_nodes=failed)[0] if h.cc not in failed else {}
        reach = {x: G.bfs(cg, x, skip_nodes=failed)[0] for x in h.hubs if x not in failed}
        delivered = {v for v in alive_nodes if v in to_cc or v in reach.get(h.zone_of[v], {})}
        det_f = sum(bays[v] for v in alive_nodes if v in fused)
        del_f = sum(bays[v] for v in alive_nodes if v in fused and v in delivered)
        del_s = sum(bays[v] for v in alive_nodes if v not in fused and v in delivered)
        return dict(det_f=det_f / alive, det_s=(alive - det_f) / alive,
                    del_f=del_f / alive, del_s=del_s / alive)
    served = set()
    for cc in (h.cc, c.backup):
        if cc is not None and cc not in failed:
            served |= set(G.bfs(cg, cc, skip_nodes=failed)[0])
    served_bays = sum(bays[v] for v in alive_nodes if v in served)
    f = served_bays / alive
    return dict(det_f=f, det_s=0.0, del_f=f, del_s=0.0)


def disc_sets(c, radius_km):
    pos = {u: (c.comm.node[u]["x_km"], c.comm.node[u]["y_km"]) for u in c.comm.nodes()}
    seen, out = set(), []
    for epi in c.comm.nodes():
        s = frozenset(v for v in pos if math.dist(pos[v], pos[epi]) <= radius_km)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def random_pairs(nodes, k, seed):
    rng = random.Random(f"n2-{seed}-{len(nodes)}")
    allp = list(itertools.combinations(sorted(nodes), 2))
    rng.shuffle(allp)
    return [frozenset(p) for p in allp[:k]]


def _damage(s):
    """How much a scenario hurts, with every tier counted as recall 1 (so it does not depend on
    the detector draws): lost local detection + lost delivery, as shares of survivors."""
    return ((1 - s["det_f"] - s["det_s"]) + (1 - s["del_f"] - s["del_s"])) if s else 0.0


def targeted_pairs(c, single_shares, credit_inferred=True):
    """The adversary this candidate must survive: take its fusion points (control centres, hubs,
    standby) and its N2_CRITICAL most damaging single losses, try every pair among them, and
    keep the N2_TARGETED_KEEP that hurt most. The attacker adapts to each design; the budget
    (number of pairs scored) is the same for all. It searches a pool, not all pairs, so a weak
    spot outside the pool is not found - it cannot make a design look worse than it is."""
    fusion = {c.h.cc} | set(c.h.hubs) | ({c.backup} if c.backup is not None else set())
    ranked = sorted(c.comm.nodes(), key=lambda v: (-_damage(single_shares[v]), -c.comm.degree(v), v))
    pool = sorted(fusion | set(ranked[:N2_CRITICAL]))
    pairs = [frozenset(p) for p in itertools.combinations(pool, 2)]
    scored = sorted(pairs, key=lambda f: (-_damage(shares(c, f, credit_inferred)), sorted(f)))
    return scored[:N2_TARGETED_KEEP]


# ---------------------------------------------------------------- structure
def structure(c):
    g, cg, bays, h, A, M = c.power.g, c.comm, c.power.bays, c.h, c.A, c.M
    total = sum(bays.values())
    br, cuts = G.bridges_and_cut_vertices(g)
    worst_isl = 0.0
    for f in g.nodes():
        comps = G.connected_components(g, skip_nodes={f})
        alive = total - bays[f]
        if alive > 0 and comps:
            worst_isl = max(worst_isl, 1 - max(sum(bays[x] for x in k) for k in comps) / alive)
    sinks = [x for x in (h.cc, c.backup) if x is not None]
    routes = min(max(G.disjoint_paths(cg, v, s, vertex=True) if v != s else 99 for s in sinks)
                 for v in cg.nodes())
    tiered = c.deployment == "tiered"
    arch = "C" if tiered else ("B+" if c.backup is not None else "B")
    to_backup = G.dijkstra(cg, c.backup)[1] if c.backup is not None else None
    load = EV.link_loads(h, bays, M, A, arch, to_backup)
    peak = max(load.values()) if load else 0.0
    if tiered:
        cores, latency = max(h.metrics["max_hub_cores"], h.metrics["max_station_cores"]), \
            h.metrics["t_hub_verdict_ms"]
    else:
        cores = HI.cores_needed(total, M.wide_batch_ms, A, M)
        latency = EV.latency_pair(h, M, A)[0]
    from . import security as SE
    return dict(
        n_sites=g.n, n_lines=g.m, n_bays=total, n_pmus=len(c.pmu.pmus),
        spof_bridges=len(br), spof_cut_vertices=len(cuts), spof_total=len(br) + len(cuts),
        worst_node_islanded=worst_isl,
        min_routes_to_cc=routes,
        blast_share=h.metrics["max_zone_share"] if tiered else 1.0,
        lateral_reach=SE.lateral_reach(cg, bays, h)["zoned_mean"],
        k_zones=h.k, n_levels=h.n_levels,
        latency_ms=latency, peak_link_util=peak / A.link_capacity_Bps,
        cc_ingress_KBps=EV._ingress(load, h.cc, cg) / 1e3, max_node_cores=cores,
        cost_total_kusd=c.costs["total"], cost_lines_kusd=c.costs["power_lines"],
        cost_detection_kusd=c.costs["detection_plane"])


# --------------------------------------------------------------- evaluation
@dataclass
class Evaluation:
    name: str
    physical: str
    deployment: str
    structure: dict
    intact: dict
    sets: dict           # family -> list of share dicts (None dropped)

    def arrays(self, key):
        return {fam: np.array([[s[k] for k in key] for s in rows]) for fam, rows in self.sets.items()}


def evaluate(c, seed=0, credit_inferred=True):
    cg = c.comm
    singles = {v: shares(c, {v}, credit_inferred) for v in cg.nodes()}
    sets = {"single": [s for s in singles.values() if s],
            "regional": [s for s in (shares(c, f, credit_inferred) for f in disc_sets(c, c.A.failure_radius_km)) if s],
            "n2_random": [s for s in (shares(c, f, credit_inferred) for f in random_pairs(cg.nodes(), N2_RANDOM_PAIRS, seed)) if s],
            "n2_targeted": [s for s in (shares(c, f, credit_inferred) for f in targeted_pairs(c, singles, credit_inferred)) if s]}
    return Evaluation(c.name, c.physical, c.deployment, structure(c),
                      shares(c, set(), credit_inferred), sets)


def coverage_metrics(ev, rec):
    """{metric: array over recall replicates}. delivered = primary, local = secondary."""
    out = {}

    def cov(s, kind):
        f, s_ = ("del_f", "del_s") if kind == "delivered" else ("det_f", "det_s")
        return s[f] * rec.fused + s[s_] * rec.station
    out["cov_intact"] = cov(ev.intact, "local")
    for kind in ("delivered", "local"):
        worst = []
        for fam, rows in ev.sets.items():
            m = np.mean([cov(s, kind) for s in rows], axis=0)
            out[f"cov_{kind}_{fam}"] = m
            worst.append(np.min([cov(s, kind) for s in rows], axis=0))
        out[f"cov_{kind}_worst"] = np.min(worst, axis=0)
    return out


# --------------------------------------------------------------- the metrics
# (key, label, family, higher_is_better, tolerance, relative?, tallied?)
# family: performance | security-outcome | security-objective | security-secondary | price | info
def metric_table(A):
    lat_tol = 0.01 * A.zone_budget_ms          # 1% of the tier-2 latency budget
    rows = [
        ("cov_intact", "attack coverage, nothing failed", "performance", True, 0.01, False, True),
        ("latency_ms", "worst bay -> first fused verdict (ms)", "performance", False, lat_tol, False, True),
        ("peak_link_util", "busiest link, share of capacity", "performance", False, 0.01, False, True),
    ]
    for fam, lab in (("single", "any one substation lost"), ("regional", f"{A.failure_radius_km:.0f} km regional outage"),
                     ("n2_random", "random 2-substation loss"), ("n2_targeted", "targeted 2-substation loss (adversary)")):
        rows.append((f"cov_delivered_{fam}", f"coverage delivered, {lab}", "security-outcome", True, 0.01, False, True))
    rows += [
        ("cov_delivered_worst", "coverage delivered, worst scenario of all", "security-outcome", True, 0.01, False, True),
        ("spof_total", "single points of failure in the power network (bridges + cut vertices)", "security-objective", False, 0.5, False, True),
        ("worst_node_islanded", "worst one-substation loss: share of the rest cut off", "security-objective", False, 0.01, False, True),
        ("blast_share", "share of bays one compromised hub exposes", "security-objective", False, 0.02, False, True),
        ("lateral_reach", "share of bays an intruder can reach laterally", "security-objective", False, 0.02, False, True),
        ("min_routes_to_cc", "fewest independent routes any substation has to a control centre", "security-objective", True, 0.5, False, True),
        ("cov_local_regional", "(secondary) coverage if local detection counts, regional outage", "security-secondary", True, 0.01, False, False),
        ("cov_local_worst", "(secondary) coverage if local detection counts, worst scenario", "security-secondary", True, 0.01, False, False),
        ("cost_total_kusd", "total cost (kUSD): lines + fibre + sensors + compute", "price", False, 0.05, True, True),
        ("cc_ingress_KBps", "(info) control-centre ingress, KB/s", "info", False, 1e9, False, False),
        ("max_node_cores", "(info) cores on the busiest node", "info", False, 1e9, False, False),
        ("n_pmus", "(info) PMUs installed", "info", False, 1e9, False, False),
    ]
    return rows


def metric_values(ev, rec):
    """{metric: scalar-or-array(B)} for one candidate on one layout."""
    out = dict(ev.structure)
    out.update(coverage_metrics(ev, rec))
    return out


# -------------------------------------------------------------------- build
def build_candidates(d, A, M, sites, seed=0, extras=True):
    """The optimized design and its comparators on ONE set of sites."""
    opt = BL.from_design(d, "optimized")
    rad_power = BL.radial_power(sites, A)
    out = {"optimized": opt,
           "naive": BL.make_candidate("naive", "radial", rad_power, "central", A, M, sites, seed)}
    bm_power, info = BL.budget_matched_power(sites, A, opt.costs["total"])
    out["budget-matched"] = BL.make_candidate("budget-matched", "budget-matched", bm_power,
                                              "central_backup", A, M, sites, seed)
    out["budget-matched"].extra = info
    if extras:                                  # which half of the design does the work?
        out["opt-lines+central"] = BL.make_candidate("opt-lines+central", "optimized", d.power,
                                                     "central", A, M, sites, seed)
        out["radial+tiered"] = BL.make_candidate("radial+tiered", "radial", rad_power, "tiered",
                                                 A, M, sites, seed)
        # CONTROL: the optimized architecture on untargeted lines at the same spend. If the
        # optimized grid still wins against this one, the line DESIGN matters, not just the money
        # and not just tiering.
        out["bm-lines+tiered"] = BL.make_candidate("bm-lines+tiered", "budget-matched", bm_power,
                                                   "tiered", A, M, sites, seed)
    return out


def scaled(A, n):
    return replace(A, n_sites=n, region_km=A.region_km * math.sqrt(n / A.n_sites))


def run_layout(A, M, rec, n, seed, credit_inferred=True, extras=True, sites=None, case=None):
    """Design + evaluate every candidate on one layout. Returns {name: metric dict}."""
    A_n = scaled(A, n) if sites is None else A
    sites = sites or synth_sites(n, seed, A_n.region_km, A_n.frac_gen, A_n.frac_junction)
    d = design_grid(A_n, M, seed=seed, sites=sites, security=False)
    cands = build_candidates(d, A_n, M, sites, seed, extras)
    if case is not None:
        cands["as-built"] = BL.make_candidate("as-built", "as-built", BL.as_built_power(case, A_n),
                                              "central", A_n, M, sites, seed)
    return {name: metric_values(evaluate(c, seed, credit_inferred), rec) for name, c in cands.items()}, \
        {name: c.deployment for name, c in cands.items()}


# ----------------------------------------------------------------- statistics
def classify(lo, hi, tol, higher_is_better):
    if (lo > tol) if higher_is_better else (hi < -tol):
        return "optimized better"
    if (hi < -tol) if higher_is_better else (lo > tol):
        return "default better"
    if lo >= -tol and hi <= tol:
        return "tie"
    return "inconclusive"


def paired_bootstrap(D, J=2000, seed=0):
    """D: [L] (layout-only metric) or [L, B] (recall-dependent). Mean difference over
    layouts, 95% interval from resampling layouts AND (if present) recall replicates."""
    rng = np.random.default_rng(seed)
    D = np.asarray(D, float)
    L = D.shape[0]
    idx = rng.integers(0, L, size=(J, L))
    if D.ndim == 1:
        boot = D[idx].mean(axis=1)
        mean = D.mean()
    else:
        b = rng.integers(0, D.shape[1], size=J)
        boot = D[idx, b[:, None]].mean(axis=1)
        mean = D.mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(mean), float(lo), float(hi)


def verdict_table(layouts, A, default, J=2000, seed=0):
    """layouts: list of {name: metric dict}. One row per metric for 'optimized' vs `default`."""
    rows = []
    for key, label, family, hib, tol, rel, tallied in metric_table(A):
        o = np.array([np.atleast_1d(l["optimized"][key]) for l in layouts], float)
        d = np.array([np.atleast_1d(l[default][key]) for l in layouts], float)
        if o.shape[1] == 1:
            o, d = o[:, 0], d[:, 0]
        diff = (o - d) / np.maximum(np.abs(d), 1e-12) if rel else o - d
        mean, lo, hi = paired_bootstrap(diff, J, seed)
        rows.append(dict(default=default, metric=key, label=label, family=family,
                         higher_is_better=hib, tolerance=tol, relative=rel,
                         optimized=float(np.mean(o)), baseline=float(np.mean(d)),
                         diff=mean, ci_lo=lo, ci_hi=hi, tallied=tallied,
                         outcome=classify(lo, hi, tol, hib) if tol < 1e8 else "info"))
    return pd.DataFrame(rows)


def family_tally(v):
    t = v[v.tallied]
    rows = []
    for fam, g in t.groupby("family", sort=False):
        c = g.outcome.value_counts()
        rows.append(dict(family=fam, metrics=len(g), optimized_better=int(c.get("optimized better", 0)),
                         tie=int(c.get("tie", 0)), default_better=int(c.get("default better", 0)),
                         inconclusive=int(c.get("inconclusive", 0))))
    return pd.DataFrame(rows)


def overall(tally):
    """The pre-declared rule. Returns (label, reasons)."""
    t = tally.set_index("family")

    def row(f):
        return t.loc[f] if f in t.index else dict(optimized_better=0, default_better=0, metrics=0)
    out, obj, perf = row("security-outcome"), row("security-objective"), row("performance")
    reasons = []
    if out["optimized_better"] < 1:
        reasons.append("no security OUTCOME is clearly better")
    if out["default_better"] + obj["default_better"] > 0:
        reasons.append(f"{int(out['default_better'] + obj['default_better'])} security metric(s) are worse")
    if perf["default_better"] > 0:
        reasons.append(f"{int(perf['default_better'])} performance metric(s) are worse")
    if reasons:
        return "NOT ESTABLISHED", reasons
    majority = out["optimized_better"] * 2 > out["metrics"]
    return ("OPTIMIZED BETTER" if majority else "OPTIMIZED BETTER (narrow)"), reasons


def verdict_text(default, v, tally, label, reasons):
    t = tally.set_index("family")

    def cnt(f):
        if f not in t.index:
            return "n/a"
        r = t.loc[f]
        return (f"{int(r.optimized_better)} better / {int(r.tie)} tie / {int(r.default_better)} worse / "
                f"{int(r.inconclusive)} inconclusive of {int(r.metrics)}")
    pr = v[v.metric == "cost_total_kusd"].iloc[0]
    price = ("costs %+.0f%% more" % (100 * pr["diff"]) if pr.outcome == "default better" else
             "costs about the same" if pr.outcome == "tie" else
             "costs %.0f%% less" % (-100 * pr["diff"]) if pr.outcome == "optimized better" else
             "price difference inconclusive")
    s = (f"vs {default}: {label}. Security outcomes: {cnt('security-outcome')}. Security objectives: "
         f"{cnt('security-objective')}. Performance: {cnt('performance')}. The optimized grid {price}.")
    if reasons:
        s += " Because: " + "; ".join(reasons) + "."
    return s
