"""
DEFAULT (non-optimized) grids, and the Candidate record every grid is judged as.

The optimized grid comes from design.design_grid(): targeted N-1 line augmentation, a minimum
dominating set of PMUs, k-medoid zoning, tiered detection. A fair comparison needs defaults that
a reasonable builder would actually produce from the SAME sites, and one that is not a strawman:

  naive default          lines  cheapest connected network (MST, radial)
                         PMUs   one per substation
                         fusion one control centre at the largest site, no zones, no backup
                         detect centralised: every substation streams to the control centre
  budget-matched default lines  the MST plus the SHORTEST extra corridors until the total spend
                                equals the optimized grid's (untargeted redundancy)
                         PMUs   one per substation; fusion: a control centre + a hot standby
                         detect centralised
  as-built (reference)   a test case's own corridors, naive deployment

The budget-matched default answers "isn't the optimized grid just spending more?": it spends the
same, only without aiming the money at weak points.

Both defaults price their detection plane as the control centre(s) plus a per-station collector
(config central_gateway_kusd_per_station = 0, i.e. favourable to the default).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import evaluate as EV
from . import hierarchy as HI
from . import observability as OB
from . import power as PW
from . import security as SE

DEPLOYMENTS = ("central", "central_backup", "tiered")


@dataclass
class Candidate:
    name: str
    physical: str              # radial | budget-matched | optimized | as-built
    deployment: str            # central | central_backup | tiered
    sites: list
    power: PW.PowerGrid
    pmu: OB.PMUPlan
    zib: list
    comm: object
    h: HI.Hierarchy
    backup: object             # backup control-centre site (central_backup only)
    costs: dict
    A: object
    M: object
    obs: set = field(default_factory=set)      # substations a PMU observes (directly or by inference)
    extra: dict = field(default_factory=dict)


def naive_cc(sites):
    """Where an operator puts the control centre without optimizing: the biggest site."""
    return max(sites, key=lambda s: (abs(s.mw), -s.id)).id


def flat_hierarchy(comm, bays, A, M, cc):
    """k = 1, hub == CC, no zoning: the centralised layout, evaluated by the same code that
    judges the optimized zones (so its blast radius, latency and policy violations are
    computed, not assumed)."""
    ids, idx, D = HI.latency_matrix(comm)
    w = np.array([bays[u] for u in ids], dtype=float)
    return HI._evaluate(comm, ids, idx, D, w, [idx[cc]], bays, A, M)


def all_pmus(g):
    ids = set(g.nodes())
    return OB.PMUPlan(pmus=ids, k=1, method="one PMU per substation (default practice)",
                      n_standard=len(ids), n_greedy=len(ids), n_final=len(ids),
                      optimal_standard=False, observed_frac=1.0, curve=[])


def make_candidate(name, physical, power, deployment, A, M, sites, seed=0):
    """Wrap a physical grid with a deployment. Centralised deployments get the naive CC, PMUs
    everywhere and no zoning; tiered gets the designed hierarchy and the minimum PMU set."""
    if deployment not in DEPLOYMENTS:
        raise ValueError(f"unknown deployment {deployment!r}")
    comm = HI.build_comm_graph(power.g, A)
    if deployment == "tiered":
        h = HI.design_hierarchy(comm, power.bays, A, M, seed=seed)[0]
        zib = ([u for u in power.g.nodes() if power.g.node[u]["kind"] == "junction"]
               if A.use_zib else [])
        pmu = OB.place_pmus(power.g, A.pmu_obs_k, A.use_zib)
        backup = None
    else:
        h = flat_hierarchy(comm, power.bays, A, M, naive_cc(sites))
        zib, pmu = [], all_pmus(power.g)
        backup = SE.backup_cc(comm, power.bays, h, A.failure_radius_km) \
            if deployment == "central_backup" else None
    costs = EV.cost_accounts(power, pmu, comm, power.bays, h, A, deployment)
    return Candidate(name=name, physical=physical, deployment=deployment, sites=sites,
                     power=power, pmu=pmu, zib=zib, comm=comm, h=h, backup=backup, costs=costs,
                     A=A, M=M, obs=OB.observed_set(power.g, pmu.pmus, zib))


def from_design(d, name="optimized"):
    """The optimized grid (design.design_grid) as a Candidate - nothing recomputed."""
    return Candidate(name=name, physical="optimized", deployment="tiered", sites=d.sites,
                     power=d.power, pmu=d.pmu, zib=d.zib, comm=d.comm, h=d.hier, backup=None,
                     costs=d.costs, A=d.A, M=d.M,
                     obs=OB.observed_set(d.power.g, d.pmu.pmus, d.zib))


# ----------------------------------------------------------------- physical defaults
def radial_power(sites, A):
    _, mst, _ = PW.candidate_pool(sites, A)
    return PW.power_grid_from_edges(sites, list(mst), A, label="radial")


def budget_matched_power(sites, A, target_total_kusd, deployment="central_backup"):
    """MST + shortest-first extra corridors from the same candidate pool, stopping when line +
    fibre spend plus the default detection plane would exceed `target_total_kusd`."""
    _, mst, pool = PW.candidate_pool(sites, A)
    n = len(sites)
    fixed = (n * A.pmu_kusd + n * A.central_gateway_kusd_per_station
             + A.cc_kusd * (2 if deployment == "central_backup" else 1))
    per_km = A.line_kusd_per_km + A.fibre_kusd_per_km
    budget = target_total_kusd - fixed
    edges = dict(mst)
    spent = sum(mst.values()) * per_km
    for (u, v), d in sorted(pool.items(), key=lambda kv: (kv[1], kv[0])):
        if (u, v) in edges:
            continue
        if spent + d * per_km > budget:
            break
        edges[(u, v)] = d
        spent += d * per_km
    power = PW.power_grid_from_edges(sites, list(edges), A, label="budget-matched")
    info = dict(line_fibre_budget_kusd=round(budget, 1), spent_kusd=round(spent, 1),
                extra_corridors=len(edges) - len(mst), over_budget_at_mst=spent > budget + 1e-6)
    return power, info


def as_built_power(case, A):
    return PW.power_grid_from_edges(case.sites, case.edges, A, label="as-built")
