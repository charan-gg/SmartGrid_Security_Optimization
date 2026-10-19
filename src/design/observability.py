"""
Stage B - decide where PMUs go.

A PMU at substation p measures its own bus and every incident line, so it makes
p AND all of p's neighbours observable (Ohm's law across each line). Making
every substation observable with as few PMUs as possible is a minimum
DOMINATING SET on the power graph; requiring k-fold observability (survive the
loss of k-1 PMUs) is its k-tuple variant. Junction substations carry no
injection, so Kirchhoff's current law lets observability PROPAGATE through them
(zero-injection rule, a BFS-like fixpoint): this is what `observed_set` checks.

Solvers: exact integer program via scipy.optimize.milp when scipy is present
(HiGHS, deterministic); otherwise greedy + reverse-delete. Both are reported so
the greedy optimality gap is visible. The ZIB pass only ever REMOVES a PMU after
re-verifying full observability, so its answer is always valid.

!! The evidence caveat belongs in the report. E1 measured detection on a testbed whose
!! merging units are all present. Observability-by-inference means some buses are seen
!! through estimated, not measured, phasors, and the dataset cannot say whether detection
!! survives that. E1 does show how much the phasors matter at all: compare its 'all' and
!! 'station' views. If they are close, the PMU count is set by state-estimation
!! observability, not by detection.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def closed_nbhd(g, u):
    return {u} | set(g.adj[u])


def observed_set(g, pmus, zib=()):
    """Substations observed given PMUs, with the zero-injection rule applied to
    fixpoint. A ZIB node z: if its closed neighbourhood is observed except for
    ONE member, that member is deduced observed (KCL at z)."""
    obs = set()
    for p in pmus:
        obs |= closed_nbhd(g, p)
    zib = [z for z in zib if z in g.adj]
    changed = True
    while changed:
        changed = False
        for z in zib:
            missing = [x for x in closed_nbhd(g, z) if x not in obs]
            if len(missing) == 1:
                obs.add(missing[0])
                changed = True
    return obs


def _coverage_counts(g, pmus):
    cnt = {u: 0 for u in g.nodes()}
    for p in pmus:
        for x in closed_nbhd(g, p):
            cnt[x] += 1
    return cnt


def _satisfied(g, pmus, k, zib):
    if k == 1:
        return len(observed_set(g, pmus, zib)) == g.n
    need = {u: min(k, len(closed_nbhd(g, u))) for u in g.nodes()}
    cnt = _coverage_counts(g, pmus)
    return all(cnt[u] >= need[u] for u in g.nodes())


def exact_pmus(g, k=1):
    """Optimal standard (no-ZIB) placement by integer programming, or None if
    scipy is unavailable or the solver fails."""
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError:
        return None
    ids = g.nodes()
    idx = {u: i for i, u in enumerate(ids)}
    n = len(ids)
    A = np.zeros((n, n))
    for u in ids:
        for v in closed_nbhd(g, u):
            A[idx[u], idx[v]] = 1.0
    lb = np.array([min(k, len(closed_nbhd(g, u))) for u in ids], dtype=float)
    res = milp(c=np.ones(n), constraints=LinearConstraint(A, lb=lb, ub=np.inf),
               integrality=np.ones(n), bounds=Bounds(0, 1))
    if not res.success:
        return None
    return {ids[i] for i in range(n) if res.x[i] > 0.5}


def greedy_pmus(g, k=1):
    """Max-marginal-coverage greedy. Returns (set, pick order)."""
    need = {u: min(k, len(closed_nbhd(g, u))) for u in g.nodes()}
    cnt = {u: 0 for u in g.nodes()}
    chosen, order = set(), []
    while any(cnt[u] < need[u] for u in g.nodes()):
        best, best_gain = None, -1
        for c in g.nodes():
            if c in chosen:
                continue
            gain = sum(1 for x in closed_nbhd(g, c) if cnt[x] < need[x])
            if gain > best_gain:
                best, best_gain = c, gain
        chosen.add(best)
        order.append(best)
        for x in closed_nbhd(g, best):
            cnt[x] += 1
    return chosen, order


def prune(g, pmus, k, zib):
    """Reverse-delete: drop any PMU whose removal keeps the grid observable."""
    keep = set(pmus)
    for p in sorted(pmus, key=lambda u: (g.degree(u), u)):    # try weakest first
        trial = keep - {p}
        if _satisfied(g, trial, k, zib):
            keep = trial
    return keep


@dataclass
class PMUPlan:
    pmus: set
    k: int
    method: str                  # how n_standard was obtained
    n_standard: int              # min PMUs without the zero-injection rule
    n_greedy: int                # greedy result, for the optimality-gap line
    n_final: int                 # after the ZIB pass (== n_standard if unused)
    optimal_standard: bool
    observed_frac: float
    curve: list = field(default_factory=list)   # [(n_pmus, observed fraction)]


def place_pmus(g, k=1, use_zib=True):
    zib = [u for u in g.nodes() if g.node[u].get("kind") == "junction"] if use_zib else []
    greedy, order = greedy_pmus(g, k)
    greedy_pruned = prune(g, greedy, k, ())
    exact = exact_pmus(g, k)
    if exact is not None:
        base, method, optimal = exact, "exact (scipy milp)", True
    else:
        base, method, optimal = greedy_pruned, "greedy + reverse-delete", False
    final = prune(g, base, k, zib) if zib else set(base)
    assert _satisfied(g, final, k, zib), "PMU plan failed its own observability check"

    # marginal-coverage curve from the greedy order (standard rule, k=1 view)
    cover, curve = set(), []
    for i, p in enumerate(order, 1):
        cover |= closed_nbhd(g, p)
        curve.append((i, len(cover) / g.n))
    frac = len(observed_set(g, final, zib)) / g.n if k == 1 else 1.0
    return PMUPlan(pmus=final, k=k, method=method, n_standard=len(base),
                   n_greedy=len(greedy_pruned), n_final=len(final),
                   optimal_standard=optimal, observed_frac=frac, curve=curve)
