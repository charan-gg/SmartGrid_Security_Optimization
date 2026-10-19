"""
Sites: the raw material the grid is synthesised from.

A site is a place that needs a substation: a generator, a load centre, or a
junction (no injection). The synthetic generator is seeded and deterministic.
To design for a real region instead, pass a CSV with columns
    id,x_km,y_km,kind,mw
(kind in gen|load|junction; mw > 0 generation, < 0 load, 0 junction).
"""
from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Site:
    id: int
    x_km: float
    y_km: float
    kind: str          # "gen" | "load" | "junction"
    mw: float          # +generation, -load, 0 junction


def dist_km(a, b):
    return math.hypot(a.x_km - b.x_km, a.y_km - b.y_km)


def synth_sites(n, seed, region_km=300.0, frac_gen=0.2, frac_junction=0.2):
    """n sites in a square region with a minimum separation (so substations do
    not stack on top of each other), kinds assigned at random, generation
    scaled to 120% of load so the system is adequately supplied."""
    if n < 4:
        raise ValueError("need at least 4 sites for an N-1 secure design")
    rng = random.Random(seed)
    min_sep = 0.55 * region_km / math.sqrt(n)
    pts, attempts = [], 0
    while len(pts) < n:
        attempts += 1
        if attempts > 300 * n:           # packing too tight: relax and retry
            min_sep *= 0.9
            attempts = 0
        p = (rng.uniform(0, region_km), rng.uniform(0, region_km))
        if all(math.dist(p, q) >= min_sep for q in pts):
            pts.append(p)
    order = list(range(n))
    rng.shuffle(order)
    n_gen = max(1, round(frac_gen * n))
    n_jn = round(frac_junction * n)
    kind = {}
    for i, idx in enumerate(order):
        kind[idx] = "gen" if i < n_gen else ("junction" if i < n_gen + n_jn else "load")
    raw = {i: (rng.uniform(150, 500) if kind[i] == "gen"
               else -rng.uniform(40, 160) if kind[i] == "load" else 0.0) for i in range(n)}
    load = -sum(v for v in raw.values() if v < 0)
    gen = sum(v for v in raw.values() if v > 0)
    scale = 1.2 * load / gen if gen > 0 else 1.0
    return [Site(i, pts[i][0], pts[i][1], kind[i], raw[i] * (scale if raw[i] > 0 else 1.0))
            for i in range(n)]


def load_sites_csv(path):
    out = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out.append(Site(int(r["id"]), float(r["x_km"]), float(r["y_km"]),
                            r["kind"].strip().lower(), float(r["mw"])))
    ids = [s.id for s in out]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate site ids")
    if len(out) < 4:
        raise ValueError("need at least 4 sites")
    return out
