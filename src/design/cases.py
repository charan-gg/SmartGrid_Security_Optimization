"""
Standard test grids as design inputs.

dataset/grids/<case>_sites.csv and <case>_lines.csv (written by tools/export_cases.py) hold a
substation graph: one node per site, one edge per transmission corridor. A case supplies

  * the SITES   (kind, net MW, relative position)  -> what the optimizer and the default
                                                       builders both start from
  * the AS-BUILT corridors                         -> a reference: the real network's own lines

The cases ship SCHEMATIC layout coordinates, not geography. load_case() rescales them into a
square region whose side follows the project's convention (300 km for 40 sites, growing with
sqrt(n) so site density stays constant) and says so in `Case.note`. Every km-based number
(line cost, fibre latency, the 50 km outage disc) is therefore relative, exactly as for the
synthetic regions. Real coordinates (e.g. the Texas A&M ACTIVSg download) can be supplied as a
--sites CSV instead.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass

from .. import config as C
from .graph import Graph, is_connected
from .sites import Site


@dataclass
class Case:
    name: str
    sites: list            # [Site], ids 0..n-1
    as_built: list         # [(u, v, n_circuits, rating_mva)]
    region_km: float
    note: str

    @property
    def edges(self):
        return [(u, v) for u, v, _, _ in self.as_built]


def available():
    return sorted(p.name[:-len("_sites.csv")] for p in C.GRID_DIR.glob("*_sites.csv"))


def region_km_for(n, base_km=300.0, base_n=40):
    return base_km * math.sqrt(n / base_n)


def _nudge_apart(pts, min_sep):
    """Deterministic: any site closer than min_sep to an earlier one is moved out along a
    fixed spiral until it clears. Keeps zero-length corridors (a division by zero in the line
    selection) out of the design code."""
    out = []
    for i, (x, y) in enumerate(pts):
        k = 0
        while any(math.hypot(x - a, y - b) < min_sep for a, b in out):
            k += 1
            ang = 2.399963 * k                      # golden angle
            r = min_sep * math.sqrt(k)
            x, y = pts[i][0] + r * math.cos(ang), pts[i][1] + r * math.sin(ang)
        out.append((x, y))
    return out


def load_case(name, region_km=None, min_sep_km=1.0) -> Case:
    sp = C.GRID_DIR / f"{name}_sites.csv"
    lp = C.GRID_DIR / f"{name}_lines.csv"
    if not sp.exists() or not lp.exists():
        raise FileNotFoundError(f"case {name!r} not found in {C.GRID_DIR} "
                                f"(available: {', '.join(available()) or 'none'})")
    with open(sp, newline="") as fh:
        rows = list(csv.DictReader(fh))
    ids = [int(r["id"]) for r in rows]
    if sorted(ids) != list(range(len(ids))):
        raise ValueError(f"{name}: site ids must be 0..n-1")
    if len(rows) < 4:
        raise ValueError(f"{name}: need at least 4 sites")
    raw = [(float(r["x"]), float(r["y"])) for r in rows]
    xs, ys = [p[0] for p in raw], [p[1] for p in raw]
    span = max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
    region = float(region_km) if region_km else region_km_for(len(rows))
    k = region / span
    pts = [((x - min(xs)) * k, (y - min(ys)) * k) for x, y in raw]
    pts = _nudge_apart(pts, min_sep_km)
    sites = [Site(int(r["id"]), pts[i][0], pts[i][1], r["kind"].strip().lower(), float(r["mw"]))
             for i, r in enumerate(rows)]

    with open(lp, newline="") as fh:
        lines = [(int(r["from_site"]), int(r["to_site"]), int(r["n_circuits"]),
                  float(r["rating_mva"])) for r in csv.DictReader(fh)]
    n = len(sites)
    for u, v, _, _ in lines:
        if not (0 <= u < n and 0 <= v < n) or u == v:
            raise ValueError(f"{name}: bad corridor {u}-{v}")
    g = Graph()
    for s in sites:
        g.add_node(s.id)
    for u, v, _, _ in lines:
        g.add_edge(u, v)
    if not is_connected(g):
        raise ValueError(f"{name}: the as-built network is not connected")
    note = (f"{name}: {n} substations, {len(lines)} corridors; schematic layout rescaled to a "
            f"{region:.0f} km square (relative geography, not real).")
    return Case(name=name, sites=sites, as_built=lines, region_km=region, note=note)
