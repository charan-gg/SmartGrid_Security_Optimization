"""
Export standard test grids to plain CSV so the project does not need pandapower to run.

Run ONCE (already done; the CSVs are in dataset/grids/):
    pip install pandapower
    python tools/export_cases.py [output_dir]

Each case becomes a SUBSTATION graph: buses joined by transformers are one substation (a
substation is one physical site; the transformer is inside the fence), so the graph the
design code sees has one node per site and one edge per transmission corridor.

    <case>_sites.csv   id,name,x,y,kind,mw,n_buses,vn_kv_max
    <case>_lines.csv   from_site,to_site,n_circuits,rating_mva

x,y are the SCHEMATIC layout coordinates shipped with the case (unitless). They are not
geography. src/design/cases.py rescales them to a square region of a stated size.
kind: gen = has generation (and net export >= 0), load = net importer, junction = no
generation, no load, no shunt (a zero-injection site).
"""
import json
import math
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandapower as pp
import pandapower.networks as pn

CASES = {          # name -> (pandapower constructor, what it is)
    "ieee14": ("case14", "IEEE 14-bus test system"),
    "ieee30": ("case30", "IEEE 30-bus test system (MATPOWER case30)"),
    "ieee39": ("case39", "IEEE 39-bus New England system"),
    "ieee57": ("case57", "IEEE 57-bus test system"),
    "ieee118": ("case118", "IEEE 118-bus test system"),
    "ieee300": ("case300", "IEEE 300-bus test system"),
    "illinois200": ("case_illinois200", "Texas A&M ACTIVSg200 synthetic grid (Illinois footprint)"),
}


def _xy(net):
    out = {}
    for i, g in net.bus["geo"].items():
        c = json.loads(g)["coordinates"] if isinstance(g, str) else g["coordinates"]
        out[i] = (float(c[0]), float(c[1]))
    return out


def export(name, fn, out_dir):
    net = getattr(pn, fn)()
    xy = _xy(net)
    buses = list(net.bus.index)
    parent = {b: b for b in buses}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for r in net.trafo.itertuples():
        if r.in_service:
            parent[find(r.hv_bus)] = find(r.lv_bus)
    if len(net.trafo3w):
        for r in net.trafo3w.itertuples():
            parent[find(r.hv_bus)] = find(r.mv_bus)
            parent[find(r.mv_bus)] = find(r.lv_bus)

    members = {}
    for b in buses:
        members.setdefault(find(b), []).append(b)
    # stable site ids 0..n-1 ordered by lowest member bus index
    roots = sorted(members, key=lambda r: min(members[r]))
    sid = {r: i for i, r in enumerate(roots)}
    site_of = {b: sid[find(b)] for b in buses}

    def total(df, col, mask=None):
        if not len(df):
            return {}
        d = df[df.in_service] if "in_service" in df else df
        return d.groupby("bus")[col].sum().to_dict()

    load = total(net.load, "p_mw")
    gen = total(net.gen, "p_mw")
    sgen = total(net.sgen, "p_mw") if len(net.sgen) else {}
    ext = set(net.ext_grid.bus) if len(net.ext_grid) else set()
    shunt = set(net.shunt.bus) if len(net.shunt) else set()

    rows = []
    for r in roots:
        mem = members[r]
        l = sum(load.get(b, 0.0) for b in mem)
        g = sum(gen.get(b, 0.0) + sgen.get(b, 0.0) for b in mem)
        has_gen = any(b in gen or b in sgen or b in ext for b in mem)
        has_load = any(b in load for b in mem)
        has_shunt = any(b in shunt for b in mem)
        if not (has_gen or has_load or has_shunt):
            kind = "junction"
        elif has_gen and g - l >= 0:
            kind = "gen"
        else:
            kind = "load"
        mw = 0.0 if kind == "junction" else round(g - l, 2)
        rows.append(dict(id=sid[r], name=f"S{sid[r]}",
                         x=round(float(np.mean([xy[b][0] for b in mem])), 5),
                         y=round(float(np.mean([xy[b][1] for b in mem])), 5),
                         kind=kind, mw=mw, n_buses=len(mem),
                         vn_kv_max=float(max(net.bus.vn_kv[b] for b in mem))))

    lines = {}
    for r in net.line.itertuples():
        if not r.in_service:
            continue
        a, b = site_of[r.from_bus], site_of[r.to_bus]
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        mva = math.sqrt(3) * float(net.bus.vn_kv[r.from_bus]) * float(r.max_i_ka)
        d = lines.setdefault(key, [0, 0.0])
        d[0] += 1
        d[1] += mva if mva < 1e5 else 0.0          # unrated (MATPOWER rate_a = 0) shows as a huge number

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{name}_sites.csv", "w", newline="") as fh:
        fh.write("id,name,x,y,kind,mw,n_buses,vn_kv_max\n")
        for s in rows:
            fh.write(f"{s['id']},{s['name']},{s['x']},{s['y']},{s['kind']},{s['mw']},{s['n_buses']},{s['vn_kv_max']}\n")
    with open(out / f"{name}_lines.csv", "w", newline="") as fh:
        fh.write("from_site,to_site,n_circuits,rating_mva\n")
        for (a, b), (n, mva) in sorted(lines.items()):
            fh.write(f"{a},{b},{n},{round(mva, 1)}\n")
    kinds = {k: sum(1 for s in rows if s["kind"] == k) for k in ("gen", "load", "junction")}
    print(f"{name:<12} {fn:<18} buses={len(buses):>3} sites={len(rows):>3} corridors={len(lines):>3} "
          f"{kinds}  ({CASES[name][1]})")
    return dict(name=name, buses=len(buses), sites=len(rows), corridors=len(lines), kinds=kinds)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / "dataset" / "grids")
    meta = [export(n, fn, out) for n, (fn, _) in CASES.items()]
    print("pandapower", pp.__version__, "->", out)
