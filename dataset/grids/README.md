# Standard test grids (design inputs)

Each `<case>_sites.csv` / `<case>_lines.csv` pair is a **substation graph**: buses joined by
transformers are merged into one site, parallel circuits are merged into one corridor.

| case | upstream | sites | corridors |
|---|---|---|---|
| ieee14 | IEEE 14-bus test system | 10 | 15 |
| ieee30 | IEEE 30-bus (MATPOWER case30) | 30 | 41 |
| ieee39 | IEEE 39-bus New England | 28 | 34 |
| ieee57 | IEEE 57-bus | 42 | 62 |
| ieee118 | IEEE 118-bus | 105 | 163 |
| ieee300 | IEEE 300-bus | 172 | 260 |
| illinois200 | Texas A&M ACTIVSg200 synthetic grid (Illinois footprint) | 134 | 177 |

**Provenance.** Exported with `tools/export_cases.py` from the `pandapower` (3.5.x) copies of the
MATPOWER / University of Washington / Texas A&M cases. pandapower is BSD-3; the cases are public
test systems (ACTIVSg: free to use, cite Birchfield et al., IEEE Trans. Power Systems 2017).

**Limits.**
- `x,y` are the *schematic* layout coordinates that ship with the cases, not geography.
  `src/design/cases.py` rescales them to a square region (300 km for 40 sites, growing with
  sqrt(n)); every km-based quantity is therefore relative.
- `kind`/`mw` are derived from load, generation and shunt data (see the export script).
- `rating_mva` = sqrt(3) * kV * max_i_ka for lines; 0 where the upstream case has no rating.
- For real Illinois coordinates, download ACTIVSg200 from the Texas A&M Electric Grid Test Case
  Repository (free, email form) and supply a sites CSV (`id,x_km,y_km,kind,mw`).
