# -*- coding: utf-8 -*-
"""
Writes the data files and generated TikZ used by presentation/main.tex, from
the saved runs (no QPU, no heavy computation). Run from the repo root:

    python presentation/build_data.py && cd presentation && latexmk -pdf main.tex
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from fit_ler import fit_logical_error_rate
from surface_code import make_stim_circuit

ROOT = pathlib.Path(__file__).resolve().parents[1]
NEW = ROOT / "runs/20261121_193303_main"
OLD = ROOT / "runs/20261121_181236_main"
FIG = ROOT / "runs/figures"
OUT = pathlib.Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)


def write(name, header, rows):
    with open(OUT / name, "w") as f:
        f.write(" ".join(header) + "\n")
        for r in rows:
            f.write(" ".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in r) + "\n")


def err(p, n):
    return float(np.sqrt(p * (1 - p) / n))


# 1. simulation models vs hardware (P_L in %)
s = json.loads((FIG / "sim_models.json").read_text())
write("sim_models.dat", ["r", "uniform", "calibrated", "characterized", "hardware", "hwerr"],
      [(r, 100 * u, 100 * c, 100 * ch, 100 * h, 100 * err(h, 30000))
       for r, u, c, ch, h in zip(s["rounds"], s["uniform"], s["calibrated"], s["characterized"], s["hardware"])])

# 2. QPU: with / without QB26, raw vs MWPM
q = json.loads((FIG / "qpu.json").read_text())
write("qpu.dat", ["r", "oldraw", "oldraw_e", "oldmw", "oldmw_e", "newraw", "newraw_e", "newmw", "newmw_e"],
      [(r, *[v for tag in ("old", "new") for key in ("raw", "mwpm")
             for v in (100 * q[tag][key][i], 100 * err(q[tag][key][i], q[tag]["n"][i]))])
       for i, r in enumerate((1, 2, 3, 4, 5))])

# 3. threshold (eps_L %, vs noise scale)
t = json.loads((FIG / "threshold.json").read_text())
write("threshold.dat", ["f", "d3", "d5", "d7"],
      [(f, 100 * a, 100 * b, 100 * c) for f, a, b, c in zip(t["scale"], t["3"], t["5"], t["7"])])

# 4. eps_L vs distance (projection)
ex = json.loads((NEW / "extrapolation.json").read_text())
keys = list(ex)
write("distance.dat", ["d", "today", "half", "quarter"],
      [(d, *[100 * ex[k]["eps"][str(d)] for k in keys]) for d in (3, 5, 7)])

# 5. decoders (P_L %) + fits; GNN = best variant (defect graph) when available
dec = json.loads((NEW / "decoders_all.json").read_text())["P_L"]
best = NEW / "gnn_best.json"
gnn_best = {}
if best.exists():
    b = json.loads(best.read_text())
    gnn_best = {int(k): v for k, v in b.get("D_all", {}).items()}
labels = ["no correction", "MWPM", "Union-Find", "BP+OSD", "GNN"]
if all(r in gnn_best for r in (1, 2, 3, 4, 5)):
    # best variant (defect graph) replaces the fixed-graph GNN everywhere downstream
    for r in (1, 2, 3, 4, 5):
        dec[str(r)]["GNN"] = gnn_best[r]
rows, fits = [], {}
for r in (1, 2, 3, 4, 5):
    row = [r] + [100 * dec[str(r)][k] for k in labels]
    row.append(100 * gnn_best[r] if r in gnn_best else float("nan"))
    rows.append(row)
write("decoders.dat", ["r", "raw", "mwpm", "uf", "bposd", "gnn", "gnndefect"], rows)
R = [2, 3, 4, 5]
series = {k: [dec[str(r)][k] for r in R] for k in labels}
if all(r in gnn_best for r in R):
    series["GNN (defect graph)"] = [gnn_best[r] for r in R]
for k, P in series.items():
    f = fit_logical_error_rate(R, P, shots=[30000] * 4, model="offset")
    fits[k] = dict(eps=100 * f.eps, err=100 * f.eps_err, A=f.A)
(OUT / "fits.json").write_text(json.dumps(fits, indent=1))
# fitted curves for the fit slide
nn = np.linspace(1.6, 5.4, 40)
cur = []
for x in nn:
    cur.append([x] + [100 * 0.5 * (1 - fits[k]["A"] * (1 - 2 * fits[k]["eps"] / 100) ** x)
                      for k in ("no correction", "MWPM", "GNN")])
write("fitcurves.dat", ["n", "raw", "mwpm", "gnn"], cur)

# 6. exclusion x decoders (P_L %)
e = json.loads((FIG / "exclusion.json").read_text())
names = ["no correction", "MWPM", "Union-Find", "BP+OSD", "GNN"]
for r in ("2", "3"):
    write(f"exclusion_r{r}.dat", ["i", "dec", "old", "new"],
          [(i, n.replace(" ", "").replace("+", ""), 100 * e["old"][r][n], 100 * e["new"][r][n])
           for i, n in enumerate(names)])

# 7. swaps
sw = json.loads((FIG / "swaps.json").read_text())
(OUT / "swaps.json").write_text(json.dumps(sw, indent=1))

# 8. layout drawing (TikZ) of the d=3 code with physical qubit names
c = make_stim_circuit(3, 1)
coords = c.get_final_qubit_coordinates()
det_xy0 = {tuple(v[:2]) for v in c.get_detector_coordinates().values() if v[2] == 0}
from internal_helpers import get_qubit_lists
data_q, anc_q = get_qubit_lists(c)
lay = {int(k): v for k, v in json.loads((NEW / "layout.json").read_text())["stim_to_name"].items()}
lines = []
for a in anc_q:
    x, y = coords[a][:2]
    typ = "Z" if (x, y) in det_xy0 else "X"
    nb = [d for d in data_q if abs(coords[d][0] - x) == 1 and abs(coords[d][1] - y) == 1]
    pts = sorted(((coords[d][0], coords[d][1]) for d in nb), key=lambda p: np.arctan2(p[1] - y, p[0] - x))
    if len(pts) == 2:     # boundary stabilizer: half-disc towards the outside
        pts = pts + [(x, y)]
    fill = "zstab" if typ == "Z" else "xstab"
    lines.append(f"\\fill[{fill}] " + " -- ".join(f"({px},{-py})" for px, py in pts) + " -- cycle;")
for q in data_q:
    x, y = coords[q][:2]
    lines.append(f"\\node[dataq] at ({x},{-y}) {{\\tiny {lay[q][2:]}}};")
for a in anc_q:
    x, y = coords[a][:2]
    typ = "Z" if (x, y) in det_xy0 else "X"
    lines.append(f"\\node[anc{typ.lower()}] at ({x},{-y}) {{\\tiny {lay[a][2:]}}};")
(OUT / "layout.tex").write_text("\n".join(lines) + "\n")
print("wrote", sorted(p.name for p in OUT.iterdir()))

# 9. LaTeX tables (decoders, GNN variants), regenerated as results arrive
def pct(v):
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1f}"

use_defect = all(r in gnn_best for r in (1, 2, 3, 4, 5))
gcol = "gnndefect" if use_defect else "gnn"
with open(OUT / "decoders_table.tex", "w") as f:
    for row in rows:
        r, raw, mw, uf, bp, g_old, g_def = row
        g = g_def if use_defect else g_old
        vals = [mw, uf, bp, g]
        best_v = min(round(v, 1) for v in vals)   # ties at display precision are all bold
        cells = [f"\\textbf{{{pct(v)}}}" if round(v, 1) == best_v else pct(v) for v in vals]
        f.write(f"{r} & {pct(raw)} & " + " & ".join(cells) + " \\\\\n")
    f.write("\\midrule\n$\\varepsilon_L$ / round & " + " & ".join(
        f"{fits[k]['eps']:.1f}" for k in ("no correction", "MWPM", "Union-Find", "BP+OSD",
                                          "GNN (defect graph)" if use_defect else "GNN")) + " \\\\\n")
(OUT / "gnn_label.tex").write_text("GNN (defect graph)" if use_defect else "GNN")

var = json.loads(best.read_text()) if best.exists() else {}
r2 = var.get("r2_partial", {})
r3 = var.get("r3", {})
with open(OUT / "gnn_variants.tex", "w") as f:
    for key, label in (("A_gcn3", "A: GCN, 3 layers, features fired, $x$, $y$, X/Z"),
                       ("B_sage4", "B: fixed DEM graph, 4 layers, $x,y,t$ + flags"),
                       ("C_sage4_type", "C: B + X/Z type"),
                       ("D_defect", "D: defect graph, $k$-NN, 4 layers (toric project)")):
        a = 100 * float(np.mean(r2[key])) if key in r2 else None
        b = 100 * r3[key] if key in r3 else None
        f.write(f"{label} & {pct(a)} & {pct(b)} \\\\\n")
print("tables written; GNN column =", "defect graph" if use_defect else "fixed-graph GNN")

# 10. GNN ablation: scramble one ancilla's detectors, how many decisions change
ab = FIG / "ablation.json"
if ab.exists():
    a = json.loads(ab.read_text())
    z_anc = [q for q, v in a["ancillas"].items() if v["MWPM"]["changed"] > 0]   # Z-type: feed the observable
    write("ablation.dat", ["i", "q", "gnn", "mwpm", "dgnn", "dmwpm"],
          [(i, q, 100 * a["ancillas"][q]["GNN"]["changed"], 100 * a["ancillas"][q]["MWPM"]["changed"],
            100 * (a["ancillas"][q]["GNN"]["P_L"] - a["P_L"]["GNN"]),
            100 * (a["ancillas"][q]["MWPM"]["P_L"] - a["P_L"]["MWPM"])) for i, q in enumerate(z_anc)])
    print("ablation:", z_anc)
