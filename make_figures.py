# -*- coding: utf-8 -*-
"""
make_figures.py

Presentation figures (PNG, slide background) from the saved runs.

    python make_figures.py sim_models swaps qpu threshold distance exclusion

Each figure caches its numbers in runs/figures/<name>.json, so re-drawing is
instant; delete the json to recompute.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pymatching

OLD = pathlib.Path("runs/20261121_181236_main")      # layout with QB26
NEW = pathlib.Path("runs/20261121_193303_main")      # layout without QB26
OUT = pathlib.Path("runs/figures")

# validated categorical order (dataviz reference palette, adjacent pairs pass)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
NEUTRAL = "#8a8780"
INK, INK2, GRID, SURF = "#13202E", "#475569", "#DDD9CE", "#F6F5F0"


# ─────────────────────────────────────────────────────────────────────────────
#  plotting helpers (2px lines, >=8px markers with surface ring, hairline grid)
# ─────────────────────────────────────────────────────────────────────────────

def _fig(w=12.8, h=6.4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15,
                         "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return fig, ax


def _line(ax, x, y, color, label, yerr=None, dashed=False, end_label=True, marker="o"):
    ax.plot(x, y, color=color, lw=2, ls="--" if dashed else "-", solid_capstyle="round",
            label=label, zorder=3)
    ax.scatter(x, y, s=70, color=color, edgecolor=SURF, linewidth=2, zorder=4, marker=marker)
    if yerr is not None:
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=color, elinewidth=1.5, capsize=3, zorder=3)
    if end_label:
        ax.annotate(label, (x[-1], y[-1]), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=13, color=INK2)


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    p = OUT / f"{name}.png"
    fig.savefig(p, facecolor=SURF)
    print("saved", p)
    return p


def _cache(name, compute):
    p = OUT / f"{name}.json"
    if p.exists():
        return json.loads(p.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    data = compute()
    p.write_text(json.dumps(data, indent=1))
    return data


def _names(run):
    return {int(k): v for k, v in json.loads((run / "layout.json").read_text())["stim_to_name"].items()}


def _sim_P(noisy, shots, seed=0):
    dem = noisy.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    d, o = noisy.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    return float(np.mean(np.any(pymatching.Matching.from_detector_error_model(dem).decode_batch(d) != o, axis=1)))


def _eps_from_P(P, n):
    return 0.5 * (1 - (1 - 2 * P) ** (1 / n))


# ─────────────────────────────────────────────────────────────────────────────
#  1. simulation with 3 noise models vs hardware
# ─────────────────────────────────────────────────────────────────────────────

def sim_models():
    from emerald_noise import EmeraldNoiseModel
    from surface_code import DEFAULT_NOISE, make_stim_circuit

    def compute():
        names = _names(NEW)
        cal = EmeraldNoiseModel.load(str(NEW / "noise_model_start.json"))
        char = EmeraldNoiseModel.load(str(NEW / "noise_model_characterized.json"))
        R = [1, 2, 3, 4, 5]
        hw = json.loads((NEW / "decoder_comparison.json").read_text())
        return dict(
            rounds=R,
            uniform=[_sim_P(make_stim_circuit(3, r, noise=DEFAULT_NOISE), 100_000, r) for r in R],
            calibrated=[_sim_P(cal.apply(make_stim_circuit(3, r), names), 100_000, r) for r in R],
            characterized=[_sim_P(char.apply(make_stim_circuit(3, r), names), 100_000, r) for r in R],
            hardware=[hw[str(r)]["mwpm"] for r in R],
        )
    d = _cache("sim_models", compute)
    fig, ax = _fig()
    R = d["rounds"]
    _line(ax, R, d["uniform"], C[2], "Uniform 0.3 % (sim)", dashed=True)
    _line(ax, R, d["calibrated"], C[0], "IQM calibration (sim)", dashed=True)
    _line(ax, R, d["characterized"], C[1], "Characterized (sim)", dashed=True)
    err = [np.sqrt(p * (1 - p) / 30_000) for p in d["hardware"]]
    _line(ax, R, d["hardware"], INK, "Emerald, measured", yerr=err, marker="s")
    ax.set_xlabel("QEC rounds")
    ax.set_ylabel("Logical error probability P_L (MWPM)")
    ax.set_xticks(R)
    ax.set_xlim(0.7, 6.6)
    ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f} %")
    ax.legend(frameon=False, loc="upper left", fontsize=13)
    return _save(fig, "sim_models")


# ─────────────────────────────────────────────────────────────────────────────
#  2. SWAPs and CZ count: default transpilation vs our zero-SWAP placement
# ─────────────────────────────────────────────────────────────────────────────

def swaps():
    def compute():
        from iqm.qiskit_iqm import IQMProvider
        from qiskit import transpile
        from emerald_noise import EmeraldNoiseModel
        from layout_search import physical_index_resolver, transpile_zero_swap
        from run_on_hardware import SERVER_URL, TOKEN
        from stim_to_qiskit import stim_to_qiskit
        from surface_code import make_stim_circuit
        backend = IQMProvider(SERVER_URL, quantum_computer="emerald", token=TOKEN).get_backend()
        circ = make_stim_circuit(3, 3)
        qc, s2d, mo = stim_to_qiskit(circ)
        ideal = sum(1 for i in qc.data if i.operation.num_qubits == 2)
        char = EmeraldNoiseModel.load(str(NEW / "noise_model_characterized.json"))
        d2s = {d: s for s, d in s2d.items()}

        def sim_layout(qt):
            init = qt.layout.initial_index_layout(filter_ancillas=True)
            names = {d2s[i]: backend.index_to_qubit_name(p) for i, p in enumerate(init)}
            return _sim_P(char.apply(circ, names), 100_000, seed=3), sorted(set(names.values()), key=lambda x: int(x[2:]))

        rows = {}
        qt = transpile(qc, backend=backend, initial_layout=list(range(qc.num_qubits)),
                       optimization_level=1, seed_transpiler=0)
        rows["Naive: QB1..QB17 in order"] = dict(cz=int(qt.count_ops().get("cz", 0)), depth=int(qt.depth()),
                                                 sim_P=None, qubits=None)
        qt = transpile(qc, backend=backend, optimization_level=1, seed_transpiler=0)
        P, q = sim_layout(qt)
        rows["Qiskit default layout"] = dict(cz=int(qt.count_ops().get("cz", 0)), depth=int(qt.depth()),
                                             sim_P=P, qubits=q, has_QB26="QB26" in q)
        lay = json.loads((NEW / "layout.json").read_text())
        phys = {int(k): int(v) for k, v in lay["stim_to_phys"].items()}
        qt, rep = transpile_zero_swap(qc, backend, phys, s2d, mo)
        names = {int(k): v for k, v in lay["stim_to_name"].items()}
        rows["Ours: calibrated + characterized"] = dict(
            cz=rep["two_qubit_gates"], depth=rep["depth"],
            sim_P=_sim_P(char.apply(circ, names), 100_000, seed=3),
            qubits=sorted(set(names.values()), key=lambda x: int(x[2:])), has_QB26=False)
        for v in rows.values():
            v["extra_cz"] = v["cz"] - ideal
            v["swaps_equiv"] = round(v["extra_cz"] / 3, 1)
        return dict(ideal_cz=ideal, rows=rows)
    d = _cache("swaps", compute)
    fig, ax = _fig(12.8, 5.2)
    labels = list(d["rows"])
    vals = [d["rows"][k]["cz"] for k in labels]
    colors = [NEUTRAL, NEUTRAL, C[0]]
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, height=0.45, color=colors, zorder=3)
    ax.axvline(d["ideal_cz"], color=INK2, lw=1, zorder=2)
    ax.annotate(f"ideal circuit: {d['ideal_cz']} CZ", (d["ideal_cz"], len(labels) - 0.45),
                xytext=(6, 0), textcoords="offset points", fontsize=13, color=INK2)
    for yi, k in zip(y, labels):
        r = d["rows"][k]
        txt = f"{r['cz']} CZ · ≈{r['swaps_equiv']:g} SWAPs" if r["extra_cz"] > 0 else f"{r['cz']} CZ · 0 SWAPs"
        if r.get("sim_P") is not None:
            txt += f" · sim. P_L {100 * r['sim_P']:.1f} %" + (" (uses QB26)" if r.get("has_QB26") else "")
        ax.annotate(txt, (r["cz"], yi), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=13, color=INK)
    ax.set_yticks(y, labels)
    ax.set_xlabel("CZ gates on Emerald, d = 3, 3 rounds")
    ax.set_xlim(0, max(vals) * 1.9)
    ax.grid(axis="y", visible=False)
    return _save(fig, "swaps")


# ─────────────────────────────────────────────────────────────────────────────
#  3. QPU results: with / without QB26, no correction vs MWPM
# ─────────────────────────────────────────────────────────────────────────────

def qpu():
    def compute():
        from emerald_noise import EmeraldNoiseModel
        from surface_code import make_stim_circuit
        out = {}
        for tag, run in (("old", OLD), ("new", NEW)):
            names = _names(run)
            cal = EmeraldNoiseModel.load(str(run / "noise_model_start.json"))
            raw, mw, n = [], [], []
            for r in (1, 2, 3, 4, 5):
                z = np.load(run / f"r{r}.npz")
                det, obs = z["det_events"].astype(bool), z["obs_flips"].astype(bool)
                te = np.random.default_rng(r).permutation(len(det))[len(det) // 2:]
                dem = cal.apply(make_stim_circuit(3, r), names).detector_error_model(
                    decompose_errors=True, approximate_disjoint_errors=True)
                pred = pymatching.Matching.from_detector_error_model(dem).decode_batch(det[te])
                raw.append(float(obs[te].mean()))
                mw.append(float(np.mean(np.any(pred != obs[te], axis=1))))
                n.append(int(len(te)))
            out[tag] = dict(raw=raw, mwpm=mw, n=n)
        return out
    d = _cache("qpu", compute)
    fig, ax = _fig()
    R = [1, 2, 3, 4, 5]
    e = lambda P, n: [np.sqrt(p * (1 - p) / k) for p, k in zip(P, n)]
    _line(ax, R, d["old"]["raw"], NEUTRAL, "With QB26, no correction", e(d["old"]["raw"], d["old"]["n"]), dashed=True, end_label=False)
    _line(ax, R, d["old"]["mwpm"], C[1], "With QB26, MWPM", e(d["old"]["mwpm"], d["old"]["n"]), end_label=False)
    _line(ax, R, d["new"]["raw"], INK2, "QB26 excluded, no correction", e(d["new"]["raw"], d["new"]["n"]), dashed=True, marker="s", end_label=False)
    _line(ax, R, d["new"]["mwpm"], C[0], "QB26 excluded, MWPM", e(d["new"]["mwpm"], d["new"]["n"]), marker="s", end_label=False)
    ax.set_xlabel("QEC rounds")
    ax.set_ylabel("Logical error probability P_L (held-out shots)")
    ax.set_xticks(R)
    ax.set_xlim(0.7, 5.3)
    ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f} %")
    ax.legend(frameon=False, loc="upper left", fontsize=13)
    return _save(fig, "qpu")


# ─────────────────────────────────────────────────────────────────────────────
#  4. threshold (simulation, characterized model scaled)
# ─────────────────────────────────────────────────────────────────────────────

def _base_model():
    from emerald_noise import EmeraldNoiseModel
    cal = EmeraldNoiseModel.load(str(NEW / "noise_model_start.json"))
    char = EmeraldNoiseModel.load(str(NEW / "noise_model_characterized.json"))
    return cal.with_qubit_scales({}, default_scale=char.default_scale)


def threshold():
    from surface_code import make_stim_circuit

    def compute():
        base = _base_model()
        F = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
        out = {"scale": F}
        for dist in (3, 5, 7):
            eps = []
            for f in F:
                c = make_stim_circuit(dist, dist)
                n = [f"S{q}" for q in range(c.num_qubits)]
                m = base.scaled(f).sample_qubits(n, np.random.default_rng(dist))
                P = _sim_P(m.apply(c, dict(zip(range(c.num_qubits), n))), 30_000, seed=int(100 * f))
                eps.append(_eps_from_P(min(P, 0.499), dist))
            out[str(dist)] = eps
        return out
    d = _cache("threshold", compute)
    fig, ax = _fig()
    F = d["scale"]
    for k, dist in enumerate((3, 5, 7)):
        _line(ax, F, d[str(dist)], C[k], f"d = {dist}")
    ax.axvline(1.0, color=INK2, lw=1)
    ax.annotate("Emerald today", (1.0, ax.get_ylim()[0]), xytext=(-8, 10), textcoords="offset points",
                ha="right", fontsize=13, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(F, [f"{f:g}×" for f in F])
    ax.minorticks_off()
    ax.set_xlabel("Physical noise, relative to Emerald (characterized model)")
    ax.set_ylabel("Logical error per round ε_L (rounds = d)")
    ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:g} %")
    ax.legend(frameon=False, loc="upper left", fontsize=13)
    return _save(fig, "threshold")


# ─────────────────────────────────────────────────────────────────────────────
#  5. LER vs distance (projection)
# ─────────────────────────────────────────────────────────────────────────────

def distance():
    ex = json.loads((NEW / "extrapolation.json").read_text())
    fig, ax = _fig()
    D = [3, 5, 7]
    for k, (lab, v) in enumerate(ex.items()):
        lab = {"current Emerald (characterized)": "Emerald today"}.get(lab, lab)
        _line(ax, D, [v["eps"][str(x)] for x in D], C[k], lab)
    ax.set_yscale("log")
    ax.set_xticks(D, [f"d = {x}" for x in D])
    ax.set_xlim(2.6, 8.2)
    ax.minorticks_off()
    ax.set_ylabel("Logical error per round ε_L (simulation)")
    ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:g} %")
    ax.legend(frameon=False, loc="lower left", fontsize=13)
    return _save(fig, "distance")


# ─────────────────────────────────────────────────────────────────────────────
#  6. bad-qubit exclusion x decoders (hardware, r = 2 and 3)
# ─────────────────────────────────────────────────────────────────────────────

def exclusion():
    from decoders import BPOSDDecoder, GNNDecoder, MWPMDecoder, UnionFindDecoder
    from emerald_noise import EmeraldNoiseModel
    from surface_code import make_stim_circuit

    def compute():
        import torch
        torch.set_num_threads(4)
        new = json.loads((NEW / "decoders_all.json").read_text())
        new = new.get("P_L", new)
        out = {"rounds": [2, 3], "new": {}, "old": {}}
        for r in (2, 3):
            out["new"][str(r)] = {k: new[str(r)][k] for k in ("no correction", "MWPM", "Union-Find", "BP+OSD", "GNN")}
        names = _names(OLD)
        cal = EmeraldNoiseModel.load(str(OLD / "noise_model_start.json"))
        for r in (2, 3):
            z = np.load(OLD / f"r{r}.npz")
            det, obs = z["det_events"].astype(bool), z["obs_flips"].astype(bool)
            perm = np.random.default_rng(r).permutation(len(det))
            tr, te = perm[: len(det) // 2], perm[len(det) // 2:]
            c = make_stim_circuit(3, r)
            noisy = cal.apply(c, names)
            dg = noisy.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
            dh = noisy.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
            row = {"no correction": float(obs[te].mean())}
            for dec in (MWPMDecoder(dg), UnionFindDecoder(dg), BPOSDDecoder(dh)):
                row[dec.name] = float(np.mean(np.any(dec.decode(det[te]) != obs[te], axis=1)))
            sd, so = cal.scaled(1.7).apply(c, names).compile_detector_sampler(seed=r).sample(
                60_000, separate_observables=True)
            g = GNNDecoder(c, dg).fit(sd, so, epochs=12)
            g.fit(det[tr], obs[tr], epochs=25, lr=5e-4)
            row["GNN"] = float(np.mean(g.decode(det[te]) != obs[te]))
            out["old"][str(r)] = row
            print("old layout", r, row, flush=True)
        return out
    d = _cache("exclusion", compute)
    import matplotlib.pyplot as plt
    labels = ["no correction", "MWPM", "Union-Find", "BP+OSD", "GNN"]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.0), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURF)
    for ax, r in zip(axes, ("2", "3")):
        ax.set_facecolor(SURF)
        ax.grid(True, axis="y", color=GRID, lw=1)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        x = np.arange(len(labels))
        w = 0.36
        old = [d["old"][r][k] for k in labels]
        new = [d["new"][r][k] for k in labels]
        ax.bar(x - w / 2 - 0.01, old, w, color=C[1], label="With QB26", zorder=3)
        ax.bar(x + w / 2 + 0.01, new, w, color=C[0], label="QB26 excluded", zorder=3)
        for xi, v in zip(x, new):
            ax.annotate(f"{100 * v:.1f}", (xi + w / 2, v), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=11, color=INK)
        for xi, v in zip(x, old):
            ax.annotate(f"{100 * v:.1f}", (xi - w / 2, v), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=11, color=INK2)
        ax.set_xticks(x, ["No corr.", "MWPM", "Union-Find", "BP+OSD", "GNN"], fontsize=13, color=INK2)
        ax.set_title(f"{r} rounds", fontsize=16, color=INK, loc="left")
        ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f} %")
        ax.tick_params(colors=INK2)
    axes[0].set_ylabel("Logical error probability P_L (hardware, held-out)", color=INK2)
    axes[0].legend(frameon=False, loc="upper left", fontsize=13)
    return _save(fig, "exclusion")


def fit():
    from fit_ler import fit_logical_error_rate
    d = json.loads((NEW / "decoders_all.json").read_text())["P_L"]
    R = [2, 3, 4, 5]
    fig, ax = _fig()
    nn = np.linspace(1.5, 5.5, 100)
    for k, (lab, col) in enumerate((("no correction", NEUTRAL), ("MWPM", C[0]), ("GNN", C[1]))):
        P = [d[str(r)][lab] for r in R]
        n = [d[str(r)]["shots_test"] for r in R]
        f = fit_logical_error_rate(R, P, shots=n, model="offset")
        name = {"no correction": "No correction"}.get(lab, lab)
        ax.plot(nn, 0.5 * (1 - f.A * (1 - 2 * f.eps) ** nn), color=col, lw=2, zorder=2,
                label=f"{name}: ε_L = {100 * f.eps:.1f} ± {100 * f.eps_err:.1f} % per round")
        ax.errorbar(R, P, yerr=[np.sqrt(p * (1 - p) / m) for p, m in zip(P, n)], fmt="none",
                    ecolor=col, elinewidth=1.5, capsize=3, zorder=3)
        ax.scatter(R, P, s=80, color=col, edgecolor=SURF, linewidth=2, zorder=4)
    ax.set_xlabel("QEC rounds")
    ax.set_ylabel("Logical error probability P_L (hardware)")
    ax.set_xticks(R)
    ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f} %")
    ax.legend(frameon=False, loc="upper left", fontsize=14)
    ax.annotate("fit: 2·P_L = 1 − A·(1 − 2ε_L)^n", (0.98, 0.04), xycoords="axes fraction",
                ha="right", fontsize=14, color=INK2)
    return _save(fig, "fit")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["sim_models", "swaps", "qpu", "threshold", "distance", "exclusion"]
    for name in todo:
        globals()[name]()
