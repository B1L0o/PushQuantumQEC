# -*- coding: utf-8 -*-
"""
What did the GNN learn about good and bad qubits?

Train the GNN on the hardware run that still contains QB26 (2 rounds), then
scramble the detectors of one ancilla at a time (shuffle them across test
shots, which keeps each detector's firing rate but destroys its information)
and measure how much each decoder's decisions change. A decoder that has
learned an ancilla is useless barely reacts when that ancilla is scrambled.

    python gnn_ablation.py        # writes runs/figures/ablation.json
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import torch

from decoders import GNNDecoder, MWPMDecoder
from emerald_noise import EmeraldNoiseModel
from surface_code import make_stim_circuit

ROOT = pathlib.Path(__file__).resolve().parent
OLD = ROOT / "runs/20261121_181236_main"
OUT = ROOT / "runs/figures/ablation.json"
R = 2


def main():
    torch.set_num_threads(4)
    names = {int(k): v for k, v in json.loads((OLD / "layout.json").read_text())["stim_to_name"].items()}
    cal = EmeraldNoiseModel.load(str(OLD / "noise_model_start.json"))
    z = np.load(OLD / f"r{R}.npz")
    det, obs = z["det_events"].astype(bool), z["obs_flips"].astype(bool)
    perm = np.random.default_rng(R).permutation(len(det))
    tr, te = perm[: len(det) // 2], perm[len(det) // 2:]

    c = make_stim_circuit(3, R)
    dg = cal.apply(c, names).detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    sd, so = cal.scaled(1.7).apply(c, names).compile_detector_sampler(seed=R).sample(
        60_000, separate_observables=True)
    gnn = GNNDecoder(c, dg).fit(sd, so, epochs=12)
    gnn.fit(det[tr], obs[tr], epochs=25, lr=5e-4)
    mw = MWPMDecoder(dg)

    # detector -> ancilla name, via the (x, y) coordinates of the ancilla qubits
    qxy = {tuple(v[:2]): q for q, v in c.get_final_qubit_coordinates().items()}
    det_anc = {i: names[qxy[tuple(v[:2])]] for i, v in c.get_detector_coordinates().items()}
    anc = sorted(set(det_anc.values()), key=lambda s: int(s[2:]))

    X, Y = det[te], obs[te]
    base = {"GNN": gnn.decode(X).reshape(len(X), -1), "MWPM": mw.decode(X).reshape(len(X), -1)}
    P0 = {k: float(np.mean(np.any(v != Y, axis=1))) for k, v in base.items()}
    rng = np.random.default_rng(0)
    out = {"rounds": R, "test_shots": int(len(X)), "P_L": P0, "ancillas": {}}
    for a in anc:
        cols = [i for i, n in det_anc.items() if n == a]
        Xs = X.copy()
        for j in cols:
            Xs[:, j] = X[rng.permutation(len(X)), j]
        row = {"fire_rate": float(X[:, cols].mean())}
        for k, dec in (("GNN", gnn), ("MWPM", mw)):
            p = dec.decode(Xs).reshape(len(X), -1)
            row[k] = dict(changed=float(np.mean(np.any(p != base[k], axis=1))),
                          P_L=float(np.mean(np.any(p != Y, axis=1))))
        out["ancillas"][a] = row
        print(a, row, flush=True)
    OUT.write_text(json.dumps(out, indent=1))
    print("baseline", P0, "->", OUT)


if __name__ == "__main__":
    main()
