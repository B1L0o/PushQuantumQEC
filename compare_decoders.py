# -*- coding: utf-8 -*-
"""
compare_decoders.py

MWPM, Union-Find, BP+OSD and GNN on the same hardware shots (TEST halves),
plus the eps_L fit for each. GNN: pre-trained on shots simulated with the
calibrated noise model (x1.7, the measured excess), fine-tuned on the
hardware TRAIN half.

    python compare_decoders.py runs/<ts>_main
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

from decoders import BPOSDDecoder, GNNDecoder, MWPMDecoder, UnionFindDecoder
from emerald_noise import EmeraldNoiseModel
from fit_ler import fit_logical_error_rate, plot_ler
from surface_code import make_stim_circuit


def main(run_dir: str, rounds=(1, 2, 3, 4, 5), sim_shots: int = 60_000,
         pre_epochs: int = 12, fine_epochs: int = 25, prior: dict | None = None):
    run = pathlib.Path(run_dir)
    names = {int(k): v for k, v in json.loads((run / "layout.json").read_text())["stim_to_name"].items()}
    cal = EmeraldNoiseModel.load(str(run / "noise_model_start.json"))
    import torch
    torch.set_num_threads(4)
    results = dict(prior or {})
    for r in rounds:
        z = np.load(run / f"r{r}.npz")
        d, o = z["det_events"].astype(bool), z["obs_flips"].astype(bool)
        perm = np.random.default_rng(r).permutation(len(d))
        tr, te = perm[: len(d) // 2], perm[len(d) // 2:]
        c = make_stim_circuit(3, r)
        noisy = cal.apply(c, names)
        dem_g = noisy.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
        dem_h = noisy.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
        row = {"no correction": float(o[te].mean()), "shots_test": int(len(te))}
        for dec in (MWPMDecoder(dem_g), UnionFindDecoder(dem_g), BPOSDDecoder(dem_h)):
            t = time.time()
            row[dec.name] = float(np.mean(np.any(dec.decode(d[te]) != o[te], axis=1)))
            print(f"r={r} {dec.name:11s} {row[dec.name]:.4f} ({time.time() - t:.0f}s)", flush=True)
        t = time.time()
        sd, so = cal.scaled(1.7).apply(c, names).compile_detector_sampler(seed=r).sample(
            sim_shots, separate_observables=True)
        gnn = GNNDecoder(c, dem_g).fit(sd, so, epochs=pre_epochs)
        gnn.fit(d[tr], o[tr], epochs=fine_epochs, lr=5e-4)
        row["GNN"] = float(np.mean(gnn.decode(d[te]) != o[te]))
        print(f"r={r} {'GNN':11s} {row['GNN']:.4f} ({time.time() - t:.0f}s)", flush=True)
        results[r] = row
        (run / "decoders_all.json").write_text(json.dumps(results, indent=1))

    labels = ["no correction", "MWPM", "Union-Find", "BP+OSD", "GNN"]
    fit_rounds = sorted(r for r in results if r >= 2)
    fits, summary = {}, {}
    for lab in labels:
        P = [results[r][lab] for r in fit_rounds]
        n = [results[r]["shots_test"] for r in fit_rounds]
        f_off = fit_logical_error_rate(fit_rounds, P, shots=n, model="offset")
        f_rm = fit_logical_error_rate(fit_rounds, P, shots=n, model="readme")
        fits[lab] = f_off
        summary[lab] = dict(offset=dict(eps=f_off.eps, err=f_off.eps_err, A=f_off.A, chi2=f_off.chi2_red),
                            readme=dict(eps=f_rm.eps, err=f_rm.eps_err, chi2=f_rm.chi2_red))
        print(f"{lab:14s} {f_off}")
    plot_ler(fits, str(run / "ler_vs_rounds_all_decoders.png"),
             "d=3 Z memory on IQM Emerald: 4 decoders (test halves)")
    (run / "decoders_all.json").write_text(json.dumps({"P_L": results, "fits": summary}, indent=1))
    print(f"saved {run}/decoders_all.json and ler_vs_rounds_all_decoders.png")


if __name__ == "__main__":
    # resume: python compare_decoders.py <run> 3 4 5 --prior done.json
    args = sys.argv[1:]
    prior = None
    if "--prior" in args:
        i = args.index("--prior")
        prior = {int(k): v for k, v in json.load(open(args[i + 1])).items()}
        args = args[:i] + args[i + 2:]
    rounds = tuple(int(a) for a in args[1:]) or (1, 2, 3, 4, 5)
    main(args[0], rounds=rounds, prior=prior)
