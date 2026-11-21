# -*- coding: utf-8 -*-
"""
analyze_main.py

Analysis of a main_runs.py directory:

  * split every round count into a TRAIN half (estimation / fitting) and a
    TEST half (all reported numbers), same split for every decoder;
  * decode the TEST half with 4 decoders:
        uniform        repo DEFAULT_NOISE (0.3 % everywhere)
        calibration    EmeraldNoiseModel from the IQM calibration
        data-driven    edge probabilities p_ij estimated on the TRAIN half
        characterized  calibration + per-qubit multipliers fitted on TRAIN
  * fit eps_L per decoder (README model and SPAM-offset model), plot;
  * model validation: predicted vs measured P_L and detector rates.

    python analyze_main.py runs/<ts>_main
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pymatching

from characterize import characterize
from emerald_noise import EmeraldNoiseModel
from estimate_dem import data_driven_matching, logical_error_probability
from fit_ler import fit_logical_error_rate, plot_ler
from surface_code import DEFAULT_NOISE, make_stim_circuit


def _matcher(circuit):
    dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    return pymatching.Matching.from_detector_error_model(dem), dem


def _sim_P(model, circuit, names, shots=200_000, seed=0):
    noisy = model.apply(circuit, names)
    m, _ = _matcher(noisy)
    d, o = noisy.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    return float(np.mean(np.any(m.decode_batch(d) != o, axis=1))), d.mean(axis=0)


def analyze(run_dir: str, rounds=None, r_min: int = 2, verbose=True) -> dict:
    run = pathlib.Path(run_dir)
    names = {int(k): v for k, v in json.loads((run / "layout.json").read_text())["stim_to_name"].items()}
    cal = EmeraldNoiseModel.load(str(run / "noise_model_start.json"))
    rounds = rounds or sorted(int(p.stem[1:]) for p in run.glob("r*.npz"))

    data = {}
    for r in rounds:
        z = np.load(run / f"r{r}.npz")
        det, obs = z["det_events"].astype(bool), z["obs_flips"].astype(bool)
        rng = np.random.default_rng(r)
        perm = rng.permutation(len(det))
        half = len(det) // 2
        data[r] = dict(circ=make_stim_circuit(3, r), train=(det[perm[:half]], obs[perm[:half]]),
                       test=(det[perm[half:]], obs[perm[half:]]))

    # characterization on TRAIN halves of all round counts (joint fit)
    char = characterize(cal, [(v["circ"], v["train"][0]) for v in data.values()], names,
                        verbose=verbose)

    table, fits = {}, {}
    for label in ("uniform", "calibration", "data-driven", "characterized"):
        P, E = [], []
        for r, v in data.items():
            c, (dt, ot) = v["circ"], v["test"]
            if label == "uniform":
                m, _ = _matcher(make_stim_circuit(3, r, noise=DEFAULT_NOISE))
            elif label == "calibration":
                m, _ = _matcher(cal.apply(c, names))
            elif label == "characterized":
                m, _ = _matcher(char.model.apply(c, names))
            else:
                _, dem = _matcher(cal.apply(c, names))
                m, *_ = data_driven_matching(v["train"][0], dem)
            p, e = logical_error_probability(m, dt, ot)
            P.append(p); E.append(e)
        table[label] = dict(P=P, err=E)
        # round 1 does not follow the exponential law (first-round boundary
        # effects): fit rounds >= r_min only, and P_L < 0.45 (no information above)
        ok = [i for i, p in enumerate(P) if p < 0.45 and rounds[i] >= r_min]
        if len(ok) >= 2:
            fits[label] = {mdl: fit_logical_error_rate([rounds[i] for i in ok], [P[i] for i in ok],
                                                       [E[i] for i in ok], model=mdl)
                           for mdl in ("readme", "offset")}

    # model validation: predicted vs measured (best decoder's P_L as "measured")
    validation = {}
    for label, model in (("calibration", cal), ("characterized", char.model)):
        rows = []
        for r, v in data.items():
            p_pred, rates_pred = _sim_P(model, v["circ"], names, seed=r)
            rates_meas = v["test"][0].mean(axis=0)
            rows.append(dict(rounds=r, P_pred=p_pred,
                             P_meas=table[label]["P"][rounds.index(r)],
                             rate_ratio_median=float(np.median(rates_meas / np.maximum(rates_pred, 1e-4)))))
        validation[label] = rows

    if verbose:
        print("\nP_L on TEST halves (same shots for every decoder):")
        print("rounds      " + "".join(f"{r:>16}" for r in rounds))
        for label, t in table.items():
            print(f"{label:12s}" + "".join(f"{p:>9.4f}±{e:.4f}" for p, e in zip(t["P"], t["err"])))
        print("\neps_L per round:")
        for label, f in fits.items():
            print(f"  {label:13s} offset: {f['offset']}\n  {'':13s} readme: {f['readme']}")
        print("\nmodel validation (predicted vs measured P_L, median measured/predicted detector rate):")
        for label, rows in validation.items():
            print(f"  {label:13s} " + " | ".join(
                f"r{x['rounds']}: {x['P_pred']:.3f} vs {x['P_meas']:.3f} (x{x['rate_ratio_median']:.2f})"
                for x in rows))

    plot = plot_ler({k: f["offset"] for k, f in fits.items()}, str(run / "ler_vs_rounds.png"),
                    title="d=3 Z memory on IQM Emerald (test halves)")
    summary = dict(rounds=rounds, table=table,
                   eps={k: {m: dict(eps=f[m].eps, err=f[m].eps_err, A=f[m].A, chi2=f[m].chi2_red)
                            for m in f} for k, f in fits.items()},
                   validation=validation, characterization=dict(
                       scales=char.scales, default=char.default_scale,
                       rms_before=char.rms_log_error_before, rms_after=char.rms_log_error_after),
                   plot=plot)
    (run / "analysis.json").write_text(json.dumps(summary, indent=1))
    char.model.save(str(run / "noise_model_characterized.json"))
    if verbose:
        print(f"\nsaved {run}/analysis.json, ler_vs_rounds.png, noise_model_characterized.json")
    return summary


if __name__ == "__main__":
    analyze(sys.argv[1])
