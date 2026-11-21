# -*- coding: utf-8 -*-
"""
smoke_test.py

Run BEFORE the main Emerald runs (a few thousand shots in total). It checks the
part of the pipeline that simulation cannot: raw hardware bits -> Stim
measurement record -> detection events.

  Test 1  known states   X on a chosen set of qubits, then every measurement of
                         the surface-code circuit (same layout, same clbits,
                         same counts -> array conversion). The bits that come
                         back must be 1 exactly where expected. Run with two
                         complementary patterns -> per-qubit readout errors
                         P(1|0) and P(0|1) (asymmetry!) as a bonus.
  Test 2  one round      d=3, 1 round through run_qiskit(). First-round
                         detectors are deterministic: their firing rates must
                         be at readout-error level, and of the same order as
                         the noise-model prediction.

Everything is saved to runs/<timestamp>_smoke/ (calibration, layout, counts).

Usage
-----
    export TOKEN_IQM=...            # never put the token in a file
    python smoke_test.py            # on Emerald
    python smoke_test.py --fake     # offline, IQMFakeAphrodite (no token)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib

import numpy as np
from qiskit import QuantumCircuit

from emerald_noise import EmeraldNoiseModel
from internal_helpers import counts_to_measurement_array
from layout_search import transpile_zero_swap
from run_on_hardware import SERVER_URL, run_qiskit
from stim_to_qiskit import stim_to_qiskit
from surface_code import make_stim_circuit


def known_state_circuit(stim_circ, ones: set[int]):
    """Same qubits / clbits / measurement order as the surface-code circuit,
    but only X on `ones` (stim qubit indices) followed by all measurements."""
    _, stim_to_dense, meas_order = stim_to_qiskit(stim_circ)
    qc = QuantumCircuit(len(stim_to_dense), len(meas_order))
    for q in ones:
        qc.x(stim_to_dense[q])
    for j, q in enumerate(meas_order):
        qc.measure(stim_to_dense[q], j)
    return qc, stim_to_dense, meas_order


def test_known_states(backend, layout, rounds: int = 2, shots: int = 1000,
                      max_error: float = 0.4, warn_error: float = 0.2) -> dict:
    """PASS = data path correct (no qubit near 50 % / swapped). Qubits between
    warn_error and max_error are physically bad (e.g. measurement-induced 1->0
    decay), not a pipeline bug: they are listed as warnings."""
    stim_circ = make_stim_circuit(3, rounds, no_reset=True)
    qubits = sorted(layout.stim_to_phys)
    pattern_a = set(qubits[::2])
    out = {"per_qubit": {}, "passed": True}
    p10, p01 = {}, {}
    for ones in (pattern_a, set(qubits) - pattern_a):
        qc, s2d, mo = known_state_circuit(stim_circ, ones)
        qc_t, _ = transpile_zero_swap(qc, backend, layout.stim_to_phys, s2d, mo)
        counts = backend.run(qc_t, shots=shots).result().get_counts()
        raw = counts_to_measurement_array(counts, len(mo), shots)
        expected = np.array([q in ones for q in mo])
        for j, q in enumerate(mo):
            err = float(np.mean(raw[:, j] != expected[j]))
            (p01 if expected[j] else p10).setdefault(q, []).append(err)
        out.setdefault("counts", []).append(counts)
    for q in qubits:
        name = layout.stim_to_name[q]
        e10 = float(np.mean(p10.get(q, [np.nan])))   # prepared 0, read 1
        e01 = float(np.mean(p01.get(q, [np.nan])))   # prepared 1, read 0
        ok = e10 < max_error and e01 < max_error
        warn = ok and max(e10, e01) >= warn_error
        out["per_qubit"][name] = dict(p_1_given_0=e10, p_0_given_1=e01, ok=ok, warn=warn)
        out["passed"] &= ok
    return out


def test_one_round(backend, layout, noise_model, shots: int = 2000,
                   max_rate: float = 0.35) -> dict:
    syn = run_qiskit(3, 1, shots=shots, backend=backend, noise_model=noise_model,
                     layout=layout)
    rates = syn["det_events"].mean(axis=0)
    noisy = noise_model.apply(make_stim_circuit(3, 1), layout.stim_to_name)
    pred = noisy.compile_detector_sampler(seed=0).sample(50_000).mean(axis=0)
    ratio = float(np.median(rates / np.maximum(pred, 1e-4)))
    passed = bool(rates.max() < max_rate and 1 / 4 < ratio < 4)
    return dict(passed=passed, detector_rates=rates.tolist(),
                predicted_rates=pred.tolist(), median_measured_over_predicted=ratio,
                raw_logical_flip_rate=float(syn["obs_flips"].mean()),
                counts=syn["counts"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true", help="offline: IQMFakeAphrodite")
    ap.add_argument("--shots", type=int, default=1000)
    args = ap.parse_args()

    if args.fake:
        from iqm.qiskit_iqm.fake_backends import IQMFakeAphrodite
        backend = IQMFakeAphrodite()
        nm = EmeraldNoiseModel.from_iqm_error_profile(backend.error_profile)
    else:
        from iqm.qiskit_iqm import IQMProvider
        from run_on_hardware import TOKEN as token
        if not token:
            raise SystemExit("No token: export TOKEN_IQM=... or put it in ~/.iqm_token")
        backend = IQMProvider(SERVER_URL, quantum_computer="emerald", token=token).get_backend()
        nm = EmeraldNoiseModel.from_iqm(token, "emerald", SERVER_URL)

    run_dir = pathlib.Path("runs") / (dt.datetime.now().strftime("%Y%m%d_%H%M%S") + "_smoke")
    run_dir.mkdir(parents=True, exist_ok=True)
    nm.save(str(run_dir / "noise_model.json"))
    print(f"[smoke] backend: {backend.num_qubits} qubits, "
          f"{len({tuple(sorted(e)) for e in backend.coupling_map.get_edges()})} couplers | calibration: {nm.source}")

    dry = run_qiskit(3, 3, backend=backend, noise_model=nm, dry_run=True)
    layout = dry["layout"]
    (run_dir / "layout.json").write_text(json.dumps(
        {"stim_to_name": layout.stim_to_name, "stim_to_phys": layout.stim_to_phys,
         "simulated_P_L_r3": layout.ler}, indent=1))

    t1 = test_known_states(backend, layout, shots=args.shots)
    print("\n[smoke] TEST 1 known states:", "PASS" if t1["passed"] else "FAIL")
    for name, v in t1["per_qubit"].items():
        print(f"   {name:>5}: P(1|0) = {v['p_1_given_0']:.3f}   P(0|1) = {v['p_0_given_1']:.3f}"
              f"{'   <-- WRONG BIT (pipeline bug?)' if not v['ok'] else ('   <-- bad qubit (physical)' if v['warn'] else '')}")

    t2 = test_one_round(backend, layout, nm, shots=2 * args.shots)
    print("\n[smoke] TEST 2 one round:", "PASS" if t2["passed"] else "FAIL")
    print(f"   detector rates : {np.round(t2['detector_rates'], 3).tolist()}")
    print(f"   model predicts : {np.round(t2['predicted_rates'], 3).tolist()}")
    print(f"   median measured/predicted = {t2['median_measured_over_predicted']:.2f}, "
          f"raw logical flip rate = {t2['raw_logical_flip_rate']:.3f}")

    (run_dir / "results.json").write_text(json.dumps({"known_states": t1, "one_round": t2},
                                                     indent=1, default=str))
    print(f"\n[smoke] saved to {run_dir}/")
    if not (t1["passed"] and t2["passed"]):
        print("[smoke] DO NOT start the main runs: fix the data path first.")


if __name__ == "__main__":
    main()
