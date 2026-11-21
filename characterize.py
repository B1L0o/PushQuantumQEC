# -*- coding: utf-8 -*-
"""
characterize.py

Calibration -> characterization loop for the Emerald noise model.

    1. Start from the CALIBRATED model (EmeraldNoiseModel.from_iqm(token)).
       It ranks layouts (layout_search) and weights the decoder.
    2. Run the surface code on hardware (one layout, several round counts).
    3. characterize(): measure the decoding-graph edge probabilities p_ij from
       the detection events (estimate_dem), then fit one error multiplier per
       physical qubit so the model reproduces them. Qubits that were not
       measured get the typical (median) multiplier.
    4. The CHARACTERIZED model replaces the calibrated one: re-rank layouts,
       re-simulate, decode. Repeat on other patches to characterize more qubits.

What a multiplier means: s_q = 2 says "every error involving qubit q (gates,
readout, idling) happens ~2x more often than calibration says". It absorbs
crosstalk, drift and anything else the calibration missed, attributed to the
qubit where it shows up.

Limits: graph-like Pauli model, one parameter per qubit (it cannot tell a bad
CZ from a bad readout on the same qubit), needs >= ~20k shots for stable
p_ij. Always compare predicted and measured LER on held-out shots.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import stim
from scipy.optimize import least_squares

from emerald_noise import EmeraldNoiseModel
from estimate_dem import estimate_edge_probabilities, graph_structure


@dataclass
class CharacterizationResult:
    model: EmeraldNoiseModel                  # characterized noise model
    scales: dict[str, float]                  # fitted multiplier per measured qubit
    default_scale: float                      # multiplier used for unmeasured qubits
    rms_log_error_before: float               # model vs measured p_ij, log space
    rms_log_error_after: float
    n_edges: int
    details: dict = field(default_factory=dict)

    def summary(self) -> str:
        worst = sorted(self.scales.items(), key=lambda kv: -kv[1])[:5]
        return (f"characterized {len(self.scales)} qubits on {self.n_edges} edges | "
                f"rms log-error {self.rms_log_error_before:.2f} -> "
                f"{self.rms_log_error_after:.2f} | default x{self.default_scale:.2f} | "
                f"worst: " + ", ".join(f"{q} x{s:.2f}" for q, s in worst))


def _edge_probs(model: EmeraldNoiseModel, circuit: stim.Circuit,
                stim_to_name: dict[int, str]):
    dem = model.apply(circuit, stim_to_name).detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=True)
    s = graph_structure(dem)
    return s


def characterize(model: EmeraldNoiseModel,
                 runs: Sequence[tuple[stim.Circuit, np.ndarray]],
                 stim_to_name: dict[int, str],
                 min_p: float = 2e-3,
                 regularization: float = 0.05,
                 unmeasured: str = "median",
                 bias_correction: bool = True,
                 bias_shots: int = 200_000,
                 verbose: bool = True) -> CharacterizationResult:
    """Fit per-qubit error multipliers so `model` reproduces measured p_ij.

    model        : calibrated EmeraldNoiseModel (the starting point).
    runs         : [(noiseless circuit that ran, det_events), ...], e.g. one per
                   round count, ALL on the same layout. Use TRAINING shots only.
    stim_to_name : the layout that ran ({stim qubit -> "QB27"}).
    min_p        : edges whose measured p is below this are ignored (too noisy).
    regularization : pull of log(s_q) towards 0 (calibration), avoids
                   over-fitting qubits the data barely constrains.
    unmeasured   : "median" -> other qubits get the median fitted multiplier;
                   "calibration" -> they keep multiplier 1.
    bias_correction : correct the p_ij estimator's systematic bias using
                   `bias_shots` shots simulated from `model` (recommended).
    """
    qubits = sorted(set(stim_to_name.values()), key=lambda s: int(s[2:]) if s[2:].isdigit() else s)

    # measured edge probabilities, per run, corrected for the estimator's own
    # bias (hyperedges, boundary approximation): bias_k = estimate on data
    # simulated from the model / the model's exact edge probability.
    targets = []
    for i, (circuit, dets) in enumerate(runs):
        s = _edge_probs(model, circuit, stim_to_name)
        e_meas, b_meas = estimate_edge_probabilities(np.asarray(dets, dtype=bool), s)
        if bias_correction:
            sim = model.apply(circuit, stim_to_name).compile_detector_sampler(
                seed=1234 + i).sample(bias_shots)
            e_sim, b_sim = estimate_edge_probabilities(sim, s)
            for k in e_meas:
                e_meas[k] *= s.edges[k] / e_sim[k]
            for k in b_meas:
                b_meas[k] *= s.boundary[k] / b_sim[k]
        keys = [("e", k) for k, p in e_meas.items() if p >= min_p] + \
               [("b", k) for k, p in b_meas.items() if p >= min_p]
        meas = np.array([e_meas[k] if t == "e" else b_meas[k] for t, k in keys])
        targets.append((circuit, keys, np.log(meas)))

    def predicted(m: EmeraldNoiseModel):
        out = []
        for circuit, keys, _ in targets:
            s = _edge_probs(m, circuit, stim_to_name)
            out.append(np.log(np.array([
                max(1e-9, s.edges.get(k, 0.0) if t == "e" else s.boundary.get(k, 0.0))
                for t, k in keys])))
        return out

    log_meas = np.concatenate([t[2] for t in targets])

    def residuals(x):
        m = model.with_qubit_scales(dict(zip(qubits, np.exp(x))), default_scale=1.0)
        r = np.concatenate(predicted(m)) - log_meas
        return np.concatenate([r, math.sqrt(regularization) * x])

    x0 = np.zeros(len(qubits))
    r0 = residuals(x0)[: len(log_meas)]
    sol = least_squares(residuals, x0, diff_step=1e-3, x_scale=1.0, max_nfev=200)
    r1 = sol.fun[: len(log_meas)]

    scales = {q: float(np.exp(v)) for q, v in zip(qubits, sol.x)}
    default = float(np.median(list(scales.values()))) if unmeasured == "median" else 1.0
    char_model = model.with_qubit_scales(
        scales, default_scale=default,
        source=f"{model.source} + characterization on {qubits[0]}..{qubits[-1]}")

    res = CharacterizationResult(
        model=char_model, scales=scales, default_scale=default,
        rms_log_error_before=float(np.sqrt(np.mean(r0 ** 2))),
        rms_log_error_after=float(np.sqrt(np.mean(r1 ** 2))),
        n_edges=len(log_meas),
        details=dict(nfev=sol.nfev, success=bool(sol.success)),
    )
    if verbose:
        print("[characterize] " + res.summary())
    return res


def readout_from_smoke_test(model: EmeraldNoiseModel, smoke_dir: str,
                            rounds: int = 2, verbose: bool = True) -> EmeraldNoiseModel:
    """Replace calibration readout numbers by the ones MEASURED in smoke_test.py.

    From the known-state test (two complementary patterns, each qubit measured
    as in the d=3 no-reset circuit with `rounds` rounds):
      readout error   p_ro   = (P(1|0) + P(0|1) at the 1st measurement) / 2
      post-measure    p_post = max(0, P(0|1) 2nd - P(0|1) 1st) / 2
    The /2 turns an effect that only hits |1> into Stim's symmetric flip
    (a no-reset ancilla is in |1> about half the time). Both are encoded via
    readout_fidelity and qnd_fidelity (p_post = 1 - qnd / readout).

    WARNING (checked on Emerald, 2026-11-21): this OVER-estimates errors
    (1-round prediction 0.58x of measured, vs 1.75x with calibration only).
    In the known-state circuit a qubit in |1> waits through all earlier
    measurements, so P(0|1) mixes readout error with T1 decay that depends on
    that circuit's timing, not the surface code's. Use it as an upper bound;
    prefer characterize() on the real surface-code runs.
    """
    import json, pathlib
    from internal_helpers import counts_to_measurement_array, get_meas_order
    from surface_code import make_stim_circuit

    d = pathlib.Path(smoke_dir)
    lay = json.loads((d / "layout.json").read_text())
    res = json.loads((d / "results.json").read_text())["known_states"]
    name = {int(k): v for k, v in lay["stim_to_name"].items()}
    mo = get_meas_order(make_stim_circuit(3, rounds))
    qubits = sorted(name)
    A = set(qubits[::2])
    first01, second01, err10 = {}, {}, {}
    for counts, ones in zip(res["counts"], (A, set(qubits) - A)):
        shots = sum(counts.values())
        raw = counts_to_measurement_array(counts, len(mo), shots)
        for q in qubits:
            cols = [j for j, m in enumerate(mo) if m == q]
            p0 = [float(1 - raw[:, j].mean()) for j in cols]   # P(read 0) per occurrence
            if q in ones:
                first01[q] = p0[0]
                if len(p0) > 1:
                    second01[q] = p0[1]
            else:
                err10[q] = float(np.mean([1 - p for p in p0]))
    out = EmeraldNoiseModel(**{k: v for k, v in __import__("dataclasses").asdict(model).items()})
    for q in qubits:
        n = name[q]
        p_ro = (err10[q] + first01[q]) / 2
        p_post = max(0.0, second01.get(q, first01[q]) - first01[q]) / 2
        out.readout_fidelity[n] = 1 - p_ro
        out.qnd_fidelity[n] = (1 - p_ro) * (1 - p_post)
    out.source = f"{model.source} + measured readout ({d.name})"
    if verbose:
        print(f"[readout] {len(qubits)} qubits: median readout error "
              f"{np.median([(err10[q] + first01[q]) / 2 for q in qubits]):.3f} "
              f"(calibration {1 - model.median['readout_fidelity']:.3f})")
    return out
