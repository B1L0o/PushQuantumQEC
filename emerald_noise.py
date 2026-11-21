# -*- coding: utf-8 -*-
"""
emerald_noise.py

Calibration-driven circuit-level noise model for IQM Emerald, applied to a
NOISELESS Stim circuit (e.g. make_stim_circuit(..., noise=None)).

What is modelled (all Pauli channels, since Stim is a Clifford simulator)
-----------------------------------------------------------------------
  * 1q gates (H, PRX, ...)  : DEPOLARIZE1, from the per-qubit PRX fidelity.
  * 2q gates (CZ, CX)       : DEPOLARIZE2, from the per-pair CZ fidelity.
                              A CX is compiled to H.CZ.H on hardware, so it also
                              gets 2 PRX errors on its target.
  * Idling                  : Pauli-twirled amplitude damping + dephasing
                              from T1 / T2 and the real layer duration. Applied
                              to every qubit that is idle (or idle for part of)
                              a TICK layer, the readout layer being the longest.
  * Readout                 : X_ERROR before M (Z_ERROR before MX), from the
                              per-qubit readout fidelity (symmetric approximation
                              of the asymmetric 0->1 / 1->0 errors).
  * Post-measurement flip   : optional, from QNDness. Matters in the NO-RESET
                              experiment, where the ancilla state after a
                              measurement is carried into the next round.
  * Reset / initialisation  : X_ERROR after R / MR (Z_ERROR after RX).

Fidelity -> Pauli probability conversion
----------------------------------------
IQM reports average gate fidelities F (RB / IRB). For a depolarizing channel on
n qubits (d = 2^n):  p = (1 - F) * (d + 1) / d   ->  1q: 1.5 (1-F),  2q: 1.25 (1-F).
RB fidelities already include decoherence DURING the gate, so idle noise is only
added for the time a qubit is NOT busy inside a layer (no double counting).

Known limitations (state them in your report)
---------------------------------------------
  * Leakage, crosstalk, coherent errors, non-Markovian drift: not modelled.
  * T1 decay is biased (|1> -> |0>); Pauli twirling makes it symmetric.
  * Readout asymmetry is averaged.
  * The model must be VALIDATED against hardware (per-detector firing rates,
    P_L vs rounds) before using it for decoder weights or extrapolation.

Typical usage
-------------
    from emerald_noise import EmeraldNoiseModel
    nm   = EmeraldNoiseModel.from_iqm(token=TOKEN)           # live calibration
    nm.save("calib_emerald_<date>.json")                     # keep the snapshot!
    noisy = nm.apply(stim_circ, qubit_map={stim_q: "QB27", ...})
    dem   = noisy.detector_error_model(decompose_errors=True)
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import Mapping, Optional

import numpy as np
import stim


# ─────────────────────────────────────────────────────────────────────────────
#  Default (median) values. REPLACE with live calibration via from_iqm().
# ─────────────────────────────────────────────────────────────────────────────
#  Sources / status:
#   prx_fidelity 0.9993, cz_fidelity 0.995 : IQM's published Emerald medians.
#   t1 54.7 us, t2 16.6 us                 : median reported in a third-party
#                                            Emerald experiment (arXiv 2608.30176).
#   readout_fidelity 0.975                 : ASSUMPTION (Garnet-like, ~2.5 %).
#   durations prx 20 ns, cz 40 ns          : IQM fake Garnet backend values.
#   measure 1.5 us                         : ASSUMPTION - get the real value
#                                            from the backend / mentors.
EMERALD_MEDIANS = dict(
    t1_us=54.7,
    t2_us=16.6,
    prx_fidelity=0.9993,
    cz_fidelity=0.995,
    readout_fidelity=0.975,
    qnd_fidelity=None,      # None -> no post-measurement flip
    init_error=0.0,         # initialisation error at the start of each shot
)

DEFAULT_DURATIONS_NS = dict(prx=20.0, cz=40.0, measure=1500.0, reset=0.0)


_NOISE_GATES = {
    "DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR",
    "PAULI_CHANNEL_1", "PAULI_CHANNEL_2", "E", "ELSE_CORRELATED_ERROR",
    "HERALDED_ERASE", "HERALDED_PAULI_CHANNEL_1",
}
_ANNOTATIONS = {"DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS", "MPAD"}
_ONE_Q = {"H", "S", "S_DAG", "X", "Y", "Z", "SQRT_X", "SQRT_X_DAG", "SQRT_Y",
          "SQRT_Y_DAG", "H_YZ", "H_XY", "C_XYZ", "C_ZYX", "I"}
_TWO_Q = {"CX", "CNOT", "CZ", "CY", "XCZ", "YCZ", "ZCX", "ZCZ"}
_MEAS = {"M": "X_ERROR", "MZ": "X_ERROR", "MX": "Z_ERROR", "MY": "X_ERROR"}
_MEAS_RESET = {"MR": ("X_ERROR", "X_ERROR"), "MRZ": ("X_ERROR", "X_ERROR"),
               "MRX": ("Z_ERROR", "Z_ERROR"), "MRY": ("X_ERROR", "X_ERROR")}
_RESET = {"R": "X_ERROR", "RZ": "X_ERROR", "RX": "Z_ERROR", "RY": "X_ERROR"}


def _pair_key(a: str, b: str) -> str:
    return "|".join(sorted((a, b)))


def idle_pauli_probs(t_ns: float, t1_us: float, t2_us: float) -> tuple[float, float, float]:
    """Pauli-twirled amplitude damping + dephasing for an idle of t_ns.

    px = py = (1 - e^{-t/T1}) / 4
    pz      = (1 - e^{-t/T2}) / 2 - (1 - e^{-t/T1}) / 4
    """
    if t_ns <= 0:
        return 0.0, 0.0, 0.0
    t_us = t_ns / 1000.0
    t2_us = min(t2_us, 2 * t1_us)  # physical bound T2 <= 2 T1
    a = 1.0 - math.exp(-t_us / t1_us)
    px = py = a / 4.0
    pz = max(0.0, (1.0 - math.exp(-t_us / t2_us)) / 2.0 - a / 4.0)
    return px, py, pz


@dataclass
class EmeraldNoiseModel:
    """Per-qubit / per-pair noise parameters + gate durations.

    Keys are physical qubit names ("QB1", ...). Pair keys are "QBa|QBb" (sorted).
    Any missing entry falls back to `median`.
    """

    t1_us: dict[str, float] = field(default_factory=dict)
    t2_us: dict[str, float] = field(default_factory=dict)
    prx_fidelity: dict[str, float] = field(default_factory=dict)
    readout_fidelity: dict[str, float] = field(default_factory=dict)
    qnd_fidelity: dict[str, float] = field(default_factory=dict)
    cz_fidelity: dict[str, float] = field(default_factory=dict)
    durations_ns: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_DURATIONS_NS))
    median: dict = field(default_factory=lambda: dict(EMERALD_MEDIANS))
    source: str = "defaults (published medians + assumptions)"
    # Characterization layer (see characterize.py): every error rate of qubit q
    # is multiplied by qubit_scale.get(q, default_scale) (coherence times are
    # divided by it; a CZ on (a, b) uses sqrt(s_a * s_b)). 1.0 = calibration only.
    qubit_scale: dict[str, float] = field(default_factory=dict)
    default_scale: float = 1.0

    # ── constructors ────────────────────────────────────────────────────────
    @classmethod
    def from_metrics(cls, full_metrics, durations_ns: Optional[dict] = None,
                     source: str = "iqm calibration") -> "EmeraldNoiseModel":
        """Build from iqm.qubit_selector FullMetricsData (iqm-qubit-selector >= 2.0)."""
        cz = {}
        for key, fid in full_metrics.cz.items():
            a, b = [s.strip(" '") for s in key.strip("[]").split(",")]
            cz[_pair_key(a, b)] = fid
        nm = cls(
            t1_us=dict(full_metrics.t1),
            t2_us=dict(full_metrics.t2),
            prx_fidelity=dict(full_metrics.single_qubit),
            readout_fidelity=dict(full_metrics.readout),
            qnd_fidelity=dict(full_metrics.readout_qndness),
            cz_fidelity=cz,
            durations_ns=dict(durations_ns or DEFAULT_DURATIONS_NS),
            source=source,
        )
        nm._update_medians()
        return nm

    @classmethod
    def from_iqm(cls, token: str, quantum_computer: str = "emerald",
                 server_url: str = "https://resonance.iqm.tech",
                 calibration_set_id=None,
                 durations_ns: Optional[dict] = None) -> "EmeraldNoiseModel":
        """Fetch the current (or a given) calibration set from IQM Resonance.

        Uses iqm-qubit-selector >= 2.0 when installed, otherwise parses the
        quality metrics itself (works with the pinned iqm-client 35.0.3, which
        no iqm-qubit-selector release supports).
        """
        from iqm.iqm_client import IQMClient

        client = IQMClient(server_url, quantum_computer=quantum_computer, token=token)
        try:
            from iqm.qubit_selector.core import MetricsDataManager
            full = MetricsDataManager().get_full_metrics_data(client, calibration_set_id)
            cal_id = calibration_set_id
        except ImportError:
            full, cal_id = _metrics_from_client(client, calibration_set_id)
        return cls.from_metrics(full, durations_ns,
                                source=f"{quantum_computer} calibration set "
                                       f"{cal_id or calibration_set_id or 'default'}")

    @classmethod
    def from_iqm_error_profile(cls, profile) -> "EmeraldNoiseModel":
        """Build from an IQMErrorProfile, e.g. IQMFakeAphrodite().error_profile.

        Useful offline (no token): realistic, non-uniform per-qubit values for
        testing layout search and decoders. Aer depolarizing parameter lam on n
        qubits (d = 2^n) -> average gate fidelity F = 1 - lam (d - 1) / d.
        """
        prx = next(iter(profile.single_qubit_gate_depolarizing_error_parameters.values()))
        cz = next(iter(profile.two_qubit_gate_depolarizing_error_parameters.values()))
        durations = dict(DEFAULT_DURATIONS_NS)
        durations["prx"] = float(next(iter(profile.single_qubit_gate_durations.values())))
        durations["cz"] = float(next(iter(profile.two_qubit_gate_durations.values())))
        nm = cls(
            t1_us={q: v / 1000.0 for q, v in profile.t1s.items()},
            t2_us={q: v / 1000.0 for q, v in profile.t2s.items()},
            prx_fidelity={q: 1.0 - lam / 2.0 for q, lam in prx.items()},
            readout_fidelity={q: 1.0 - (e["0"] + e["1"]) / 2.0
                              for q, e in profile.readout_errors.items()},
            cz_fidelity={_pair_key(a, b): 1.0 - 0.75 * lam for (a, b), lam in cz.items()},
            durations_ns=durations,
            source=f"IQM error profile '{profile.name}'",
        )
        nm._update_medians()
        return nm

    @classmethod
    def uniform(cls, **overrides) -> "EmeraldNoiseModel":
        """Same parameters on every qubit (medians, optionally overridden)."""
        nm = cls()
        nm.median.update({k: v for k, v in overrides.items() if k in nm.median})
        nm.durations_ns.update({k: v for k, v in overrides.items() if k in nm.durations_ns})
        nm.source = "uniform"
        return nm

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=1)

    @classmethod
    def load(cls, path: str) -> "EmeraldNoiseModel":
        with open(path) as f:
            return cls(**json.load(f))

    # ── parameter helpers ───────────────────────────────────────────────────
    def _update_medians(self) -> None:
        for name, d in (("t1_us", self.t1_us), ("t2_us", self.t2_us),
                        ("prx_fidelity", self.prx_fidelity),
                        ("readout_fidelity", self.readout_fidelity),
                        ("qnd_fidelity", self.qnd_fidelity),
                        ("cz_fidelity", self.cz_fidelity)):
            if d:
                self.median[name] = statistics.median(d.values())

    def scaled(self, factor: float) -> "EmeraldNoiseModel":
        """Return a copy with every error probability multiplied by `factor`
        (coherence times divided by it). Use for threshold sweeps / sensitivity."""
        def f_err(fid):  # scale an infidelity
            return None if fid is None else max(0.0, 1.0 - factor * (1.0 - fid))
        m = dict(self.median)
        for k in ("prx_fidelity", "cz_fidelity", "readout_fidelity", "qnd_fidelity"):
            m[k] = f_err(m[k])
        m["t1_us"] /= factor
        m["t2_us"] /= factor
        m["init_error"] *= factor
        return EmeraldNoiseModel(
            t1_us={q: v / factor for q, v in self.t1_us.items()},
            t2_us={q: v / factor for q, v in self.t2_us.items()},
            prx_fidelity={q: f_err(v) for q, v in self.prx_fidelity.items()},
            readout_fidelity={q: f_err(v) for q, v in self.readout_fidelity.items()},
            qnd_fidelity={q: f_err(v) for q, v in self.qnd_fidelity.items()},
            cz_fidelity={p: f_err(v) for p, v in self.cz_fidelity.items()},
            durations_ns=dict(self.durations_ns), median=m,
            source=f"{self.source} x{factor}",
            qubit_scale=dict(self.qubit_scale), default_scale=self.default_scale,
        )

    def with_qubit_scales(self, scales: Mapping[str, float],
                          default_scale: Optional[float] = None,
                          source: Optional[str] = None) -> "EmeraldNoiseModel":
        """Copy of this model with per-qubit error multipliers (characterization).

        scales        : {"QB20": 2.1, ...} fitted on the qubits that were measured.
        default_scale : multiplier for every other qubit (e.g. the typical
                        measured/calibration ratio); None keeps the current one.
        """
        out = EmeraldNoiseModel(**{k: v for k, v in asdict(self).items()})
        out.qubit_scale = dict(scales)
        if default_scale is not None:
            out.default_scale = float(default_scale)
        out.source = source or f"{self.source} + characterization"
        return out

    def _s(self, q: str) -> float:
        return self.qubit_scale.get(q, self.default_scale)

    def sample_qubits(self, names, rng=None) -> "EmeraldNoiseModel":
        """Return a model for arbitrary qubit names (e.g. a d=11 patch that does
        not fit on Emerald) by drawing each qubit's / pair's parameters from the
        empirical distribution of this calibration. Pairs are drawn on demand
        in apply() via the median, so call this, then `assign_pairs()`."""
        rng = rng or np.random.default_rng()
        out = EmeraldNoiseModel(durations_ns=dict(self.durations_ns),
                                median=dict(self.median),
                                source=f"sampled from {self.source}",
                                default_scale=self.default_scale)
        for attr in ("t1_us", "t2_us", "prx_fidelity", "readout_fidelity", "qnd_fidelity"):
            pool = list(getattr(self, attr).values())
            if pool:
                getattr(out, attr).update({n: float(rng.choice(pool)) for n in names})
        out._cz_pool = list(self.cz_fidelity.values())
        out._rng = rng
        return out

    def _get(self, attr: str, q: str):
        return getattr(self, attr).get(q, self.median[attr])

    def _cz_fid(self, a: str, b: str) -> float:
        key = _pair_key(a, b)
        if key not in self.cz_fidelity and getattr(self, "_cz_pool", None):
            self.cz_fidelity[key] = float(self._rng.choice(self._cz_pool))
        return self.cz_fidelity.get(key, self.median["cz_fidelity"])

    def p1(self, q: str) -> float:
        return self._s(q) * 1.5 * (1.0 - self._get("prx_fidelity", q))

    def p2(self, a: str, b: str) -> float:
        return math.sqrt(self._s(a) * self._s(b)) * 1.25 * (1.0 - self._cz_fid(a, b))

    def p_readout(self, q: str) -> float:
        return self._s(q) * (1.0 - self._get("readout_fidelity", q))

    def p_post_measure(self, q: str) -> float:
        """Heuristic: QND fidelity ~ F_readout * (1 - p_post)."""
        qnd = self._get("qnd_fidelity", q)
        if qnd is None:
            return 0.0
        ro = self._get("readout_fidelity", q)
        return self._s(q) * max(0.0, 1.0 - qnd / ro)

    def t1_t2(self, q: str) -> tuple[float, float]:
        s = self._s(q)
        return self._get("t1_us", q) / s, self._get("t2_us", q) / s

    # ── circuit transformation ──────────────────────────────────────────────
    def apply(self, circuit: stim.Circuit,
              qubit_map: Optional[Mapping[int, str]] = None,
              idle_noise: bool = True) -> stim.Circuit:
        """Return `circuit` with Emerald noise inserted.

        circuit    : noiseless Stim circuit with TICKs between layers
                     (stim.Circuit.generated and most generators do this).
        qubit_map  : Stim qubit index -> physical name ("QB27"). If None, every
                     qubit gets the median parameters (uniform model).
        idle_noise : add T1/T2 noise to idle qubits in every layer.
        """
        for inst in circuit.flattened():
            if inst.name in _NOISE_GATES:
                raise ValueError("Input circuit already contains noise "
                                 f"({inst.name}); pass a noiseless circuit.")
        name = (lambda q: qubit_map.get(q, f"Q{q}")) if qubit_map else (lambda q: f"Q{q}")
        all_qubits = sorted(_used_qubits(circuit))
        return self._apply_block(circuit, name, all_qubits, idle_noise)

    def _apply_block(self, block: stim.Circuit, name, all_qubits, idle_noise) -> stim.Circuit:
        out = stim.Circuit()
        layer: list[stim.CircuitInstruction] = []

        def flush():
            if layer:
                out.__iadd__(self._noisy_layer(layer, name, all_qubits, idle_noise))
                layer.clear()

        for inst in block:
            if isinstance(inst, stim.CircuitRepeatBlock):
                flush()
                body = self._apply_block(inst.body_copy(), name, all_qubits, idle_noise)
                out.append(stim.CircuitRepeatBlock(inst.repeat_count, body))
            elif inst.name == "TICK":
                flush()
                out.append("TICK")
            else:
                layer.append(inst)
        flush()
        return out

    def _noisy_layer(self, layer, name, all_qubits, idle_noise) -> stim.Circuit:
        d = self.durations_ns
        busy: dict[int, float] = {}         # qubit -> busy time in this layer (ns)
        pre = stim.Circuit()                # noise before ops (readout flips)
        body = stim.Circuit()               # the ops themselves
        post = stim.Circuit()               # gate / reset noise after ops

        for inst in layer:
            body.append(inst)
            g = inst.name
            qs = [t.value for t in inst.targets_copy() if t.is_qubit_target]
            if g in _ANNOTATIONS or not qs:
                continue
            if g in _ONE_Q:
                for q in qs:
                    _add(post, "DEPOLARIZE1", [q], self.p1(name(q)))
                    busy[q] = max(busy.get(q, 0.0), d["prx"])
            elif g in _TWO_Q:
                for a, b in zip(qs[::2], qs[1::2]):
                    _add(post, "DEPOLARIZE2", [a, b], self.p2(name(a), name(b)))
                    t = d["cz"]
                    if g not in ("CZ", "ZCZ"):     # CX etc. = H . CZ . H on target
                        _add(post, "DEPOLARIZE1", [b], 2 * self.p1(name(b)))
                        t += 2 * d["prx"]
                    busy[a] = max(busy.get(a, 0.0), t)
                    busy[b] = max(busy.get(b, 0.0), t)
            elif g in _MEAS:
                for q in qs:
                    _add(pre, _MEAS[g], [q], self.p_readout(name(q)))
                    _add(post, _MEAS[g], [q], self.p_post_measure(name(q)))
                    busy[q] = d["measure"]
            elif g in _MEAS_RESET:
                e_meas, e_reset = _MEAS_RESET[g]
                for q in qs:
                    _add(pre, e_meas, [q], self.p_readout(name(q)))
                    _add(post, e_reset, [q], self.median["init_error"])
                    busy[q] = d["measure"] + d["reset"]
            elif g in _RESET:
                for q in qs:
                    _add(post, _RESET[g], [q], self.median["init_error"])
                    busy[q] = d["reset"]
            else:
                raise ValueError(f"Gate {g} not handled by the Emerald noise model.")

        # Idle noise: every qubit idles for (layer duration - its own busy time).
        idle = stim.Circuit()
        if idle_noise and busy:
            t_layer = max(busy.values())
            for q in all_qubits:
                t_idle = t_layer - busy.get(q, 0.0)
                px, py, pz = idle_pauli_probs(t_idle, *self.t1_t2(name(q)))
                if px + py + pz > 0:
                    idle.append("PAULI_CHANNEL_1", [q], [px, py, pz])

        return pre + body + post + idle


@dataclass
class _Metrics:
    """Same fields / key formats as iqm.qubit_selector FullMetricsData."""
    cz: dict = field(default_factory=dict)
    single_qubit: dict = field(default_factory=dict)
    readout: dict = field(default_factory=dict)
    readout_qndness: dict = field(default_factory=dict)
    t1: dict = field(default_factory=dict)
    t2: dict = field(default_factory=dict)


def _metrics_from_client(client, calibration_set_id=None):
    """Parse a calibration set's quality metrics with iqm-client alone.

    Field names look like 'metrics.ssro.measure.<impl>.QB1.fidelity',
    'gates.cz.<impl>.QB1__QB2.fidelity', '...t1_time.QB3...' (same rules as
    iqm-qubit-selector 2.0's MetricsDataManager). Returns (metrics, cal_set_id).
    """
    import re
    arch = client.get_dynamic_quantum_architecture(calibration_set_id)
    obs = client.get_quality_metric_set(arch.calibration_set_id).observations
    comps = set(arch.components)

    def impls(gate):
        g = arch.gates.get(gate)
        return set(g.implementations) if g is not None else set()

    def default_impl(gate):
        g = arch.gates.get(gate)
        return g.default_implementation if g is not None else None

    def loci(field_name):
        return [t for t in re.split(r"\.|__", field_name) if t in comps]

    prx_impls, cz_impls, meas_impl = impls("prx"), impls("cz"), default_impl("measure")
    m = _Metrics()
    for o in obs:
        if not isinstance(o.value, (int, float)):
            continue
        f, v = o.dut_field, float(o.value)
        names = loci(f)
        if not names:
            continue
        q = names[0]
        if "metrics.rrc." in f and "measure" in f and ".qndness" in f:
            m.readout_qndness[q] = v
        elif ".fidelity" in f and ("measure_fidelity" in f or "ssro" in f) and \
                (meas_impl is None or meas_impl in f):
            m.readout[q] = v
        elif "prx" in f and any(i in f for i in prx_impls) and ".fidelity" in f:
            m.single_qubit[q] = v
        elif "t1" in f:
            m.t1[q] = v * 1e6          # seconds -> microseconds
        elif "t2" in f:
            m.t2[q] = v * 1e6
        elif ".fidelity" in f and "cz" in f and any(i in f for i in cz_impls) \
                and "clifford" not in f and len(names) >= 2:
            m.cz[str([names[0], names[1]])] = v
    return m, arch.calibration_set_id


def _add(circ: stim.Circuit, gate: str, qs, p: float) -> None:
    if p > 0:
        circ.append(gate, qs, min(p, 0.5 if "DEPOLARIZE" not in gate else 0.75))


def _used_qubits(circuit: stim.Circuit) -> set[int]:
    used = set()
    for inst in circuit.flattened():
        if inst.name in _ANNOTATIONS or inst.name == "TICK":
            continue
        used.update(t.value for t in inst.targets_copy() if t.is_qubit_target)
    return used
