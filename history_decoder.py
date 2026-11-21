# -*- coding: utf-8 -*-
"""
history_decoder.py

Decoders that use the measurement HISTORY of the no-reset circuit.

Why: without reset an ancilla enters round k in the state it was measured in
at round k-1 (m_(k-1)). On Emerald, errors depend strongly on that state
(measured: a detector fires up to 3x more often when its ancilla read 1 just
before), which a static DEM / MWPM cannot represent.

Two decoders:

  HistoryAwareMWPM   classical. Edge probabilities are estimated separately for
                     each context (state the edge's ancilla entered the round
                     in: 0 or 1) on TRAINING shots; every test shot is decoded
                     with its own matching graph built from its context.
  LearnedDecoder     gradient-boosted classifier on the detection events,
                     trained on labelled hardware shots (obs_flips). It needs
                     no noise model, so it is not limited by the DEM format.

Note: the ancilla raw outcomes are a deterministic function of the detection
events in the no-reset circuit (m_k = d_k xor m_(k-2)), so they add no
information to a nonlinear decoder; they ARE needed by HistoryAwareMWPM to
pick each shot's weights. NEVER give the final data-qubit readout to a decoder
as a feature: the logical observable is a parity of those bits.
"""

from __future__ import annotations

import math

import numpy as np
import pymatching
import stim

from estimate_dem import P_MAX, P_MIN, estimate_edge_probabilities, graph_structure
from internal_helpers import get_meas_order, get_qubit_lists


# ─────────────────────────────────────────────────────────────────────────────
#  Context: state each detector's ancilla entered its round in
# ─────────────────────────────────────────────────────────────────────────────

def detector_context(circuit: stim.Circuit, raw_meas: np.ndarray) -> np.ndarray:
    """(shots, num_detectors) array: raw outcome of the detector's ancilla in the
    PREVIOUS round (0 for the first round and for final data-qubit detectors)."""
    mo = get_meas_order(circuit)
    _, anc = get_qubit_lists(circuit)
    qcoords = circuit.get_final_qubit_coordinates()
    by_xy = {tuple(qcoords[a][:2]): a for a in anc}
    meas_cols = {a: [j for j, q in enumerate(mo) if q == a] for a in anc}
    ctx = np.zeros((raw_meas.shape[0], circuit.num_detectors), dtype=bool)
    n_rounds = max(len(v) for v in meas_cols.values())
    for det, (x, y, t) in circuit.get_detector_coordinates().items():
        a = by_xy.get((x, y))
        t = int(round(t))
        if a is None or t < 1 or t >= n_rounds:
            continue                      # first round or final data-qubit detectors
        ctx[:, det] = raw_meas[:, meas_cols[a][t - 1]].astype(bool)
    return ctx


# ─────────────────────────────────────────────────────────────────────────────
#  Classical: history-aware MWPM
# ─────────────────────────────────────────────────────────────────────────────

class HistoryAwareMWPM:
    """MWPM whose edge weights depend on each shot's ancilla history.

    The context of an edge is the context of its lower-index detector (for a
    boundary edge, of its detector): 0 or 1. Probabilities per context are
    estimated with the pairwise-correlation method on the training shots of
    that context, falling back to the model DEM when an estimate is invalid.
    """

    def __init__(self, model_dem: stim.DetectorErrorModel, circuit: stim.Circuit):
        self.s = graph_structure(model_dem)
        self.circuit = circuit
        self.p = {}                       # context -> (edges, boundary)

    def fit(self, det_train: np.ndarray, raw_train: np.ndarray, min_shots: int = 500):
        ctx = detector_context(self.circuit, raw_train)
        base_e, base_b = estimate_edge_probabilities(det_train, self.s, fallback_to_prior=True)
        self.p_ctx_e, self.p_ctx_b = {}, {}
        for c in (0, 1):
            e_c, b_c = {}, {}
            for (i, j) in self.s.edges:
                sel = ctx[:, i] == c
                e_c[(i, j)] = self._est_pair(det_train[sel], i, j, base_e[(i, j)], min_shots)
            for i in self.s.boundary:
                sel = ctx[:, i] == c
                b_c[i] = self._est_boundary(det_train[sel], i, e_c, base_b[i], min_shots)
            self.p_ctx_e[c], self.p_ctx_b[c] = e_c, b_c
        return self

    @staticmethod
    def _est_pair(d, i, j, prior, min_shots):
        if len(d) < min_shots:
            return prior
        xi, xj = d[:, i].mean(), d[:, j].mean()
        xij = np.mean(d[:, i] & d[:, j])
        den = 1 - 2 * xi - 2 * xj + 4 * xij
        if den <= 0:
            return prior
        arg = 1 - 4 * (xij - xi * xj) / den
        p = 0.5 - 0.5 * math.sqrt(arg) if arg >= 0 else float("nan")
        return float(min(p, P_MAX)) if np.isfinite(p) and p > P_MIN else prior

    def _est_boundary(self, d, i, e_c, prior, min_shots):
        if len(d) < min_shots:
            return prior
        prod = 1.0
        for (a, b), p in e_c.items():
            if i in (a, b):
                prod *= 1 - 2 * p
        p = 0.5 * (1 - (1 - 2 * d[:, i].mean()) / prod)
        return float(min(p, P_MAX)) if np.isfinite(p) and p > P_MIN else prior

    def decode(self, det: np.ndarray, raw: np.ndarray) -> np.ndarray:
        ctx = detector_context(self.circuit, raw)
        out = np.zeros((len(det), self.s.num_observables), dtype=bool)
        cache = {}
        for k in range(len(det)):
            key = ctx[k].tobytes()
            m = cache.get(key)
            if m is None:
                m = pymatching.Matching()
                for (i, j) in self.s.edges:
                    p = self.p_ctx_e[int(ctx[k, i])][(i, j)]
                    m.add_edge(i, j, fault_ids=set(self.s.edge_obs[(i, j)]),
                               weight=math.log((1 - p) / p), error_probability=p,
                               merge_strategy="independent")
                for i in self.s.boundary:
                    p = self.p_ctx_b[int(ctx[k, i])][i]
                    m.add_boundary_edge(i, fault_ids=set(self.s.boundary_obs[i]),
                                        weight=math.log((1 - p) / p), error_probability=p,
                                        merge_strategy="independent")
                covered = set(self.s.boundary) | {x for e in self.s.edges for x in e}
                for dd in range(self.s.num_detectors):
                    if dd not in covered:
                        m.add_boundary_edge(dd, weight=math.log((1 - P_MIN) / P_MIN))
                if len(cache) < 20_000:
                    cache[key] = m
            out[k] = m.decode(det[k].astype(np.uint8)).astype(bool)
        return out


# ─────────────────────────────────────────────────────────────────────────────
#  Learned decoder
# ─────────────────────────────────────────────────────────────────────────────

class LearnedDecoder:
    """Gradient-boosted classifier: detection events -> logical flip."""

    def __init__(self, **kw):
        from sklearn.ensemble import HistGradientBoostingClassifier
        kw = dict(dict(max_iter=300, learning_rate=0.05, random_state=0), **kw)
        self.clf = HistGradientBoostingClassifier(**kw)

    def fit(self, det_train: np.ndarray, obs_train: np.ndarray):
        self.clf.fit(det_train.astype(np.uint8), obs_train[:, 0].astype(bool)
                     if obs_train.ndim == 2 else obs_train.astype(bool))
        return self

    def decode(self, det: np.ndarray) -> np.ndarray:
        return self.clf.predict(det.astype(np.uint8)).astype(bool)[:, None]
