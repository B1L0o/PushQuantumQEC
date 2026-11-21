# -*- coding: utf-8 -*-
"""
estimate_dem.py

Estimate the decoding-graph edge probabilities directly from detection events
(hardware or simulated), using pairwise detector correlations, and build a
PyMatching decoder from them.

Method (Google, Nature 2021, "Exponential suppression of bit or phase errors",
supplementary; see also Takou & Brown, arXiv 2504.20212):

    p_ij = 1/2 - 1/2 sqrt( 1 - 4 (<x_i x_j> - <x_i><x_j>)
                               / (1 - 2<x_i> - 2<x_j> + 4<x_i x_j>) )

Boundary edge of detector i, once its bulk edges are known:

    1 - 2<x_i> = (1 - 2 p_iB) * prod_j (1 - 2 p_ij)

Assumptions / limitations (state them in the report):
  * independent errors, each flipping at most 2 detectors (graph-like);
    hyperedges (Y errors, some CZ errors) and leakage are only partly captured.
  * statistical noise -> some estimates are negative / NaN; they are clipped.
  * ALWAYS estimate on one set of shots and evaluate the LER on another one.

The STRUCTURE of the graph (which detector pairs are edges, which edges flip
the logical observable) comes from a model DEM, e.g. the one produced by
emerald_noise.EmeraldNoiseModel. Only the probabilities are taken from data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pymatching
import stim

P_MIN = 1e-6          # clip estimates to [P_MIN, P_MAX]
P_MAX = 0.45


@dataclass
class GraphStructure:
    """Graph-like edges of a decomposed DEM, merged per detector pair."""
    edges: dict[tuple[int, int], float]      # (i, j) -> model probability
    boundary: dict[int, float]               # i -> model boundary probability
    edge_obs: dict[tuple[int, int], frozenset]
    boundary_obs: dict[int, frozenset]
    num_detectors: int
    num_observables: int


def _xor_prob(p: float, q: float) -> float:
    return p * (1 - q) + q * (1 - p)


def graph_structure(dem: stim.DetectorErrorModel) -> GraphStructure:
    """Extract graph-like edges from a DEM built with decompose_errors=True."""
    edges, boundary, edge_obs, boundary_obs = {}, {}, {}, {}
    best_e, best_b = {}, {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        comps, cur = [], []
        for t in inst.targets_copy():
            if t.is_separator():
                comps.append(cur)
                cur = []
            else:
                cur.append(t)
        comps.append(cur)
        for comp in comps:
            dets = sorted(t.val for t in comp if t.is_relative_detector_id())
            obs = frozenset(t.val for t in comp if t.is_logical_observable_id())
            if len(dets) == 2:
                k = (dets[0], dets[1])
                edges[k] = _xor_prob(edges.get(k, 0.0), p)
                if p > best_e.get(k, -1):
                    best_e[k], edge_obs[k] = p, obs
            elif len(dets) == 1:
                k = dets[0]
                boundary[k] = _xor_prob(boundary.get(k, 0.0), p)
                if p > best_b.get(k, -1):
                    best_b[k], boundary_obs[k] = p, obs
            elif len(dets) > 2:
                raise ValueError("DEM is not graph-like: build it with decompose_errors=True")
    return GraphStructure(edges, boundary, edge_obs, boundary_obs,
                          dem.num_detectors, dem.num_observables)


def estimate_edge_probabilities(dets: np.ndarray, structure: GraphStructure,
                                fallback_to_prior: bool = False
                                ) -> tuple[dict[tuple[int, int], float], dict[int, float]]:
    """Estimate bulk and boundary edge probabilities from detection events.

    dets : (shots, num_detectors) bool array.
    fallback_to_prior : when an estimate is invalid (negative / NaN, from
        statistical noise), use the model's probability instead of P_MIN.
        Use True when building a decoder: a P_MIN edge gets an almost infinite
        weight and breaks matching (seen with 10k shots on no-reset circuits).
    """
    x = dets.astype(np.float64)
    mean = x.mean(axis=0)
    est_edges = {}
    for (i, j) in structure.edges:
        xij = np.mean(x[:, i] * x[:, j])
        num = 4 * (xij - mean[i] * mean[j])
        den = 1 - 2 * mean[i] - 2 * mean[j] + 4 * xij
        arg = 1 - num / den if den > 0 else float("nan")
        p = 0.5 - 0.5 * math.sqrt(arg) if arg >= 0 else float("nan")
        if np.isfinite(p) and p > P_MIN:
            est_edges[(i, j)] = float(min(p, P_MAX))
        else:
            est_edges[(i, j)] = structure.edges[(i, j)] if fallback_to_prior else P_MIN

    # boundary: 1 - 2<x_i> = (1 - 2 p_iB) * prod_j (1 - 2 p_ij)
    prod = np.ones(structure.num_detectors)
    for (i, j), p in est_edges.items():
        prod[i] *= 1 - 2 * p
        prod[j] *= 1 - 2 * p
    est_boundary = {}
    for i in structure.boundary:
        ratio = (1 - 2 * mean[i]) / prod[i]
        p = 0.5 * (1 - ratio)
        if np.isfinite(p) and p > P_MIN:
            est_boundary[i] = float(min(p, P_MAX))
        else:
            est_boundary[i] = structure.boundary[i] if fallback_to_prior else P_MIN
    return est_edges, est_boundary


def matching_from_probabilities(structure: GraphStructure,
                                edges: dict[tuple[int, int], float],
                                boundary: dict[int, float]) -> pymatching.Matching:
    """PyMatching decoder with weights log((1-p)/p) from the given probabilities."""
    m = pymatching.Matching()
    for (i, j), p in edges.items():
        m.add_edge(i, j, fault_ids=set(structure.edge_obs[(i, j)]),
                   weight=math.log((1 - p) / p), error_probability=p,
                   merge_strategy="independent")
    for i, p in boundary.items():
        m.add_boundary_edge(i, fault_ids=set(structure.boundary_obs[i]),
                            weight=math.log((1 - p) / p), error_probability=p,
                            merge_strategy="independent")
    # isolated detectors (no edge) still need to exist in the graph
    covered = set(boundary) | {d for e in edges for d in e}
    for d in range(structure.num_detectors):
        if d not in covered:
            m.add_boundary_edge(d, weight=math.log((1 - P_MIN) / P_MIN))
    return m


def data_driven_matching(dets_train: np.ndarray, model_dem: stim.DetectorErrorModel
                         ) -> tuple[pymatching.Matching, dict, dict, GraphStructure]:
    """Convenience: structure from the model DEM, probabilities from data."""
    s = graph_structure(model_dem)
    e, b = estimate_edge_probabilities(dets_train, s, fallback_to_prior=True)
    return matching_from_probabilities(s, e, b), e, b, s


def logical_error_probability(matcher: pymatching.Matching, dets: np.ndarray,
                              obs: np.ndarray) -> tuple[float, float]:
    """P_L and its 1-sigma binomial error."""
    fail = np.any(matcher.decode_batch(dets) != obs, axis=1)
    p = float(fail.mean())
    return p, math.sqrt(p * (1 - p) / len(fail))
