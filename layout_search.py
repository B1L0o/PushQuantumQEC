# -*- coding: utf-8 -*-
"""
layout_search.py

Calibration-informed, zero-SWAP qubit placement for the surface code on IQM
Emerald (or any IQM backend), and a transpile step that PROVES zero SWAPs.

    1. enumerate_layouts()   every exact embedding of the code's interaction
                             graph into the backend coupling map (VF2, so
                             every CX/CZ pair is a native coupler -> 0 SWAP).
    2. score_layouts()       rank them with the calibrated noise model
                             (emerald_noise.EmeraldNoiseModel):
                               a) fast proxy = total DEM error mass, all layouts
                               b) Stim + PyMatching LER, top-k layouts only
    3. transpile_zero_swap() transpile with that initial_layout and
                             routing_method="none" (fails instead of adding a
                             SWAP), then verify: no SWAP, every 2q gate on a
                             coupler, and every measurement still lands in the
                             classical bit the m2d converter expects.

Usage
-----
    from layout_search import find_best_layout, transpile_zero_swap
    best, ranking = find_best_layout(stim_circ, backend, noise_model, rounds=3)
    qc, stim_to_dense, meas_order = stim_to_qiskit(stim_circ)
    qc_t, report = transpile_zero_swap(qc, backend, best.stim_to_phys, stim_to_dense, meas_order)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pymatching
import rustworkx as rx
import stim
from qiskit import QuantumCircuit, transpile

from emerald_noise import EmeraldNoiseModel


# ─────────────────────────────────────────────────────────────────────────────
#  Graphs
# ─────────────────────────────────────────────────────────────────────────────

def code_interaction_graph(stim_circuit: stim.Circuit) -> tuple[rx.PyGraph, list[int]]:
    """Undirected graph of the 2-qubit gates of the circuit. Node k = stim qubit qubits[k]."""
    pairs, qubits = set(), set()
    for inst in stim_circuit.flattened():
        if inst.name in ("CX", "CNOT", "CZ"):
            qs = [t.value for t in inst.targets_copy() if t.is_qubit_target]
            for a, b in zip(qs[::2], qs[1::2]):
                pairs.add((min(a, b), max(a, b)))
                qubits.update((a, b))
    qubits = sorted(qubits)
    idx = {q: k for k, q in enumerate(qubits)}
    g = rx.PyGraph()
    g.add_nodes_from(qubits)
    g.add_edges_from_no_data([(idx[a], idx[b]) for a, b in pairs])
    return g, qubits


def coupling_graph(backend) -> rx.PyGraph:
    """Undirected coupling graph of the backend. Node i = physical qubit index i."""
    g = rx.PyGraph()
    g.add_nodes_from(range(backend.num_qubits))
    edges = {(min(a, b), max(a, b)) for a, b in backend.coupling_map.get_edges()}
    g.add_edges_from_no_data(sorted(edges))
    return g


# ─────────────────────────────────────────────────────────────────────────────
#  1. Enumeration
# ─────────────────────────────────────────────────────────────────────────────

def enumerate_layouts(stim_circuit: stim.Circuit, backend,
                      max_layouts: Optional[int] = None,
                      exclude: Sequence[str] = ()) -> list[dict[int, int]]:
    """All zero-SWAP embeddings: list of {stim qubit -> physical qubit index}.

    exclude : physical qubit names never to use (e.g. qubits found broken by
              characterization, like a qubit randomized by its own readout).
    """
    code, qubits = code_interaction_graph(stim_circuit)
    chip = coupling_graph(backend)
    banned = {i for i in range(backend.num_qubits) if backend.index_to_qubit_name(i) in set(exclude)}
    if banned:
        chip.remove_nodes_from(sorted(banned))
    layouts = []
    # vf2_mapping(G, H, subgraph=True) yields {node of G -> node of H} with H a
    # (non-induced) subgraph of G.
    for m in rx.vf2_mapping(chip, code, subgraph=True, induced=False, id_order=False):
        layouts.append({qubits[c]: p for p, c in m.items()})
        if max_layouts and len(layouts) >= max_layouts:
            break
    if not layouts:
        raise RuntimeError("No zero-SWAP embedding of this code on the backend coupling map.")
    return layouts


# ─────────────────────────────────────────────────────────────────────────────
#  2. Scoring
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LayoutScore:
    stim_to_phys: dict[int, int]
    stim_to_name: dict[int, str]
    proxy: float                          # total DEM error mass (lower = better)
    ler: Optional[float] = None           # simulated P_L after `rounds` rounds
    ler_err: Optional[float] = None
    physical_qubits: list[str] = field(default_factory=list)


def _dem(noise_model, stim_circuit, stim_to_name):
    noisy = noise_model.apply(stim_circuit, stim_to_name)
    return noisy, noisy.detector_error_model(decompose_errors=True,
                                             approximate_disjoint_errors=True)


def score_layouts(stim_circuit: stim.Circuit, layouts, backend,
                  noise_model: EmeraldNoiseModel, top_k: Optional[int] = None,
                  shots: int = 20_000, seed: int = 0,
                  verbose: bool = True) -> list[LayoutScore]:
    """Proxy-rank every layout, then simulate the LER of the top_k.

    stim_circuit : NOISELESS no-reset circuit (make_stim_circuit(..., noise=None)).
    top_k        : None -> simulate ALL layouts (d=3 on Emerald: ~100 layouts,
                   a few seconds). The proxy is only a weak predictor
                   (Spearman ~0.4 in tests), so only use top_k for big searches.
    Returns all layouts, the simulated ones first, sorted by LER then proxy.
    """
    top_k = len(layouts) if top_k is None else top_k
    t0 = time.time()
    scores = []
    for lay in layouts:
        names = {q: backend.index_to_qubit_name(p) for q, p in lay.items()}
        _, dem = _dem(noise_model, stim_circuit, names)
        proxy = sum(i.args_copy()[0] for i in dem.flattened() if i.type == "error")
        scores.append(LayoutScore(lay, names, proxy,
                                  physical_qubits=sorted(set(names.values()),
                                                         key=lambda s: int(s[2:]))))
    scores.sort(key=lambda s: s.proxy)
    if verbose:
        print(f"[layout_search] {len(scores)} zero-SWAP layouts proxy-scored "
              f"in {time.time() - t0:.1f}s")

    for k, s in enumerate(scores[:top_k]):
        noisy, dem = _dem(noise_model, stim_circuit, s.stim_to_name)
        m = pymatching.Matching.from_detector_error_model(dem)
        dets, obs = noisy.compile_detector_sampler(seed=seed + k).sample(
            shots, separate_observables=True)
        fail = np.any(m.decode_batch(dets) != obs, axis=1)
        s.ler = float(fail.mean())
        s.ler_err = float(np.sqrt(s.ler * (1 - s.ler) / shots))
    scores[:top_k] = sorted(scores[:top_k], key=lambda s: s.ler)
    if verbose:
        print(f"[layout_search] top {top_k} simulated ({shots} shots each) "
              f"in {time.time() - t0:.1f}s total")
    return scores


def find_best_layout(stim_circuit: stim.Circuit, backend,
                     noise_model: Optional[EmeraldNoiseModel] = None,
                     top_k: Optional[int] = None, shots: int = 20_000,
                     confirm: int = 3, confirm_shots: int = 100_000,
                     max_layouts: Optional[int] = None,
                     exclude: Sequence[str] = (),
                     verbose: bool = True) -> tuple[LayoutScore, list[LayoutScore]]:
    """Enumerate + score, then re-simulate the `confirm` best on FRESH shots.

    The re-simulation removes the winner's-curse bias (the minimum of many
    noisy estimates is optimistic); the confirmed numbers are the ones to report.
    Returns (best layout, full ranking).
    """
    noise_model = noise_model or EmeraldNoiseModel()
    layouts = enumerate_layouts(stim_circuit, backend, max_layouts, exclude)
    ranking = score_layouts(stim_circuit, layouts, backend, noise_model,
                            top_k=top_k, shots=shots, verbose=verbose)
    if confirm:
        fresh = score_layouts(stim_circuit, [s.stim_to_phys for s in ranking[:confirm]],
                              backend, noise_model, shots=confirm_shots,
                              seed=10_000, verbose=False)
        ranking[:confirm] = fresh
        if verbose:
            for s in fresh:
                print(f"[layout_search] confirmed P_L = {s.ler:.4f} +- {s.ler_err:.4f} "
                      f"on {s.physical_qubits[0]}..{s.physical_qubits[-1]}")
    return ranking[0], ranking


def build_qubit_map(best: LayoutScore, stim_to_dense: dict[int, int]) -> dict[int, int]:
    """{dense Qiskit index -> physical index}, the format run_qiskit() expects."""
    return {stim_to_dense[s]: p for s, p in best.stim_to_phys.items()}


# ─────────────────────────────────────────────────────────────────────────────
#  3. Zero-SWAP transpilation + proof
# ─────────────────────────────────────────────────────────────────────────────

def initial_layout_list(stim_to_dense: dict[int, int],
                        stim_to_phys: dict[int, int]) -> list[int]:
    """Qiskit initial_layout: position i = physical qubit of dense qubit i."""
    dense_to_stim = {d: s for s, d in stim_to_dense.items()}
    return [stim_to_phys[dense_to_stim[i]] for i in range(len(dense_to_stim))]


def transpile_zero_swap(qc: QuantumCircuit, backend, stim_to_phys: dict[int, int],
                        stim_to_dense: dict[int, int], meas_order: list[int],
                        optimization_level: int = 1, **transpile_kwargs
                        ) -> tuple[QuantumCircuit, dict]:
    """Transpile with a fixed layout and NO routing, then verify the result.

    Raises if a SWAP would be needed, if a 2q gate is off-coupler, or if any
    measurement no longer writes the physical qubit the m2d converter expects.
    """
    layout = initial_layout_list(stim_to_dense, stim_to_phys)
    qc_t = transpile(qc, backend=backend, initial_layout=layout,
                     routing_method="none", optimization_level=optimization_level,
                     **transpile_kwargs)

    couplers = {(min(a, b), max(a, b)) for a, b in backend.coupling_map.get_edges()}
    expected_pairs = set()
    for inst in qc.data:
        if inst.operation.num_qubits == 2:
            a, b = (layout[qc.find_bit(q).index] for q in inst.qubits)
            expected_pairs.add((min(a, b), max(a, b)))

    phys = physical_index_resolver(qc_t)
    ops = qc_t.count_ops()
    used_pairs, off_coupler, meas_ok = set(), [], True
    for inst in qc_t.data:
        qs = [phys(q) for q in inst.qubits]
        if inst.operation.num_qubits == 2:
            pair = (min(qs), max(qs))
            used_pairs.add(pair)
            if pair not in couplers:
                off_coupler.append(pair)
        if inst.operation.name == "measure":
            clbit = qc_t.find_bit(inst.clbits[0]).index
            if qs[0] != stim_to_phys[meas_order[clbit]]:
                meas_ok = False

    # Ground truth: the instructions IQM's own serializer would send to the QPU.
    serialized_off_coupler = []
    try:
        from iqm.qiskit_iqm.qiskit_to_iqm import serialize_instructions
        name_to_idx = {backend.index_to_qubit_name(i): i for i in range(backend.num_qubits)}
        for op in serialize_instructions(
                qc_t, {i: backend.index_to_qubit_name(i) for i in range(backend.num_qubits)}):
            if op.name == "cz":
                a, b = (name_to_idx[n] for n in op.locus)
                if (min(a, b), max(a, b)) not in couplers:
                    serialized_off_coupler.append(op.locus)
    except ImportError:
        serialized_off_coupler = None   # iqm-client not installed: skip this check

    report = dict(
        swaps=ops.get("swap", 0),
        serialized_off_coupler=serialized_off_coupler,
        two_qubit_gates=sum(1 for i in qc_t.data if i.operation.num_qubits == 2),
        two_qubit_gates_ideal=sum(1 for i in qc.data if i.operation.num_qubits == 2),
        off_coupler=off_coupler,
        pairs_match_code=used_pairs == expected_pairs,
        measurements_mapped_correctly=meas_ok,
        depth=qc_t.depth(),
        ops=dict(ops),
        initial_layout=layout,
    )
    if (report["swaps"] or off_coupler or serialized_off_coupler
            or not meas_ok or not report["pairs_match_code"]):
        raise RuntimeError(f"Zero-SWAP verification failed: {report}")
    return qc_t, report


def physical_index_resolver(qc_t: QuantumCircuit):
    """Map a qubit of a transpiled circuit to its physical index, exactly as
    IQM's serializer does: through circuit.layout when it is valid (IQM's
    transpile keeps the original qubit objects and stores the mapping there),
    otherwise the qubit's position in the circuit."""
    lay = qc_t.layout
    if lay is not None and lay.initial_layout.get_registers() == set(qc_t.qregs):
        table = {q: i for i, q in lay.initial_layout.get_physical_bits().items()}
        return lambda q: table[q]
    return lambda q: qc_t.find_bit(q).index
