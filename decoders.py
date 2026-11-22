# -*- coding: utf-8 -*-
"""
decoders.py

Four decoders behind one interface, all fed with the same detector error
model (DEM) built from the noise model and the same detection events:

    MWPMDecoder       minimum-weight perfect matching (PyMatching)
    UnionFindDecoder  weighted Union-Find (ldpc), on the graph-like DEM
    BPOSDDecoder      belief propagation + ordered-statistics post-processing
                      (ldpc), on the full DEM hypergraph (no decomposition)
    GNNDecoder        graph neural network on the FIXED space-time graph
                      (nodes = detectors, edges = DEM edges), trained on
                      labelled shots (hardware TRAIN half), shared weights,
                      message passing, graph-level readout -> logical flip

    dec = MWPMDecoder(dem); pred = dec.decode(det_events)     # (shots, num_obs) bool
    gnn = GNNDecoder(circuit, dem).fit(det_train, obs_train); gnn.decode(det_test)
"""

from __future__ import annotations

import math

import numpy as np
import pymatching
import stim
from scipy.sparse import csc_matrix

from estimate_dem import graph_structure


# ─────────────────────────────────────────────────────────────────────────────
#  DEM -> matrices
# ─────────────────────────────────────────────────────────────────────────────

def dem_to_matrices(dem: stim.DetectorErrorModel):
    """Check matrix H (detectors x errors), logical matrix L (observables x
    errors) and prior probabilities of every error mechanism of the DEM."""
    rows_h, cols_h, rows_l, cols_l, priors = [], [], [], [], []
    col = 0
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        dets, obs = set(), set()
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets ^= {t.val}
            elif t.is_logical_observable_id():
                obs ^= {t.val}
        if not dets and not obs:
            continue
        rows_h += sorted(dets); cols_h += [col] * len(dets)
        rows_l += sorted(obs); cols_l += [col] * len(obs)
        priors.append(inst.args_copy()[0])
        col += 1
    H = csc_matrix((np.ones(len(rows_h), np.uint8), (rows_h, cols_h)),
                   shape=(dem.num_detectors, col))
    L = csc_matrix((np.ones(len(rows_l), np.uint8), (rows_l, cols_l)),
                   shape=(dem.num_observables, col))
    return H, L, np.clip(np.array(priors), 1e-9, 0.5 - 1e-9)


def _graph_matrices(dem: stim.DetectorErrorModel):
    """Graph-like version (one column per detector pair / boundary edge)."""
    s = graph_structure(dem)
    cols = list(s.edges) + list(s.boundary)
    rows_h, cols_h, rows_l, cols_l, p = [], [], [], [], []
    for c, key in enumerate(cols):
        if isinstance(key, tuple):
            rows_h += list(key); cols_h += [c, c]
            obs, prob = s.edge_obs[key], s.edges[key]
        else:
            rows_h.append(key); cols_h.append(c)
            obs, prob = s.boundary_obs[key], s.boundary[key]
        rows_l += sorted(obs); cols_l += [c] * len(obs)
        p.append(prob)
    H = csc_matrix((np.ones(len(rows_h), np.uint8), (rows_h, cols_h)),
                   shape=(s.num_detectors, len(cols)))
    L = csc_matrix((np.ones(len(rows_l), np.uint8), (rows_l, cols_l)),
                   shape=(s.num_observables, len(cols)))
    return H, L, np.clip(np.array(p), 1e-9, 0.5 - 1e-9), s


# ─────────────────────────────────────────────────────────────────────────────
#  Classical decoders
# ─────────────────────────────────────────────────────────────────────────────

class MWPMDecoder:
    name = "MWPM"

    def __init__(self, dem: stim.DetectorErrorModel):
        self.m = pymatching.Matching.from_detector_error_model(dem)

    def decode(self, det: np.ndarray) -> np.ndarray:
        return self.m.decode_batch(det.astype(np.uint8)).astype(bool)


class UnionFindDecoder:
    """Weighted Union-Find (Delfosse & Nickerson; Huang, Newman & Brown weights).

    Nodes = detectors + one virtual boundary node. Edge length = integer
    rounding of its log-likelihood weight log((1-p)/p). Odd clusters (odd
    number of defects, not touching the boundary) grow by one unit per step
    on every edge they touch; fully grown edges merge clusters. Then each
    cluster's spanning forest is peeled from the leaves to get the correction.
    (ldpc's UnionFindDecoder was tried first: it ignores the very uneven
    circuit-level weights and was worse than no correction in simulation.)
    """
    name = "Union-Find"

    def __init__(self, dem: stim.DetectorErrorModel, resolution: float = 2.0):
        s = graph_structure(dem)
        n = s.num_detectors
        self.n, self.B = n, n                       # boundary node index = n
        edges, obs, length = [], [], []
        for (i, j), p in s.edges.items():
            edges.append((i, j)); obs.append(s.edge_obs[(i, j)])
            length.append(math.log((1 - p) / max(p, 1e-9)))
        for i, p in s.boundary.items():
            edges.append((i, n)); obs.append(s.boundary_obs[i])
            length.append(math.log((1 - p) / max(p, 1e-9)))
        lmin = max(min(length), 1e-3)
        self.w = np.maximum(1, np.round(resolution * np.array(length) / lmin)).astype(int)
        self.edges = edges
        self.num_obs = s.num_observables
        self.obs_mask = np.zeros((len(edges), self.num_obs), bool)
        for e, o in enumerate(obs):
            for k in o:
                self.obs_mask[e, k] = True
        self.adj = [[] for _ in range(n + 1)]
        for e, (u, v) in enumerate(edges):
            self.adj[u].append(e); self.adj[v].append(e)

    def _decode_one(self, defects: np.ndarray) -> np.ndarray:
        n1 = self.n + 1
        parent = list(range(n1))
        parity = [0] * n1
        boundary = [False] * n1
        boundary[self.B] = True
        for d in np.flatnonzero(defects):
            parity[d] = 1

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        grown = np.zeros(len(self.edges), int)
        full = np.zeros(len(self.edges), bool)
        in_cluster = [False] * n1                 # node reached by some odd cluster
        for d in np.flatnonzero(defects):
            in_cluster[d] = True
        while True:
            odd_roots = {r for r in (find(x) for x in range(n1) if in_cluster[x])
                         if parity[r] % 2 == 1 and not boundary[r]}
            if not odd_roots:
                break
            newly_full = []
            for x in range(n1):
                if in_cluster[x] and find(x) in odd_roots:
                    for e in self.adj[x]:
                        if not full[e]:
                            grown[e] += 1
                            if grown[e] >= self.w[e]:
                                full[e] = True
                                newly_full.append(e)
            for e in newly_full:
                u, v = self.edges[e]
                in_cluster[u] = in_cluster[v] = True
                ru, rv = find(u), find(v)
                if ru != rv:
                    parent[ru] = rv
                    parity[rv] += parity[ru]
                    boundary[rv] = boundary[rv] or boundary[ru]

        # peeling on a spanning forest of the fully grown edges
        tree_adj = [[] for _ in range(n1)]
        for e in np.flatnonzero(full):
            u, v = self.edges[e]
            tree_adj[u].append((v, e)); tree_adj[v].append((u, e))
        seen = [False] * n1
        pred = np.zeros(self.num_obs, bool)
        mark = defects.astype(int).tolist() + [0]
        roots = [self.B] + [x for x in range(self.n) if in_cluster[x]]
        for root in roots:
            if seen[root]:
                continue
            order, par_edge, stack = [], {}, [root]
            seen[root] = True
            while stack:
                x = stack.pop()
                order.append(x)
                for y, e in tree_adj[x]:
                    if not seen[y]:
                        seen[y] = True
                        par_edge[y] = (x, e)
                        stack.append(y)
            for x in reversed(order[1:]):
                if mark[x] % 2 == 1:
                    px, e = par_edge[x]
                    pred ^= self.obs_mask[e]
                    mark[px] += 1
        return pred

    def decode(self, det: np.ndarray) -> np.ndarray:
        out = np.zeros((len(det), self.num_obs), dtype=bool)
        for k, syn in enumerate(det.astype(bool)):
            if syn.any():
                out[k] = self._decode_one(syn)
        return out


class BPOSDDecoder:
    name = "BP+OSD"

    def __init__(self, dem: stim.DetectorErrorModel, max_iter: int = 30, osd_order: int = 7):
        from ldpc import BpOsdDecoder
        self.H, self.L, p = dem_to_matrices(dem)
        self.bp = BpOsdDecoder(self.H, error_channel=list(p), max_iter=max_iter,
                               bp_method="product_sum", osd_method="osd_cs",
                               osd_order=osd_order)

    def decode(self, det: np.ndarray) -> np.ndarray:
        out = np.zeros((len(det), self.L.shape[0]), dtype=bool)
        for k, syn in enumerate(det.astype(np.uint8)):
            if syn.any():
                out[k] = (self.L @ self.bp.decode(syn)) % 2
        return out


# ─────────────────────────────────────────────────────────────────────────────
#  GNN on the fixed space-time graph
# ─────────────────────────────────────────────────────────────────────────────

class GNNDecoder:
    """Message-passing GNN, fixed topology, shared weights.

    Node features: fired, x, y, t (normalised), is-first-round, is-final-round.
    Edge weights: DEM log-likelihood weight of the edge (normalised), so the
    prior noise model enters the graph. Layers:
        h <- relu(h W_self + (A h) W_nb) + h          (residual)
    Readout: [mean, max] over nodes -> MLP -> logit of the logical flip.
    """
    name = "GNN"

    def __init__(self, circuit: stim.Circuit, dem: stim.DetectorErrorModel,
                 hidden: int = 64, layers: int = 4, seed: int = 0):
        import torch
        self.torch = torch
        torch.manual_seed(seed)
        s = graph_structure(dem)
        n = s.num_detectors
        A = np.zeros((n, n), np.float32)
        for (i, j), p in s.edges.items():
            w = math.log((1 - p) / max(p, 1e-6))
            A[i, j] = A[j, i] = 1.0 / (1.0 + w)        # likely edge -> strong coupling
        A /= np.maximum(A.sum(1, keepdims=True), 1e-6)
        coords = circuit.get_detector_coordinates()
        xyz = np.array([coords[k] for k in range(n)], np.float32)
        span = np.maximum(xyz.max(0) - xyz.min(0), 1)
        pos = (xyz - xyz.min(0)) / span
        t = xyz[:, 2]
        static = np.column_stack([pos, t == t.min(), t == t.max()]).astype(np.float32)
        self.A = torch.tensor(A)
        self.static = torch.tensor(static)
        f_in = 1 + static.shape[1]
        nn = torch.nn
        self.inp = nn.Linear(f_in, hidden)
        self.w_self = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.w_nb = nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.params = [p for m in (self.inp, *self.w_self, *self.w_nb, self.head)
                       for p in m.parameters()]

    def _forward(self, det):
        torch = self.torch
        x = torch.cat([det.unsqueeze(-1),
                       self.static.unsqueeze(0).expand(det.shape[0], -1, -1)], -1)
        h = torch.relu(self.inp(x))
        for ws, wn in zip(self.w_self, self.w_nb):
            h = torch.relu(ws(h) + wn(self.A @ h)) + h
        g = torch.cat([h.mean(1), h.max(1).values], -1)
        return self.head(g).squeeze(-1)

    def fit(self, det: np.ndarray, obs: np.ndarray, epochs: int = 40,
            batch: int = 512, lr: float = 2e-3, val_frac: float = 0.1, verbose=False):
        torch = self.torch
        X = torch.tensor(det.astype(np.float32))
        y = torch.tensor((obs[:, 0] if obs.ndim == 2 else obs).astype(np.float32))
        n_val = int(len(X) * val_frac)
        Xv, yv, Xt, yt = X[:n_val], y[:n_val], X[n_val:], y[n_val:]
        opt = torch.optim.Adam(self.params, lr=lr, weight_decay=1e-5)
        best, best_state, patience = 1.0, None, 0
        for ep in range(epochs):
            perm = torch.randperm(len(Xt))
            for k in range(0, len(Xt), batch):
                idx = perm[k:k + batch]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    self._forward(Xt[idx]), yt[idx])
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                err = float(((self._forward(Xv) > 0).float() != yv).float().mean())
            if err < best - 1e-4:
                best, patience = err, 0
                best_state = [p.detach().clone() for p in self.params]
            else:
                patience += 1
                if patience >= 6:
                    break
            if verbose:
                print(f"  epoch {ep}: val error {err:.4f}")
        if best_state:
            for p, b in zip(self.params, best_state):
                p.data.copy_(b)
        return self

    def decode(self, det: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            out = []
            for k in range(0, len(det), 4096):
                out.append((self._forward(torch.tensor(det[k:k + 4096].astype(np.float32))) > 0).numpy())
        return np.concatenate(out)[:, None]
