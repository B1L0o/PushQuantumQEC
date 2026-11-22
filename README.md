# A distance-3 surface code on IQM Emerald

**Epita-Q** — Alexis, Cyril, Robin, Victor · IQM QEC challenge (the organizers' instructions are in [CHALLENGE.md](CHALLENGE.md))

We built the full quantum-memory pipeline and ran it on IQM's 54-qubit **Emerald**:

- a rotated surface code, distance 3, with **no mid-circuit reset**;
- 1 to 5 rounds, **400 000 hardware shots** in total;
- compiled with **zero SWAPs**;
- decoded by **four decoders** (MWPM, Union-Find, BP+OSD and a graph neural network);
- fitted to a logical error rate per round.

Two further pieces:

- **A noise model** that starts from IQM's calibration, is then **characterized from our own syndromes**, and is validated against the hardware. It is used to choose the qubits, to weight the decoders and to project to larger codes.
- **A finding the calibration didn't show.** The syndromes showed that one qubit, QB26, becomes random after its first readout. Excluding it improved the logical error more than any change of decoder.

**Start here:** [notebooks/surface_code_emerald.ipynb](notebooks/surface_code_emerald.ipynb), the whole experiment with all results (runs offline in under a minute), and the slides [presentation/main.pdf](presentation/main.pdf).

---

## Results

| | |
|---|---|
| Logical error rate per round, $\varepsilon_L$ | MWPM **9.8 ± 0.1 %**, Union-Find 9.5 %, BP+OSD 10.0 %, GNN 10.5 % (no decoder: 11.5 %) |
| Best $P_L$, 2 rounds | GNN **6.8 %**, against 15.0 % without a decoder (−55 % logical errors) |
| SWAPs | **0**: 72 CZs, exactly those of the code; a naive mapping needs 222 CZs (≈ 50 SWAPs) |
| Noise model vs hardware | characterized model within 4–9 %; calibration alone is 3–5× too optimistic |
| Most important factor for the logical error | excluding QB26: MWPM 15.0 % → 8.2 % at 2 rounds |
| Threshold (projection) | ≈ 0.45× today's noise; $\Lambda \approx 2$ at a quarter of today's noise |

Every hardware number is measured on **held-out shots**: the half never used to estimate the noise or train a decoder.

### $P_L$ on Emerald (QB26 excluded, 30 000 test shots per point, in %)

| Rounds | No decoder | MWPM | Union-Find | BP+OSD | GNN |
|---|---|---|---|---|---|
| 1 | 5.9 | 0.8 | 1.2 | 0.8 | 0.8 |
| 2 | 15.0 | 8.2 | 8.1 | 7.4 | **6.8** |
| 3 | 23.0 | 16.3 | 15.7 | 15.8 | **15.5** |
| 4 | 29.3 | 23.1 | **22.4** | 22.9 | 22.7 |
| 5 | 34.0 | 28.2 | **27.9** | 28.3 | 29.1 |

---

## What we did, by judging criterion

### 1. Theoretical correctness

- **No-reset detectors.** Without reset, an ancilla measures $m_k = s_k \oplus m_{k-1}$, so a change of stabilizer is $s_k \oplus s_{k-1} = m_k \oplus m_{k-2}$. Each detector compares the same ancilla's results two rounds apart. Code: `make_stim_circuit(no_reset=True)` in `surface_code.py`.
- **Checked without noise.** No detector ever fires and the logical value never flips, so every detector is deterministic.
- **Data path checked on hardware.** Before the main runs, `smoke_test.py` prepared known bitstrings on Emerald and checked that all 17 qubits come back through the full chain (Qiskit counts, bit-order reversal, our measurement mapping).

### 2. Hardware optimization: zero SWAPs

`layout_search.py`:

1. **Every placement.** We list all embeddings of the code's 17-qubit interaction graph into Emerald's coupling map (subgraph isomorphism, `rustworkx.vf2_mapping`): 76 placements, or 12 without QB26.
2. **Ranking by simulation.** Each placement is simulated with the noise model applied to its own qubits, and decoded with MWPM. The best 3 are re-simulated with 100k shots.
3. **Compilation with routing off** (`routing_method="none"`). Any gate that needed a SWAP would make the transpile fail.
4. **Proof on the instructions sent to the QPU.** Same CZ count as the code (72), every CZ on a real coupler (physical qubits read back through `circuit.layout`), and the measurement order checked against the decoder's.

| Compilation | CZs | SWAPs | Uses QB26 | Simulated $P_L$ (3 rounds) |
|---|---|---|---|---|
| Naive (QB1…QB17) | 222 | ≈ 50 | — | — |
| Qiskit default layout | 72 | 0 | yes | 21.8 % |
| **Ours** | **72** | **0** | no | **17.1 %** |

### 3. Logical-error statistics and reporting

- 60 000 shots per round count, in 10 000-shot jobs, with the calibration saved at the start and end of each batch (`main_runs.py`).
- Train/test split 50/50 with a fixed seed. All decoders are scored on the same test shots, with binomial error bars.
- Fit of $2P_L(n) = 1 - A(1-2\varepsilon_L)^n$ on rounds 2–5 by weighted least squares (`fit_ler.py`). $A$ absorbs state preparation and the final readout. Round 1 is left out: without reset it has no detector two rounds back. The model given in the challenge instructions, with $A = 1$, fits worse, because the error per round grows with depth.

### 4. Advanced features

**Noise model** (`emerald_noise.py`):
- **From IQM's calibration:** $T_1$, $T_2$, gate and readout fidelities, per qubit and per coupler.
- **Gates:** depolarizing noise, $p_{1q} = 1.5(1-F)$ and $p_{2q} = 1.25(1-F)$.
- **Readout:** a flip with probability $1 - F_{\text{readout}}$.
- **Idling:** Pauli-twirled $T_1/T_2$ noise on every waiting qubit in every layer, including the 1.5 µs readout.

**Characterization from the syndromes** (`estimate_dem.py`, `characterize.py`):
- Estimate the probability $p_{ij}$ of each edge of the detector graph from correlations of detection events (Google 2021).
- Fit one multiplier per qubit so that the model reproduces them, with a bias correction from simulations of the model itself.
- Validation on test shots: within 4–9 %, against a factor 3–5 for the calibration alone.

**Four decoders** (`decoders.py`, `compare_decoders.py`):

| Decoder | How it works |
|---|---|
| MWPM | PyMatching, on the detector error model of the noise model |
| Union-Find | our own weighted implementation (Delfosse–Nickerson): clusters grow around fired detectors, then peeling; near-linear time |
| BP+OSD | `ldpc`: belief propagation on the full, undecomposed error model (sees correlated X/Z errors), completed by ordered-statistics decoding |
| GNN | the detectors that fired form a graph ($k = 6$ nearest neighbours in $(x, y, t)$, weights 1/distance); 4 GraphConv layers, then a graph-level classifier; pre-trained on simulation, fine-tuned on the training half of the hardware shots |

**Projection to larger codes** (`make_figures.py`):
- Simulation of $d = 3, 5, 7$ with the characterized model.
- Each qubit's noise is drawn from Emerald's measured spread, then the whole model is scaled by a factor.
- Results: threshold at ≈ 0.45× today's noise; $\Lambda$ = 0.76 today, 1.04 at ÷2, 1.97 at ÷4.

### Bonus: what limits the logical error, found experimentally

1. **QB26.** It reads 1 in 8 % of shots in round 1, then 50 % from round 2. Its readout scrambles it, and without reset the damage carries forward. Calibration rated it among the best qubits (readout 99.8 %, $T_1$ = 71 µs). Our characterization gave it ×87, and with it every decoder was worse than no correction. Excluding it was the biggest improvement of the project.
2. **History-dependent errors.** How often an ancilla's detector fires depends on its previous outcome (QB30: 13 % after a 0, 47 % after a 1). An ancilla left in $|1\rangle$ relaxes during the long readout, and a fixed-probability model can't represent that.
3. **One-sided readout.** $|1\rangle$ is read as 0 far more often than $|0\rangle$ is read as 1 (2–15 % against ≤ 1 %). A second readout is worse than the first.
4. **The GNN learns this from the data.** On the run with QB26 it reaches 11.5 % against ≈ 17.7 % for the classical decoders. An ablation that scrambles one ancilla at a time (`gnn_ablation.py`) shows it relies on QB26 three times less than MWPM does.

Also found: QB42, an X-type ancilla, is random too. It doesn't affect this Z-basis memory, but an X-basis memory would have to exclude it.

---

## Repository

| Step | File |
|---|---|
| No-reset circuit | `surface_code.py` |
| Noise model | `emerald_noise.py` |
| $p_{ij}$ estimation, data-driven matching | `estimate_dem.py` |
| Characterization | `characterize.py` |
| Placement search and zero-SWAP checks | `layout_search.py` |
| Hardware execution | `run_on_hardware.py`, `smoke_test.py`, `main_runs.py` |
| Decoders and their comparison | `decoders.py`, `compare_decoders.py`, `history_decoder.py` |
| Fit of $\varepsilon_L$ | `fit_ler.py`, `analyze_main.py` |
| Figures, threshold, projection | `make_figures.py` |
| GNN ablation | `gnn_ablation.py` |
| Walkthrough notebook | `notebooks/surface_code_emerald.ipynb` (regenerated by `notebooks/build_notebook.py`) |
| Slides | `presentation/` |

**Data** (`runs/`):

| Folder | Content |
|---|---|
| `20261121_175437_smoke/` | known-state test |
| `20261121_181236_main/` | first placement, with QB26, 100 000 shots |
| `20261121_193303_main/` | final placement, without QB26, 300 000 shots, plus the characterized model and decoder results |
| `figures/` | saved results behind every figure |

Each `r{n}.npz` holds the detection events, the logical flips and the raw measurements for $n$ rounds. Next to it, `r{n}_meta.json` holds the counts and job details. The layout and the calibrations are saved with each run.

## Reproduce

```bash
pip install -r requirements.txt

# everything, offline, from the saved data
jupyter notebook notebooks/surface_code_emerald.ipynb

# recompute decoders and figures from the saved data
python compare_decoders.py runs/20261121_193303_main
python make_figures.py
python gnn_ablation.py
python presentation/build_data.py && cd presentation && latexmk -pdf main.tex
```

On the QPU, the Resonance token is read from `TOKEN_IQM` or `~/.iqm_token` and is never stored in the repository:

```bash
python smoke_test.py --shots 1000                                     # known-state test (--fake: offline)
python main_runs.py --exclude QB26 --rounds 1 2 3 4 5 --shots 60000   # placement search + runs
python analyze_main.py runs/<run_dir>
```

## Limits

- Distance 3 only on hardware: larger codes are projections from a model validated at $d = 3$.
- The model has no leakage and no failures like QB26's, so the projections are optimistic.
- Z-basis memory only.

**Next steps:** decoding that uses each ancilla's previous outcome and detects leakage; dynamical decoupling on the data qubits during readout; the X basis; automatic exclusion of failing qubits at the start of each session.
