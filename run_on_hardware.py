from surface_code import *
from stim_to_qiskit import stim_to_qiskit
from build_emerald_qubit_rotated import *
from extract_syndromes import *
from qiskit import transpile    


import os
# Token from the environment, never from a committed file: export TOKEN_IQM=...
def _read_token() -> str:
    """TOKEN_IQM / IQM_TOKEN env var, else ~/.iqm_token (outside the repo, chmod 600)."""
    tok = os.environ.get("TOKEN_IQM") or os.environ.get("IQM_TOKEN")
    if not tok:
        path = os.path.expanduser("~/.iqm_token")
        if os.path.exists(path):
            tok = open(path).read().strip()
    return tok or ""
TOKEN = _read_token()
SERVER_URL = "https://resonance.iqm.tech"


# ─────────────────────────────────────────────────────────────────────────────
#  5.  HARDWARE EXECUTION ON IQM RESONANCE
#
#  Two equal tracks. Pick ONE. Both return the same syndromes dict
#  (at least "det_events" and "obs_flips"), so decode_hardware_results()
#  works for either.
#
#  NOTE: Emerald does not support mid-circuit reset -> use the no-reset circuit.
# ─────────────────────────────────────────────────────────────────────────────

def run_qiskit(
    distance: int    = 3,
    rounds: int      = 1,
    shots: int       = 5_000,
    token: str       = None,
    quantum_computer: str = "emerald",
    no_reset: bool   = True,
    backend          = None,
    noise_model      = None,
    layout           = None,
    optimization_level: int = 1,
    dry_run: bool    = False,
) -> dict:
    """
    Build in qiskit. Runs the surface code on IQM Emerald via IQM Resonance.

        Stim circuit → Qiskit circuit → calibration-informed zero-SWAP layout
        → transpile (routing disabled, verified) → run → counts
        → raw measurements → extract_syndromes()

    Parameters
    ----------
    distance         : code distance (use 3 for emerald)
    rounds           : measurement rounds
    shots            : number of circuit executions
    token            : IQM Resonance API token (falls back to TOKEN)
    quantum_computer : e.g. "emerald"
    no_reset         : build the no-reset circuit (required on Emerald)
    backend          : optional backend (e.g. IQMFakeAphrodite() for offline tests);
                       None -> connect to IQM Resonance.
    noise_model      : emerald_noise.EmeraldNoiseModel used to rank layouts.
                       None -> live calibration of the backend when a token is
                       available, else the published Emerald medians.
    layout           : layout_search.LayoutScore to force a placement
                       (e.g. the same one for every round count). None -> search.
    optimization_level : passed to transpile (1 or 3; both keep 0 SWAPs).
    dry_run          : build, place and transpile only; do not submit a job.

    Returns
    -------
    syndromes : dict  output of extract_syndromes()
                "det_events" (shots, num_detectors), "obs_flips" (shots, num_observables), ...
                plus "layout" (LayoutScore), "transpile_report" (zero-SWAP proof),
                "noise_model_source" and, if available, "calibration_set_id".
    """
    from emerald_noise import EmeraldNoiseModel
    from layout_search import find_best_layout, transpile_zero_swap

    # Build circuits, exchange for make_qiskit_circuit() if desired
    stim_circ  = make_stim_circuit(distance, rounds, no_reset=no_reset, noise=None)
    qc, stim_to_dense, meas_order = stim_to_qiskit(stim_circ)

    # Connect to IQM Resonance (unless a backend, e.g. a fake one, is given)
    if backend is None:
        from iqm.qiskit_iqm import IQMProvider
        provider = IQMProvider(SERVER_URL, quantum_computer=quantum_computer,
                               token=token or TOKEN)
        backend  = provider.get_backend()

    # Noise model for ranking layouts: live calibration if possible
    if noise_model is None:
        if token or TOKEN:
            noise_model = EmeraldNoiseModel.from_iqm(token or TOKEN, quantum_computer,
                                                     SERVER_URL)
        else:
            noise_model = EmeraldNoiseModel()
            print("[run_qiskit] no token: ranking layouts with published Emerald medians")

    # Calibration-informed zero-SWAP placement (enumerate all exact embeddings,
    # rank them by simulated LER under the calibrated noise model)
    if layout is None:
        layout, _ = find_best_layout(stim_circ, backend, noise_model)
    print(f"[run_qiskit] layout on {layout.physical_qubits} "
          f"(simulated P_L = {layout.ler:.4f} +- {layout.ler_err:.4f})")

    # Transpile with that layout and routing disabled; raises if any SWAP,
    # off-coupler gate or measurement-order change would occur
    qc_t, report = transpile_zero_swap(qc, backend, layout.stim_to_phys, stim_to_dense,
                                       meas_order, optimization_level=optimization_level)
    print(f"[run_qiskit] transpiled: {report['swaps']} SWAPs, "
          f"{report['two_qubit_gates']}/{report['two_qubit_gates_ideal']} CZ, "
          f"depth {report['depth']}")

    extra = dict(layout=layout, transpile_report=report,
                 noise_model_source=noise_model.source)
    if dry_run:
        return dict(extra, circuit=qc_t)

    # Run
    job        = backend.run(qc_t, shots=shots)
    result     = job.result()
    counts     = result.get_counts()
    # Keep track of the calibration the job ran with (drift!), when exposed
    cal_id = getattr(job, "calibration_set_id", None) or \
        getattr(getattr(result, "metadata", None) or {}, "get", lambda *_: None)("calibration_set_id")
    if cal_id is not None:
        extra["calibration_set_id"] = str(cal_id)

    # Convert counts dict → (shots, num_measurements) numpy array
    raw_meas   = counts_to_measurement_array(counts, stim_circ.num_measurements,
                                               shots)
    # m2d converter must come from the circuit that actually ran
    # keep the raw data: needed to re-decode, re-characterize and for the
    # history-aware decoder (no-reset: raw m_(k-1) is the ancilla state)
    extra.update(raw_meas=raw_meas, counts=counts)
    return dict(extract_syndromes(raw_meas, stim_circ), **extra)



def run_qrisp(
    distance: int    = 3,
    rounds: int      = 1,
    shots: int       = 5_000,
    token: str       = None,
    quantum_computer: str = "emerald",
    no_reset: bool   = True,
) -> dict:
    """
    Build in qrisp. Runs the surface code on IQM Emerald via IQM Resonance.

        Stim circuit → Qrisp DetectorExperiment → IQMBackend → run
        → detection events + observable flips

    The DetectorExperiment stores the info the detectors and observables itself, so
    counts_to_measurement_array() and extract_syndromes() are not needed here.

    Parameters
    ----------
    distance         : code distance (use 3 for emerald)
    rounds           : measurement rounds
    shots            : number of circuit executions
    token            : IQM Resonance API token 
    quantum_computer : e.g. "emerald"
    no_reset         : build the no-reset circuit (required on Emerald)

    Returns
    -------
    syndromes : dict  same keys as extract_syndromes() from qiskit function
                "det_events" (shots, num_detectors), "obs_flips" (shots, num_observables)
    """
    import os
    from iqm.qrisp_iqm import IQMBackend
    from stim_to_qrisp import stim_to_detector_experiment

    os.environ["IQM_TOKEN"] = token or TOKEN

    # Build circuits, exchange for make_qrisp_experiment() if desired
    stim_circ  = make_stim_circuit(distance, rounds, no_reset=no_reset, noise=None)

    # CREATE AN EXPERIMENT TO BE EXECUTED ON HARDWARE
    experiment = []
    #experiment = stim_to_detector_experiment(stim_circ, no_reset=no_reset)
    #experiment = make_qrisp_experiment(distance, rounds, no_reset=no_reset)

    # Connect to IQM Resonance
    # Layout: think about the pass_manager. iqm.qrisp_iqm.vf2pp_layout finds an
    # exact embedding on the coupling map (zero SWAPs) or fails.
    backend    = IQMBackend(
        server_url=SERVER_URL,
        device_instance=quantum_computer,
    # further arguments
    )

    # Run the experiment on the backend and collect detection events + observable flips
    # TO BE DONE
    det_events, obs_flips = [], []

    return {"det_events": det_events, "obs_flips": obs_flips}



def decode_hardware_results(
    syndromes: dict,
    noisy_circuit: Optional[stim.Circuit] = None,
    dem: Optional[stim.DetectorErrorModel] = None,
) -> tuple[float, float]:
    """
    Decodes the output of run_qiskit() or run_qrisp() with PyMatching (MWPM) and
    returns the logical error probability for this round count.

    PyMatching weights its matching graph with a detector error model (DEM), and the
    DEM must describe the circuit that ran on hardware (same detectors, same order).
    The hardware circuit is built with noise=None, so its own DEM would be empty:
    pass a NOISY copy of that same circuit, e.g.

        noisy = make_stim_circuit(distance, rounds, no_reset=True, noise=DEFAULT_NOISE)
        ler, err = decode_hardware_results(syndromes, noisy_circuit=noisy)

    Calibration-based noise models improve the decoding: they enter through the
    noise of `noisy_circuit` (or of the `dem`), not through this function.

    Parameters
    ----------
    syndromes     : dict  output of run_qiskit() / run_qrisp(). Uses
                    "det_events" (shots, num_detectors) and "obs_flips" (shots, num_observables).
    noisy_circuit : stim.Circuit  the circuit that ran on hardware (no-reset detector
                    definitions on Emerald) with a noise model. The DEM is derived from it.
    dem           : stim.DetectorErrorModel  alternative to noisy_circuit, e.g. one that
                    Qrisp derives from its DetectorExperiment. It must be decomposed into
                    graphlike errors (detector_error_model(decompose_errors=True)):
                    PyMatching silently ignores undecomposed errors that flip >2 detectors.
    Give exactly one of noisy_circuit / dem.

    Returns
    -------
    ler : float   logical error probability after this many rounds (NOT per round): the
                  fraction of shots where the decoder's predicted observable flip differs
                  from the measured one. Fit over several round counts to get the
                  logical error rate (see README).
    err : float   1-sigma binomial statistical error on ler.
    """
    if (noisy_circuit is None) == (dem is None):
        raise ValueError("Give exactly one of noisy_circuit / dem.")

    # Normalise the syndromes (run_qrisp() may hand over plain lists).
    # C-contiguous uint8 is what PyMatching's decode_batch expects.
    det_events = np.ascontiguousarray(syndromes["det_events"], dtype=np.uint8)
    obs_flips  = np.asarray(syndromes["obs_flips"], dtype=bool)
    if obs_flips.ndim == 1:                      # one observable given as a flat list
        obs_flips = obs_flips[:, None]
    if det_events.ndim != 2 or obs_flips.ndim != 2:
        raise ValueError(
            "det_events must have shape (shots, num_detectors) and obs_flips "
            f"(shots, num_observables); got {det_events.shape} and {obs_flips.shape}."
        )
    shots = det_events.shape[0]
    if shots == 0 or obs_flips.shape[0] != shots:
        raise ValueError(
            f"Need the same, non-zero number of shots in det_events ({det_events.shape[0]}) "
            f"and obs_flips ({obs_flips.shape[0]})."
        )

    if dem is None:
        # decompose_errors: PyMatching needs edges (<= 2 detectors); this raises instead of
        #   silently degrading if a decomposition is impossible.
        # approximate_disjoint_errors: lets PAULI_CHANNEL_* noise (e.g. from calibration
        #   data) through; harmless for plain depolarizing noise.
        dem = noisy_circuit.detector_error_model(
            decompose_errors=True, approximate_disjoint_errors=True
        )

    if dem.num_errors == 0:
        raise ValueError(
            "The DEM has no error mechanisms, so there is nothing to weight the decoder "
            "with. Was it built from the noiseless hardware circuit (noise=None)?"
        )
    if dem.num_detectors != det_events.shape[1]:
        raise ValueError(
            f"The DEM has {dem.num_detectors} detectors but det_events has "
            f"{det_events.shape[1]} columns. The DEM must come from the circuit that ran "
            "(same distance, rounds, basis and no_reset)."
        )

    matcher   = pymatching.Matching.from_detector_error_model(dem)
    predicted = matcher.decode_batch(det_events).astype(bool)
    if predicted.shape != obs_flips.shape:
        # Without this check numpy could broadcast the mismatch away and report 0 errors.
        raise ValueError(
            f"The decoder predicts {predicted.shape[1]} observable(s) but obs_flips has "
            f"{obs_flips.shape[1]}."
        )

    # A shot is a logical failure if ANY observable is mispredicted.
    failed = np.any(predicted != obs_flips, axis=1)
    n_fail = int(failed.sum())
    ler    = n_fail / shots


    p_err = ler if 0 < n_fail < shots else (n_fail + 0.5) / (shots + 1)
    err   = float(np.sqrt(p_err * (1.0 - p_err) / shots))
    return float(ler), err