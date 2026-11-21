import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

from stim_to_qiskit import stim_to_qiskit
from extract_syndromes import extract_syndromes
from internal_helpers import counts_to_measurement_array
#Partie Cyril
from surface_code import make_stim_circuit
#Partie Victor
from run_on_hardware import decode_hardware_results

DISTANCE = 3
ROUNDS_LIST = [2, 3, 4, 5]
SHOTS = 20_000
P = 1e-3

NON_GATES = {"measure", "barrier", "reset", "delay"}


def decode(syndromes: dict, stim_noisy) -> tuple[float, float]:
    """Décode les syndromes avec le DEM du circuit bruité et renvoie (P_error, incertitude)."""
    return decode_hardware_results(syndromes, noisy_circuit=stim_noisy)


def stim_noise(p: float) -> dict:
    """Renvoie le bruit Stim (dépolarisation après les portes + erreur de lecture) de probabilité p."""
    return {
        "after_clifford_depolarization": p,
        "before_measure_flip_probability": p,
    }


def aer_noise_model(p: float, qc) -> NoiseModel:
    """TEMPORAIRE (en attendant Alexis) : modèle Aer équivalent à stim_noise(p), sur les portes présentes dans qc."""
    # Stim DEPOLARIZE_n(p) <=> Qiskit depolarizing_error(p * 4^n / (4^n - 1), n)
    gates_1q, gates_2q = set(), set()
    for inst in qc.data:
        op = inst.operation
        if op.name in NON_GATES:
            continue
        (gates_1q if op.num_qubits == 1 else gates_2q).add(op.name)

    nm = NoiseModel()
    if gates_1q:
        nm.add_all_qubit_quantum_error(depolarizing_error(4 * p / 3, 1), sorted(gates_1q))
    if gates_2q:
        nm.add_all_qubit_quantum_error(depolarizing_error(16 * p / 15, 2), sorted(gates_2q))
    nm.add_all_qubit_readout_error(ReadoutError([[1 - p, p], [p, 1 - p]]))
    return nm


def run_aer(qc, shots: int, noise_model=None, backend=None) -> dict:
    """Simule le circuit Qiskit (bruit donné ou bruit d'un backend) et renvoie les counts."""
    if backend is not None:
        sim = AerSimulator.from_backend(backend)
        qc = transpile(qc, backend=backend, optimization_level=1)
    else:
        sim = AerSimulator(method="stabilizer", noise_model=noise_model)
    return sim.run(qc, shots=shots).result().get_counts()


def simulate_syndromes(qc, stim_hw, shots: int, noise_model=None, backend=None) -> dict:
    """Simule puis convertit en syndromes par le même chemin que run_qiskit() sur hardware."""
    counts = run_aer(qc, shots, noise_model=noise_model, backend=backend)
    raw_meas = counts_to_measurement_array(counts, stim_hw.num_measurements, shots)
    return extract_syndromes(raw_meas, stim_hw)


def check_noiseless(qc, stim_hw) -> bool:
    """Vérifie que, sans bruit, aucun détecteur ne s'allume et que l'observable reste à 0."""
    synd = simulate_syndromes(qc, stim_hw, shots=1_000)
    dets = np.asarray(synd["det_events"], dtype=bool)
    obs = np.asarray(synd["obs_flips"], dtype=bool)
    if dets.any() or obs.any():
        print(f"  ÉCHEC sans bruit : détecteurs {np.flatnonzero(dets.any(axis=0)).tolist()}, "
              f"observable non nulle dans {int(obs.reshape(len(obs), -1).any(axis=1).sum())} shots")
        return False
    return True


def run_one(rounds: int, backend=None) -> dict:
    """Lance l'expérience complète pour un nombre de rounds et compare Qiskit/Aer à Stim seul."""
    #Les circuits de Cyril avec ou sans bruit
    stim_hw = make_stim_circuit(DISTANCE, rounds, no_reset=True, noise=None)
    stim_noisy = make_stim_circuit(DISTANCE, rounds, no_reset=True, noise=stim_noise(P))
    qc, stim_to_dense, meas_order = stim_to_qiskit(stim_hw)

    if not check_noiseless(qc, stim_hw):
        return {"rounds": rounds, "ok": False}

    #Le noise model d'Alexis (temporaire)
    noise_model = None if backend is not None else aer_noise_model(P, qc)

    synd_q = simulate_syndromes(qc, stim_hw, SHOTS, noise_model=noise_model, backend=backend)
    dets_s, obs_s = stim_noisy.compile_detector_sampler().sample(
        SHOTS, separate_observables=True
    )
    synd_s = {"det_events": dets_s, "obs_flips": obs_s}

    diff = np.abs(
        np.asarray(synd_q["det_events"], dtype=float).mean(axis=0) - dets_s.mean(axis=0)
    )

    return {
        "rounds": rounds,
        "ok": True,
        "qiskit": decode(synd_q, stim_noisy),
        "stim": decode(synd_s, stim_noisy),
        "max_det_diff": float(diff.max()),
        "worst_det": int(diff.argmax()),
    }


def main(backend=None) -> list[dict]:
    """Exécute run_one pour chaque nombre de rounds, affiche les résultats et les renvoie."""
    print(f"d = {DISTANCE}, p = {P:.0e}, shots = {SHOTS}, "
          f"backend = {getattr(backend, 'name', None)}")
    results = []
    for rounds in ROUNDS_LIST:
        r = run_one(rounds, backend)
        results.append(r)
        if not r["ok"]:
            print(f"  rounds = {rounds} : échec du test sans bruit")
            continue
        (ler_q, err_q), (ler_s, err_s) = r["qiskit"], r["stim"]
        print(f"  rounds = {rounds} : Qiskit {ler_q:.2e} ± {err_q:.1e}  "
              f"Stim {ler_s:.2e} ± {err_s:.1e}  "
              f"écart max détecteur {r['max_det_diff']:.1e} (n°{r['worst_det']})")
    return results


if __name__ == "__main__":
    main()