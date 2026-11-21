import numpy as np
import stim
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
from stim_to_qiskit import stim_to_qiskit
#Partie Cyril
from surface_code import make_stim_circuit          
#Partie Victor
from decoding import decode_hardware_results        

DISTANCE = 3
ROUNDS_LIST = [2, 3, 4, 5]
SHOTS = 20_000
P = 1e-3

def decode(det_events, obs_flips, stim_noisy):
    """Décode les détecteurs avec le DEM du circuit bruité et renvoie P_error et son incertitude."""
    #Le decode de Victor
    pass

def stim_noise(p: float) -> dict:
    """Renvoie le bruit Stim (dépolarisation après les portes + erreur de lecture) de probabilité p."""
    return {
        "after_clifford_depolarization": p,
        "before_measure_flip_probability": p,
    }

def aer_noise_model(p: float) -> NoiseModel:
    """Construit le modèle de bruit Aer équivalent à stim_noise(p)."""
    #Le modele d'Alexis
    pass


def run_qiskit(qc, shots: int, noise_model=None, backend=None) -> list[str]:
    """Simule le circuit Qiskit (bruit donné ou bruit d'un backend) et renvoie une bitstring par shot."""
    if backend is not None:
        sim = AerSimulator.from_backend(backend)
        qc = transpile(qc, backend=backend, optimization_level=1)
    else:
        sim = AerSimulator(method="stabilizer", noise_model=noise_model)
    return sim.run(qc, shots=shots, memory=True).result().get_memory()


def memory_to_measurements(memory: list[str], num_measurements: int) -> np.ndarray:
    """Convertit les bitstrings Qiskit (little-endian) en tableau booléen (shots, mesures) dans l'ordre Stim."""
    rows = [s.replace(" ", "")[::-1] for s in memory]
    meas = np.array([[c == "1" for c in r] for r in rows], dtype=np.bool_)
    if meas.shape[1] != num_measurements:
        raise ValueError(
            f"{meas.shape[1]} bits classiques Qiskit pour {num_measurements} mesures Stim"
        )
    return meas


def measurements_to_detectors(stim_circ: stim.Circuit, meas: np.ndarray):
    """Calcule les détecteurs et l'observable à partir des mesures brutes, selon le circuit Stim."""
    return stim_circ.compile_m2d_converter().convert(
        measurements=meas, separate_observables=True
    )

def run_one(rounds: int, backend=None) -> dict:
    """Lance l'expérience complète pour un nombre de rounds et compare Qiskit/Aer à Stim seul."""
    #Les circuits de Cyril avec ou sans bruit
    stim_hw = make_stim_circuit(DISTANCE, rounds, no_reset=True, noise=None)
    stim_noisy = make_stim_circuit(DISTANCE, rounds, no_reset=True, noise=stim_noise(P))
    qc = stim_to_qiskit(stim_hw)

    #Le noise model d'Alexis
    noise_model = None if backend is not None else aer_noise_model(P)

    meas = memory_to_measurements(
        run_qiskit(qc, SHOTS, noise_model=noise_model, backend=backend),
        stim_hw.num_measurements,
    )

    dets_q, obs_q = measurements_to_detectors(stim_hw, meas)
    dets_s, obs_s = stim_noisy.compile_detector_sampler().sample(
        SHOTS, separate_observables=True
    )

    diff = np.abs(dets_q.mean(axis=0) - dets_s.mean(axis=0))

    return {
        "rounds": rounds,
        "ok": True,
        "qiskit": decode(dets_q, obs_q, stim_noisy),
        "stim": decode(dets_s, obs_s, stim_noisy),
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
        if r["ok"]:
            print(f"  rounds = {rounds} : Qiskit {r['qiskit']}  Stim {r['stim']}  "
                  f"écart max détecteur {r['max_det_diff']:.1e} (n°{r['worst_det']})")
    return results


if __name__ == "__main__":
    main()