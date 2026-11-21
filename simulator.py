#Partie Cyril
from surface_code import make_stim_circuit
#Partie Victor
from run_on_hardware import decode_hardware_results
#Partie Alexis
from emerald_noise import EmeraldNoiseModel
#Fit
from fit_logical_error_rate import fit_from_simulation, plot_fit, summary

DISTANCE = 3
ROUNDS_LIST = [2, 3, 4, 5]
SHOTS = 20_000


def emerald_baseline(noise_model: EmeraldNoiseModel, qubit_map=None,
                     idle_noise: bool = True) -> list[dict]:
    """P_error attendu sur Emerald pour chaque nombre de rounds, avec le bruit d'Alexis."""
    results = []
    for rounds in ROUNDS_LIST:
        #pas bruit
        stim_hw = make_stim_circuit(DISTANCE, rounds, no_reset=True, noise=None)
        #bruit
        stim_noisy = noise_model.apply(stim_hw, qubit_map=qubit_map, idle_noise=idle_noise)
        dets, obs = stim_noisy.compile_detector_sampler().sample(SHOTS, separate_observables=True)
        ler, err = decode_hardware_results(
            {"det_events": dets, "obs_flips": obs}, noisy_circuit=stim_noisy
        )
        results.append({"rounds": rounds, "ok": True, "stim": (ler, err)})
        print(f"  rounds = {rounds} : P_error {ler:.3e} ± {err:.1e}")
    return results


def main(noise_model: EmeraldNoiseModel = None, qubit_map=None,
         idle_noise: bool = True, plot: bool = True):
    """Calcule le baseline Emerald et son fit ε_L."""
    # Sans calibration live, les médianes publiées d'Emerald (voir emerald_noise.py).
    noise_model = noise_model or EmeraldNoiseModel.uniform()
    print(f"Baseline Emerald : d = {DISTANCE}, shots = {SHOTS}, "
          f"bruit = {noise_model.source}, idle = {idle_noise}")
    fit = fit_from_simulation(
        emerald_baseline(noise_model, qubit_map=qubit_map, idle_noise=idle_noise),
        source="stim",
    )
    print(summary(fit, "ε_L attendu sur Emerald :"))
    if plot:
        plot_fit(fit, "Simulation Emerald")
    return fit


if __name__ == "__main__":
    main()