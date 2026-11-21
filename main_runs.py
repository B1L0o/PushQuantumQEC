# -*- coding: utf-8 -*-
"""
main_runs.py

Main memory-experiment runs on Emerald: d=3, Z basis, several round counts,
ONE fixed layout (taken from a smoke-test directory), all raw data saved.

    python main_runs.py --layout-from runs/<ts>_smoke --rounds 1 2 3 4 5 --shots 20000

Output: runs/<ts>_main/
    noise_model_start.json / noise_model_end.json   calibration before / after (drift)
    layout.json                                     the layout used
    r<k>.npz                                        det_events, obs_flips, raw_meas
    r<k>_meta.json                                  counts, transpile report, job info
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import time

import numpy as np

from emerald_noise import EmeraldNoiseModel
from layout_search import LayoutScore
from run_on_hardware import SERVER_URL, TOKEN, run_qiskit


def load_layout(smoke_dir: pathlib.Path) -> LayoutScore:
    d = json.loads((smoke_dir / "layout.json").read_text())
    names = {int(k): v for k, v in d["stim_to_name"].items()}
    phys = {int(k): int(v) for k, v in d["stim_to_phys"].items()}
    return LayoutScore(stim_to_phys=phys, stim_to_name=names, proxy=float("nan"),
                       ler=d.get("simulated_P_L_r3"), ler_err=0.0,
                       physical_qubits=sorted(set(names.values()), key=lambda s: int(s[2:])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout-from", help="reuse the layout of a smoke/main run dir")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="no --layout-from: search the best zero-SWAP layout without these qubits")
    ap.add_argument("--rounds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--shots", type=int, default=20_000)
    ap.add_argument("--chunk", type=int, default=10_000, help="max shots per job")
    ap.add_argument("--basis", default="Z")
    args = ap.parse_args()
    if not TOKEN:
        raise SystemExit("No token: export TOKEN_IQM=... or put it in ~/.iqm_token")

    from iqm.qiskit_iqm import IQMProvider
    backend = IQMProvider(SERVER_URL, quantum_computer="emerald", token=TOKEN).get_backend()
    out = pathlib.Path("runs") / (dt.datetime.now().strftime("%Y%m%d_%H%M%S") + "_main")
    out.mkdir(parents=True, exist_ok=True)
    nm = EmeraldNoiseModel.from_iqm(TOKEN, "emerald", SERVER_URL)
    nm.save(str(out / "noise_model_start.json"))
    if args.layout_from:
        layout = load_layout(pathlib.Path(args.layout_from))
    else:
        from layout_search import find_best_layout
        from surface_code import make_stim_circuit
        layout, _ = find_best_layout(make_stim_circuit(3, 3), backend, nm, exclude=args.exclude)
    (out / "layout.json").write_text(json.dumps(
        {"stim_to_name": layout.stim_to_name, "stim_to_phys": layout.stim_to_phys,
         "simulated_P_L_r3": layout.ler, "excluded": args.exclude,
         "from": args.layout_from}, indent=1))
    print(f"[main] layout {layout.physical_qubits} | {nm.source}", flush=True)

    for r in args.rounds:
        t0 = time.time()
        parts, metas = [], []
        left = args.shots
        while left > 0:
            n = min(args.chunk, left)
            syn = run_qiskit(3, r, shots=n, backend=backend, noise_model=nm, layout=layout)
            parts.append(syn)
            metas.append(dict(shots=n, counts=syn["counts"],
                              transpile_report=syn["transpile_report"],
                              calibration_set_id=syn.get("calibration_set_id"),
                              finished=dt.datetime.now().isoformat()))
            left -= n
        det = np.concatenate([p["det_events"] for p in parts])
        obs = np.concatenate([p["obs_flips"] for p in parts])
        raw = np.concatenate([p["raw_meas"] for p in parts])
        np.savez_compressed(out / f"r{r}.npz", det_events=det, obs_flips=obs, raw_meas=raw)
        (out / f"r{r}_meta.json").write_text(json.dumps(
            dict(rounds=r, basis=args.basis, shots=int(len(det)), jobs=metas,
                 layout=layout.stim_to_name), indent=1, default=str))
        print(f"[main] rounds={r}: {len(det)} shots in {time.time() - t0:.0f}s | "
              f"mean detector rate {det.mean():.3f} | raw logical flip {obs.mean():.3f}",
              flush=True)

    EmeraldNoiseModel.from_iqm(TOKEN, "emerald", SERVER_URL).save(str(out / "noise_model_end.json"))
    print(f"[main] done -> {out}/", flush=True)


if __name__ == "__main__":
    main()
