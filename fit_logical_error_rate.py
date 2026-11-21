"""
Logical error probability P_error (one decoded experiment, after n rounds)
-> logical error rate per round ε_L (fit over several round counts).

    2 P_error = 1 - (1 - 2 ε_L)^n_rounds        i.e.  F = 1 - 2 P_error = (1 - 2 ε_L)^n_rounds
"""

from typing import Sequence

import numpy as np
from scipy.optimize import curve_fit


def p_error_model(n_rounds, eps_L):
    """P_error attendu après n_rounds pour un taux d'erreur logique par round ε_L."""
    return 0.5 * (1.0 - (1.0 - 2.0 * eps_L) ** n_rounds)


def eps_single_point(p_error: float, n_rounds: int) -> float:
    """ε_L à partir d'un seul nombre de rounds : ε_L = ½ [1 - (1 - 2 P_error)^(1/n_rounds)]."""
    return 0.5 * (1.0 - (1.0 - 2.0 * p_error) ** (1.0 / n_rounds))


def fit_logical_error_rate(
    rounds: Sequence[int],
    p_error: Sequence[float],
    p_error_err: Sequence[float],
) -> dict:
    """Ajuste 2 P_error = 1 - (1 - 2 ε_L)^n_rounds sur plusieurs nombres de rounds et renvoie ε_L ± σ."""
    n = np.asarray(rounds, dtype=float)
    p = np.asarray(p_error, dtype=float)
    dp = np.asarray(p_error_err, dtype=float)
    if not (len(n) == len(p) == len(dp)):
        raise ValueError("rounds, p_error et p_error_err doivent avoir la même longueur.")

    keep = p < 0.5
    n, p, dp = n[keep], p[keep], dp[keep]
    if len(n) < 2:
        raise ValueError(f"Il faut au moins 2 nombres de rounds avec P_error < 0.5 (il en reste {len(n)}).")

    eps_points = np.array([eps_single_point(pi, ni) for pi, ni in zip(p, n)])

    popt, pcov = curve_fit(
        p_error_model, n, p,
        p0=[float(np.clip(np.median(eps_points), 1e-6, 0.49))],
        sigma=dp, absolute_sigma=True, bounds=([0.0], [0.5]),
    )
    eps, deps = float(popt[0]), float(np.sqrt(pcov[0, 0]))

    residuals = (p - p_error_model(n, eps)) / dp
    dof = len(n) - 1

    return {
        "eps_L": eps,
        "eps_L_err": deps,
        "chi2_red": float(np.sum(residuals ** 2) / dof),
        "rounds": n,
        "p_error": p,
        "p_error_err": dp,
        "eps_single_point": eps_points,
    }


def fit_from_simulation(results: list[dict], source: str = "stim") -> dict:
    """Fit directement sur la liste renvoyée par le simulateur (r[source] = (P_error, σ))."""
    ok = [r for r in results if r.get("ok")]
    return fit_logical_error_rate(
        [r["rounds"] for r in ok],
        [r[source][0] for r in ok],
        [r[source][1] for r in ok],
    )


def plot_fit(fit: dict, label: str = "", ax=None, show: bool = True):
    """Trace P_error en fonction du nombre de rounds, avec la courbe ajustée."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(fit["rounds"], fit["p_error"], yerr=fit["p_error_err"],
                fmt="o", capsize=3, label=f"{label} P_error décodé".strip())
    n = np.linspace(0, max(fit["rounds"]) + 1, 200)
    ax.plot(n, p_error_model(n, fit["eps_L"]),
            label=f"fit : ε_L = {fit['eps_L']:.2e} ± {fit['eps_L_err']:.1e}")
    ax.axhline(0.5, ls="--", lw=0.8, color="gray")
    ax.set_xlabel("Nombre de rounds")
    ax.set_ylabel("Logical error probability P_error")
    ax.set_ylim(0, 0.55)
    ax.legend()
    if show:
        plt.show()
    return ax


def summary(fit: dict, name: str = "") -> str:
    """Résumé texte du fit et des estimations point par point."""
    points = ", ".join(f"n={int(n)}: {e:.2e}" for n, e in zip(fit["rounds"], fit["eps_single_point"]))
    return (f"{name} ε_L = {fit['eps_L']:.3e} ± {fit['eps_L_err']:.1e} par round "
            f"(χ²_red = {fit['chi2_red']:.2f}) | point par point : {points}").strip()