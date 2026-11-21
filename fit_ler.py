# -*- coding: utf-8 -*-
"""
fit_ler.py

Logical error rate per round from P_L measured at several round counts.

README model (Google, Nature 2021 / 2023):
    2 P_L(n) = 1 - (1 - 2 eps_L)^n                      ("readme")
Optional SPAM-robust variant (state prep / final readout errors give an
offset that does not grow with n):
    2 P_L(n) = 1 - A (1 - 2 eps_L)^n                    ("offset")

Both are weighted least-squares fits on the logical fidelity F = 1 - 2 P_L,
using the binomial error of each P_L. Report eps_L with its 1-sigma error and
say which model you used; if they disagree, the offset model is the safer one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import curve_fit


@dataclass
class LERFit:
    eps: float
    eps_err: float
    A: float
    A_err: float
    model: str
    chi2_red: float
    rounds: np.ndarray
    P: np.ndarray
    P_err: np.ndarray

    def __str__(self):
        a = f", A = {self.A:.3f} +- {self.A_err:.3f}" if self.model == "offset" else ""
        return (f"eps_L = {self.eps:.4f} +- {self.eps_err:.4f} per round "
                f"({self.model} model{a}, chi2/dof = {self.chi2_red:.2f})")


def binomial_err(P: float, shots: int) -> float:
    p = min(max(P, 0.5 / shots), 1 - 0.5 / shots)
    return float(np.sqrt(p * (1 - p) / shots))


def fit_logical_error_rate(rounds: Sequence[int], P: Sequence[float],
                           P_err: Optional[Sequence[float]] = None,
                           shots: Optional[Sequence[int]] = None,
                           model: str = "readme") -> LERFit:
    """Fit eps_L. Give P_err, or shots to compute binomial errors."""
    n = np.asarray(rounds, float)
    P = np.asarray(P, float)
    if P_err is None:
        P_err = [binomial_err(p, s) for p, s in zip(P, shots)] if shots is not None \
            else np.full_like(P, 0.01)
    P_err = np.asarray(P_err, float)
    if np.any(P >= 0.5):
        raise ValueError("Some P_L >= 0.5: those round counts carry no information, drop them.")

    F, F_err = 1 - 2 * P, 2 * P_err
    if model == "readme":
        f = lambda n, eps: (1 - 2 * eps) ** n
        p0 = [max(1e-4, (1 - F[0] ** (1 / n[0])) / 2)]
    elif model == "offset":
        f = lambda n, eps, A: A * (1 - 2 * eps) ** n
        p0 = [max(1e-4, (1 - F[0] ** (1 / n[0])) / 2), 1.0]
    else:
        raise ValueError("model must be 'readme' or 'offset'")
    popt, pcov = curve_fit(f, n, F, p0=p0, sigma=F_err, absolute_sigma=True,
                           bounds=([0, 0] if model == "offset" else [0],
                                   [0.5, 1.5] if model == "offset" else [0.5]))
    res = (F - f(n, *popt)) / F_err
    dof = max(1, len(n) - len(popt))
    err = np.sqrt(np.diag(pcov))
    return LERFit(eps=float(popt[0]), eps_err=float(err[0]),
                  A=float(popt[1]) if model == "offset" else 1.0,
                  A_err=float(err[1]) if model == "offset" else 0.0,
                  model=model, chi2_red=float(np.sum(res ** 2) / dof),
                  rounds=n, P=P, P_err=P_err)


def plot_ler(fits: dict[str, LERFit], path: str = "ler_vs_rounds.png",
             title: str = "Logical error probability vs rounds") -> str:
    """One curve per label (e.g. decoder or layout). Saves a PNG, returns its path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, fit in fits.items():
        nn = np.linspace(0, fit.rounds.max() + 0.5, 100)
        line = ax.errorbar(fit.rounds, fit.P, yerr=fit.P_err, fmt="o", capsize=3)
        ax.plot(nn, 0.5 * (1 - fit.A * (1 - 2 * fit.eps) ** nn), color=line[0].get_color(),
                label=f"{label}: $\\varepsilon_L$ = {100 * fit.eps:.2f} $\\pm$ {100 * fit.eps_err:.2f} %")
    ax.axhline(0.5, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("rounds")
    ax.set_ylabel("logical error probability $P_L$")
    ax.set_title(title)
    ax.set_ylim(0, 0.55)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
