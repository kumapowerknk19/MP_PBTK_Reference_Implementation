from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, least_squares

from .model import Kinetics, MouseModel, load_config


def load_targets(path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        path = Path(__file__).resolve().parents[1] / "input" / "mouse_calibration_targets.csv"
    return pd.read_csv(path)


def predictions_at_unit_fabs(
    config: dict,
    k_a: float,
    k_b_prime: float,
    turnover_time_h: float,
) -> np.ndarray:
    kinetics = Kinetics(
        k_a_mL_per_cell_h=k_a,
        k_b_prime_per_h=k_b_prime,
        f_abs=1.0,
        turnover_time_h=turnover_time_h,
    )
    model = MouseModel(config, kinetics)
    return model.calibration_outputs(model.run_calibration_schedule())


def profile_f_abs(
    pred_at_fabs_1: np.ndarray,
    observed: np.ndarray,
    se: np.ndarray,
    bounds: tuple[float, float],
) -> float:
    """Weighted least-squares optimum for a linear multiplicative f_abs."""
    w = 1.0 / (se**2)
    num = np.sum(pred_at_fabs_1 * observed * w)
    den = np.sum(pred_at_fabs_1**2 * w)
    if den <= 0:
        return float(bounds[0])
    return float(np.clip(num / den, bounds[0], bounds[1]))


def evaluate_candidate(
    config: dict,
    targets: pd.DataFrame,
    k_a: float,
    k_b_prime: float,
    turnover_time_h: float,
    f_abs_bounds: tuple[float, float],
):
    observed = targets["observed"].to_numpy(float)
    se = targets["SE"].to_numpy(float)

    g = predictions_at_unit_fabs(config, k_a, k_b_prime, turnover_time_h)
    f_abs = profile_f_abs(g, observed, se, f_abs_bounds)
    pred = f_abs * g
    residuals = (pred - observed) / se
    Q = float(np.sum(residuals**2))

    return {
        "k_a": float(k_a),
        "k_b_prime": float(k_b_prime),
        "f_abs": float(f_abs),
        "predicted": pred,
        "standardized_residuals": residuals,
        "Q": Q,
        "sqrt_Q": float(np.sqrt(Q)),
        "F_fit": float(np.sqrt(Q / 2.0)),  # n=5, p=3
    }


def calibrate(
    config: dict | None = None,
    targets: pd.DataFrame | None = None,
    turnover_time_h: float | None = None,
    seed: int = 42,
    quick: bool = False,
):
    config = load_config() if config is None else config
    targets = load_targets() if targets is None else targets
    turnover_time_h = (
        float(config["reference_kinetics"]["turnover_time_h"])
        if turnover_time_h is None
        else float(turnover_time_h)
    )

    b = config["calibration_bounds"]
    ka_bounds = tuple(map(float, b["k_a_mL_per_cell_h"]))
    kb_bounds = tuple(map(float, b["k_b_prime_per_h"]))
    fabs_bounds = tuple(map(float, b["f_abs"]))

    log_bounds = [
        (np.log10(ka_bounds[0]), np.log10(ka_bounds[1])),
        (np.log10(kb_bounds[0]), np.log10(kb_bounds[1])),
    ]

    def obj(x):
        r = evaluate_candidate(
            config,
            targets,
            10.0 ** x[0],
            10.0 ** x[1],
            turnover_time_h,
            fabs_bounds,
        )
        return r["Q"]

    de = differential_evolution(
        obj,
        log_bounds,
        seed=seed,
        popsize=8 if quick else 15,
        maxiter=25 if quick else 120,
        tol=1e-6,
        polish=False,
        updating="immediate",
        workers=1,
    )

    def residual_vec(x):
        r = evaluate_candidate(
            config,
            targets,
            10.0 ** x[0],
            10.0 ** x[1],
            turnover_time_h,
            fabs_bounds,
        )
        return r["standardized_residuals"]

    ls = least_squares(
        residual_vec,
        de.x,
        bounds=(
            np.array([b[0] for b in log_bounds]),
            np.array([b[1] for b in log_bounds]),
        ),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=300 if quick else 2000,
    )

    result = evaluate_candidate(
        config,
        targets,
        10.0 ** ls.x[0],
        10.0 ** ls.x[1],
        turnover_time_h,
        fabs_bounds,
    )
    result["turnover_time_h"] = turnover_time_h
    result["seed"] = seed
    result["quick"] = bool(quick)
    return result
