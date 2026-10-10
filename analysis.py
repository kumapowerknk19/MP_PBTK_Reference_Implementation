from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import qmc, spearmanr

from .model import (
    ORGANS,
    Kinetics,
    HumanModel,
    load_config,
    particle_mass_ug,
    reference_kinetics,
)
from .calibration import calibrate, load_targets, evaluate_candidate


def human_burden_at_years(
    config: dict,
    kinetics: Kinetics,
    particles_per_day: float,
    years: list[int],
    particle_diameter_um: float | None = None,
    particle_density_g_cm3: float | None = None,
):
    diameter = (
        float(config["particle"]["diameter_um"])
        if particle_diameter_um is None
        else float(particle_diameter_um)
    )
    density = (
        float(config["particle"]["density_g_cm3"])
        if particle_density_g_cm3 is None
        else float(particle_density_g_cm3)
    )
    m_particle_ug = particle_mass_ug(diameter, density)
    oral_ug_day = float(particles_per_day) * m_particle_ug

    model = HumanModel(config, kinetics)
    days = [int(y) * 365 for y in years]
    states = model.periodic_endpoints(oral_ug_day, days)

    rows = []
    for year, day in zip(years, days):
        z = states[day]
        ng_g = model.organ_burden_ng_g(z)
        pg_cell = model.macrophage_burden_pg_per_cell(z)
        blood = model.systemic_blood_ng_mL(z)
        for organ, b, q in zip(ORGANS, ng_g, pg_cell):
            rows.append({
                "year": int(year),
                "organ": organ,
                "burden_ng_g": float(b),
                "burden_pg_per_cell": float(q),
                "systemic_blood_ng_mL": float(blood),
                "particles_per_day": float(particles_per_day),
                "oral_mass_ug_day": float(oral_ug_day),
            })
    return pd.DataFrame(rows)


def oat_sensitivity(config: dict | None = None):
    config = load_config() if config is None else config
    ref = reference_kinetics(config)
    factors = list(map(float, config["oat_factors"]))
    central = float(config["exposure_particles_per_day"]["Central"])
    year = 10

    ref_df = human_burden_at_years(config, ref, central, [year])
    ref_map = ref_df.set_index("organ")["burden_ng_g"].to_dict()

    rows = []
    for parameter in ["k_a", "k_b_prime", "f_abs", "intake"]:
        for factor in factors:
            kin = ref
            particles = central

            if parameter == "k_a":
                kin = Kinetics(
                    ref.k_a_mL_per_cell_h * factor,
                    ref.k_b_prime_per_h,
                    ref.f_abs,
                    ref.turnover_time_h,
                )
            elif parameter == "k_b_prime":
                kin = Kinetics(
                    ref.k_a_mL_per_cell_h,
                    ref.k_b_prime_per_h * factor,
                    ref.f_abs,
                    ref.turnover_time_h,
                )
            elif parameter == "f_abs":
                kin = Kinetics(
                    ref.k_a_mL_per_cell_h,
                    ref.k_b_prime_per_h,
                    ref.f_abs * factor,
                    ref.turnover_time_h,
                )
            elif parameter == "intake":
                particles = central * factor

            df = human_burden_at_years(config, kin, particles, [year])
            for _, r in df.iterrows():
                ref_b = ref_map[r["organ"]]
                rows.append({
                    "parameter": parameter,
                    "factor": factor,
                    "organ": r["organ"],
                    "burden_10y_ng_g": r["burden_ng_g"],
                    "reference_burden_10y_ng_g": ref_b,
                    "percent_change_vs_reference": 100.0 * (r["burden_ng_g"] / ref_b - 1.0),
                })

    return pd.DataFrame(rows)


def local_elasticity_pm10(config: dict | None = None):
    config = load_config() if config is None else config
    ref = reference_kinetics(config)
    central = float(config["exposure_particles_per_day"]["Central"])

    rows = []
    for parameter in ["k_a", "k_b_prime"]:
        dfs = {}
        for factor in [0.9, 1.1]:
            if parameter == "k_a":
                kin = Kinetics(
                    ref.k_a_mL_per_cell_h * factor,
                    ref.k_b_prime_per_h,
                    ref.f_abs,
                    ref.turnover_time_h,
                )
            else:
                kin = Kinetics(
                    ref.k_a_mL_per_cell_h,
                    ref.k_b_prime_per_h * factor,
                    ref.f_abs,
                    ref.turnover_time_h,
                )
            dfs[factor] = human_burden_at_years(config, kin, central, [10]).set_index("organ")

        for organ in ORGANS:
            lo = float(dfs[0.9].loc[organ, "burden_ng_g"])
            hi = float(dfs[1.1].loc[organ, "burden_ng_g"])
            E = (np.log(hi) - np.log(lo)) / (np.log(1.1) - np.log(0.9))
            rows.append({
                "parameter": parameter,
                "organ": organ,
                "elasticity_loglog_pm10pct": float(E),
            })
    return pd.DataFrame(rows)


def lhs_parameter_sampling(config: dict | None = None, profile: str = "quick", seed: int = 20260916):
    config = load_config() if config is None else config
    targets = load_targets()
    ref = config["reference_kinetics"]
    bounds = config["calibration_bounds"]

    n_broad = 50000 if profile == "manuscript" else 1000
    n_local = 10000 if profile == "manuscript" else 1000

    # Broad LHS: k_a and k_b' log10-uniform, f_abs linear.
    sampler = qmc.LatinHypercube(d=3, seed=seed)
    u = sampler.random(n_broad)

    ka_lo, ka_hi = np.log10(bounds["k_a_mL_per_cell_h"])
    kb_lo, kb_hi = np.log10(bounds["k_b_prime_per_h"])
    fa_lo, fa_hi = map(float, bounds["f_abs"])

    broad = pd.DataFrame({
        "k_a": 10.0 ** (ka_lo + u[:, 0] * (ka_hi - ka_lo)),
        "k_b_prime": 10.0 ** (kb_lo + u[:, 1] * (kb_hi - kb_lo)),
        "f_abs": fa_lo + u[:, 2] * (fa_hi - fa_lo),
    })

    # Local LHS: all linear within multiplicative windows around reference.
    lf = config["local_sampling_factors"]
    sampler2 = qmc.LatinHypercube(d=3, seed=seed + 1)
    u2 = sampler2.random(n_local)

    local_ranges = [
        (ref["k_a_mL_per_cell_h"] * lf["k_a"][0], ref["k_a_mL_per_cell_h"] * lf["k_a"][1]),
        (ref["k_b_prime_per_h"] * lf["k_b_prime"][0], ref["k_b_prime_per_h"] * lf["k_b_prime"][1]),
        (ref["f_abs"] * lf["f_abs"][0], ref["f_abs"] * lf["f_abs"][1]),
    ]
    local = pd.DataFrame({
        "k_a": local_ranges[0][0] + u2[:, 0] * (local_ranges[0][1] - local_ranges[0][0]),
        "k_b_prime": local_ranges[1][0] + u2[:, 1] * (local_ranges[1][1] - local_ranges[1][0]),
        "f_abs": local_ranges[2][0] + u2[:, 2] * (local_ranges[2][1] - local_ranges[2][0]),
    })

    observed = targets["observed"].to_numpy(float)
    se = targets["SE"].to_numpy(float)
    turnover = float(ref["turnover_time_h"])

    def evaluate_frame(df):
        Qs = []
        for r in df.itertuples(index=False):
            # Here f_abs is sampled explicitly rather than profiled.
            result = evaluate_candidate(
                config,
                targets,
                float(r.k_a),
                float(r.k_b_prime),
                turnover,
                (float(r.f_abs), float(r.f_abs)),
            )
            Qs.append(result["Q"])
        out = df.copy()
        out["Q"] = np.asarray(Qs)
        out["score_sqrtQ"] = np.sqrt(out["Q"])
        return out

    broad = evaluate_frame(broad)
    local = evaluate_frame(local)

    # Use the frozen reference Q, not the best random LHS draw.
    ref_eval = evaluate_candidate(
        config,
        targets,
        float(ref["k_a_mL_per_cell_h"]),
        float(ref["k_b_prime_per_h"]),
        turnover,
        (float(ref["f_abs"]), float(ref["f_abs"])),
    )
    Sref = np.sqrt(ref_eval["Q"])

    local["within_10pct"] = local["score_sqrtQ"] / Sref <= 1.10
    local["within_20pct"] = local["score_sqrtQ"] / Sref <= 1.20

    df20 = local[local["within_20pct"]].copy()
    corr_rows = []
    pairs = [
        ("k_a", "k_b_prime"),
        ("k_a", "f_abs"),
        ("k_b_prime", "f_abs"),
    ]
    for x, y in pairs:
        if len(df20) >= 3:
            rho, p = spearmanr(df20[x], df20[y])
        else:
            rho, p = np.nan, np.nan
        corr_rows.append({
            "x": x,
            "y": y,
            "rho": rho,
            "p_value": p,
            "n": len(df20),
        })

    return broad, local, pd.DataFrame(corr_rows)


def turnover_uncertainty(
    config: dict | None = None,
    profile: str = "quick",
    seed: int = 42,
):
    config = load_config() if config is None else config
    targets = load_targets()
    months = list(map(float, config["turnover_months"]))
    central = float(config["exposure_particles_per_day"]["Central"])

    rows = []
    human_rows = []

    for j, month in enumerate(months):
        turnover_h = month * 30.0 * 24.0
        result = calibrate(
            config=config,
            targets=targets,
            turnover_time_h=turnover_h,
            seed=seed + j,
            quick=(profile != "manuscript"),
        )
        rows.append({
            "turnover_month": month,
            "turnover_time_h": turnover_h,
            "k_a": result["k_a"],
            "k_b_prime": result["k_b_prime"],
            "f_abs": result["f_abs"],
            "Q": result["Q"],
        })

        kin = Kinetics(
            result["k_a"],
            result["k_b_prime"],
            result["f_abs"],
            turnover_h,
        )
        df = human_burden_at_years(config, kin, central, [10])
        for _, r in df.iterrows():
            human_rows.append({
                "turnover_month": month,
                "organ": r["organ"],
                "burden_10y_ng_g": r["burden_ng_g"],
            })

    profile_df = pd.DataFrame(rows)
    profile_df["delta_Q"] = profile_df["Q"] - profile_df["Q"].min()
    profile_df["within_reference"] = (
        profile_df["delta_Q"] <= float(config["profile_reference_delta_Q"])
    )

    human_df = pd.DataFrame(human_rows)
    ref4 = (
        human_df[human_df["turnover_month"] == 4.0]
        .set_index("organ")["burden_10y_ng_g"]
        .to_dict()
    )
    human_df["percent_change_vs_4mo"] = [
        100.0 * (b / ref4[o] - 1.0)
        for o, b in zip(human_df["organ"], human_df["burden_10y_ng_g"])
    ]

    return profile_df, human_df
