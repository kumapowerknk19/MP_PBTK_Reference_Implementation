from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from scipy.linalg import expm

ORGANS = ("lung", "kidney", "spleen", "liver")
BLOOD_POOLS = ORGANS + ("other", "vein", "artery")


def load_config(path: str | Path | None = None) -> dict:
    if path is None:
        path = Path(__file__).resolve().parents[1] / "config" / "reference_config.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass(frozen=True)
class Kinetics:
    k_a_mL_per_cell_h: float
    k_b_prime_per_h: float
    f_abs: float
    turnover_time_h: float

    @property
    def k_y_per_h(self) -> float:
        return 1.0 / self.turnover_time_h


def particle_mass_ug(diameter_um: float, density_g_cm3: float) -> float:
    """Mass of a spherical particle [ug/particle]."""
    diameter_cm = diameter_um * 1e-4
    return density_g_cm3 * np.pi / 6.0 * diameter_cm**3 * 1e6


class MassConservingPBTK:
    """Generic 18-state macrophage-explicit PBTK model.

    State order [ug]:
      0:4   macrophage-associated amounts in lung/kidney/spleen/liver
      4:11  vascular free-particle amounts in 7 blood pools
      11:18 vascular monocyte-associated amounts in 7 blood pools

    All internal transfers are equal-and-opposite. No post-absorption
    elimination, dissolution, or degradation is included.
    """

    def __init__(self, physiology: dict, kinetics: Kinetics):
        self.physiology = physiology
        self.kinetics = kinetics

        self.v = np.array(
            [physiology["blood_volumes_mL"][p] for p in BLOOD_POOLS], dtype=float
        )
        self.vt = np.array(
            [physiology["tissue_volumes_mL"][o] for o in ORGANS], dtype=float
        )
        self.nmacro = np.array(
            [physiology["macrophage_counts_cells"][o] for o in ORGANS], dtype=float
        )
        self.mono_density = float(physiology["monocyte_density_cells_per_mL"])
        self.nmono = self.mono_density * self.v
        self.macro_density = self.nmacro / self.vt

        # k_b,i = k_b' / macrophage density_i
        self.k_b_mL_per_cell_h = kinetics.k_b_prime_per_h / self.macro_density

        # Homeostasis: k_x,i * N_mono,i = k_y * N_macro,i
        self.k_x_per_h = kinetics.k_y_per_h * self.nmacro / self.nmono[:4]

        self.K = self._build_matrix()
        self._check_mass_conservation()

    def _add_transfer(self, source: int, target: int, rate_per_h: float):
        self.K[source, source] -= rate_per_h
        self.K[target, source] += rate_per_h

    def _build_matrix(self) -> np.ndarray:
        self.K = np.zeros((18, 18), dtype=float)

        # Blood-flow transport of free and monocyte-associated particles.
        for source, target, q_mL_min in self.physiology["flow_edges_mL_min"]:
            src = BLOOD_POOLS.index(source)
            dst = BLOOD_POOLS.index(target)
            q_mL_h = float(q_mL_min) * 60.0
            rate = q_mL_h / self.v[src]
            self._add_transfer(4 + src, 4 + dst, rate)
            self._add_transfer(11 + src, 11 + dst, rate)

        # Free -> monocyte-associated uptake in every vascular pool.
        mono_rate = self.kinetics.k_a_mL_per_cell_h * self.mono_density
        for j in range(7):
            self._add_transfer(4 + j, 11 + j, mono_rate)

        # Target-organ macrophage processes.
        for i in range(4):
            # Equivalent to k_b,i * N_macro,i * C_free,i
            macro_rate = (
                self.kinetics.k_b_prime_per_h * self.vt[i] / self.v[i]
            )
            self._add_transfer(4 + i, i, macro_rate)
            self._add_transfer(11 + i, i, self.k_x_per_h[i])
            self._add_transfer(i, 4 + i, self.kinetics.k_y_per_h)

        return self.K

    def _check_mass_conservation(self):
        residual = float(np.max(np.abs(self.K.sum(axis=0))))
        if residual > 1e-10:
            raise RuntimeError(
                f"Internal-transfer mass conservation failed: {residual:.3e}"
            )

    def transition(self, dt_h: float) -> np.ndarray:
        return expm(self.K * float(dt_h))

    def macrophage_burden_ug_per_cell(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=float)[:4] / self.nmacro

    def macrophage_burden_pg_per_cell(self, z: np.ndarray) -> np.ndarray:
        return self.macrophage_burden_ug_per_cell(z) * 1e6


class MouseModel(MassConservingPBTK):
    def __init__(self, config: dict, kinetics: Kinetics):
        super().__init__(config["mouse"], kinetics)
        self.config = config

    def run_calibration_schedule(self) -> np.ndarray:
        d = self.config["mouse"]["dosing"]
        z = np.zeros(18, dtype=float)

        absorbed = float(d["oral_dose_ug"]) * self.kinetics.f_abs
        P = self.transition(float(d["dose_interval_h"]))

        # First dose at t=0.
        z[9] += absorbed
        for _ in range(1, int(d["n_doses"])):
            z = P @ z
            z[9] += absorbed

        z = self.transition(float(d["post_last_dose_h"])) @ z
        return z

    def calibration_outputs(self, z: np.ndarray) -> np.ndarray:
        wet = np.array(
            [self.config["mouse"]["organ_wet_masses_g"][o] for o in ORGANS],
            dtype=float,
        )
        z = np.asarray(z, dtype=float)
        organ_ug = z[:4] + z[4:8] + z[11:15]
        organ_conc = organ_ug / wet

        vv = self.config["mouse"]["blood_volumes_mL"]["vein"]
        va = self.config["mouse"]["blood_volumes_mL"]["artery"]
        blood_ug = z[9] + z[10] + z[16] + z[17]
        blood_conc = blood_ug / (vv + va)
        return np.r_[organ_conc, blood_conc]


class HumanModel(MassConservingPBTK):
    def __init__(self, config: dict, kinetics: Kinetics):
        super().__init__(config["human"], kinetics)
        self.config = config
        self.tissue_density = float(config["human"]["tissue_density_g_mL"])

    def periodic_endpoints(
        self,
        oral_dose_ug_day: float,
        days: list[int],
        bolus_hour: float = 12.0,
    ) -> dict[int, np.ndarray]:
        """Exact repeated-daily affine recurrence using matrix powers."""
        days = sorted(set(int(d) for d in days))
        absorbed = float(oral_dose_ug_day) * self.kinetics.f_abs

        pulse = np.zeros(18, dtype=float)
        pulse[9] = absorbed

        P_day = self.transition(24.0)
        g = self.transition(24.0 - bolus_hour) @ pulse

        T = np.eye(19, dtype=float)
        T[:18, :18] = P_day
        T[:18, 18] = g

        y = np.r_[np.zeros(18, dtype=float), 1.0]
        out = {0: y[:18].copy()} if 0 in days else {}
        prev = 0
        cache = {}

        for day in [d for d in days if d > 0]:
            gap = day - prev
            if gap not in cache:
                cache[gap] = np.linalg.matrix_power(T, gap)
            y = cache[gap] @ y
            out[day] = y[:18].copy()
            prev = day

        return out

    def organ_burden_ng_g(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        macro_ug_per_mL = z[:4] / self.vt
        return macro_ug_per_mL * 1000.0 / self.tissue_density

    def systemic_blood_ng_mL(self, z: np.ndarray) -> float:
        z = np.asarray(z, dtype=float)
        vv = self.config["human"]["blood_volumes_mL"]["vein"]
        va = self.config["human"]["blood_volumes_mL"]["artery"]
        mass_ug = z[9] + z[10] + z[16] + z[17]
        return float(mass_ug / (vv + va) * 1000.0)


def reference_kinetics(config: dict) -> Kinetics:
    k = config["reference_kinetics"]
    return Kinetics(
        k_a_mL_per_cell_h=float(k["k_a_mL_per_cell_h"]),
        k_b_prime_per_h=float(k["k_b_prime_per_h"]),
        f_abs=float(k["f_abs"]),
        turnover_time_h=float(k["turnover_time_h"]),
    )
