from __future__ import annotations

import math
import os
import json
import sys
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
import pandas as pd

from problem_v2_spec import AERO_INPUT_COLS, AERO_OUTPUT_COLS, INPUT_COLS_V2
from branch_geometry_v2 import build_geom_from_row
import profile_timing as prof


PROJECT_ROOT = Path(__file__).resolve().parent.parent
AVL_SRC = PROJECT_ROOT / "avl_optimize_portable" / "src_avl_full"
if str(AVL_SRC) not in sys.path:
    sys.path.insert(0, str(AVL_SRC))

import AeroCoeff_AVL as ac  # type: ignore  # noqa: E402
import sizing as sz  # type: ignore  # noqa: E402


INFO_COLS = [
    "mtow_out", "m_V", "D_V", "m_pu", "m_fue", "p2w", "N",
    "m_const_f_w", "m_const_a_w", "m_const_v_w", "m_F", "m_SS",
    "cx", "cy", "mz", "mx_beta", "my_beta", "K", "center_mass",
    "alpha_bal", "delta_bal", "A", "dmz_dcy",
    "M_fue_kg", "L_mission_km", "L_cruise_km", "L_climb_km", "L_declimb_km",
    "t_cruise_h", "t_climb_h", "t_declimb_h", "q_fuel", "q_g_per_ton_km",
]

FREECAD_COLS = [
    "f_delta", "f_twist", "f_root_chord", "f_tip_chord",
    "f_x_loc_root_chord", "f_y_loc_root_chord", "f_z_loc_root_chord",
    "f_x_loc_tip_chord", "f_y_loc_tip_chord", "f_z_loc_tip_chord",
    "a_delta", "a_twist", "a_root_chord", "a_tip_chord",
    "a_x_loc_root_chord", "a_y_loc_root_chord", "a_z_loc_root_chord",
    "a_x_loc_tip_chord", "a_y_loc_tip_chord", "a_z_loc_tip_chord",
    "v_root_chord", "v_tip_chord",
    "v_x_loc_root_chord", "v_y_loc_root_chord", "v_z_loc_root_chord",
    "v_x_loc_tip_chord", "v_y_loc_tip_chord", "v_z_loc_tip_chord",
    "nose_f_aspect", "center_f_aspect", "tail_f_aspect",
    "fuse_diameter", "fuse_x_loc", "n_vertical", "n_fuse",
    "distance_two_fuse",
]


def candidate_row(des_par, kk: int) -> dict[str, float]:
    arr = np.asarray(des_par, dtype=float)
    row = arr[kk] if arr.ndim == 2 else arr
    if row.size != len(INPUT_COLS_V2):
        raise ValueError(f"Expected {len(INPUT_COLS_V2)} variables, got {row.size}")
    return {name: float(row[i]) for i, name in enumerate(INPUT_COLS_V2)}


def polyfit2d(alpha_vals: np.ndarray, delta_vals: np.ndarray, values: np.ndarray) -> np.ndarray:
    alpha_grid, delta_grid = np.meshgrid(alpha_vals, delta_vals)
    mat = np.array([np.ones(alpha_grid.size), alpha_grid.flatten(), delta_grid.flatten()]).T
    coeff, _, _, _ = np.linalg.lstsq(mat, values.flatten(), rcond=None)
    return coeff


def parse_values(text: str, default: str) -> np.ndarray:
    raw = text or default
    return np.array([float(x.strip()) for x in raw.split(",") if x.strip()], dtype=float)


def secondary_s_ref_from_mass(m0: float, p0: float) -> float:
    if abs(p0) < 1e-12:
        return 1.0
    return max(float(m0) / float(p0), 1e-12)


def secondary_cy_req_from_mass(case: dict[str, float], m0: float) -> float:
    q_cr = ac.inputValuesAtmospheric(case["V"], case["H"])
    s_ref = secondary_s_ref_from_mass(m0, float(case["p0"]))
    if abs(q_cr) < 1e-12:
        return 0.0
    return float(9.81 * float(m0) / s_ref / q_cr)


def with_secondary_inputs(case: dict[str, float], m0: float) -> dict[str, float]:
    out = dict(case)
    out["m0"] = float(m0)
    out["S_ref"] = secondary_s_ref_from_mass(m0, float(case["p0"]))
    out["cy_req"] = secondary_cy_req_from_mass(case, m0)
    return out


def apply_delta(f_geo: np.ndarray, a_geo: np.ndarray, delta: float) -> tuple[np.ndarray, np.ndarray]:
    f = f_geo.copy()
    a = a_geo.copy()
    if a[6] <= f[6]:
        a[5] = delta
    else:
        f[5] = delta
    return f, a


class AeroBackend(Protocol):
    def eval(
        self,
        case: dict[str, float],
        alpha: float,
        delta: float,
        kk: int,
        center_mass: float | None = None,
        beta: float = 0.0,
    ) -> tuple[float, float, float, float, float]:
        ...


class AvlBackend:
    def eval(
        self,
        case: dict[str, float],
        alpha: float,
        delta: float,
        kk: int,
        center_mass: float | None = None,
        beta: float = 0.0,
    ) -> tuple[float, float, float, float, float]:
        f0, a0, v, fuse, scheme_fuse = build_geom_from_row(case, ac)
        f, a = apply_delta(f0, a0, delta)
        flight = ac.input_flight_cond(case["V"], alpha, case["H"], beta)
        if center_mass is None:
            cx, cy, mx, my, mz = ac.aero_calc(f, a, v, fuse, scheme_fuse, flight, kk)
        else:
            cx, cy, mx, my, mz = ac.aero_calc(f, a, v, fuse, scheme_fuse, flight, kk, center_mass)
        return float(cx), float(cy), float(mz), float(mx), float(my)


class MlpBackend:
    _cache: dict[str, tuple[object, object, object]] = {}

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        if not self.model_dir.is_absolute():
            self.model_dir = PROJECT_ROOT / self.model_dir
        key = str(self.model_dir.resolve())
        if key not in self._cache:
            from tensorflow import keras
            from tensorflow.keras import layers

            orig_from_config = layers.BatchNormalization.from_config

            def patched_from_config(cls, config):
                config = dict(config)
                config.pop("renorm", None)
                config.pop("renorm_clipping", None)
                config.pop("renorm_momentum", None)
                return orig_from_config(config)

            layers.BatchNormalization.from_config = classmethod(patched_from_config)
            model = keras.models.load_model(self.model_dir / "aero_mlp_v2_best.keras", compile=False)
            sx = joblib.load(self.model_dir / "scaler_X.joblib")
            sy = joblib.load(self.model_dir / "scaler_Y.joblib")
            self._cache[key] = (model, sx, sy)
        self.model, self.sx, self.sy = self._cache[key]
        metrics_path = self.model_dir / "metrics.json"
        self.output_cols = list(AERO_OUTPUT_COLS)
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            file_cols = metrics.get("output_cols")
            if isinstance(file_cols, list) and file_cols:
                self.output_cols = [str(x) for x in file_cols]
        expected_x = len(AERO_INPUT_COLS)
        actual_x = int(getattr(self.sx, "n_features_in_", expected_x))
        if actual_x != expected_x:
            raise ValueError(
                f"MLP scaler expects {actual_x} inputs, but new20 requires {expected_x}. "
                "Retrain models/aero_mlp_v2 from a new data/aero_labeled_v2.csv."
            )

    def eval(
        self,
        case: dict[str, float],
        alpha: float,
        delta: float,
        kk: int,
        center_mass: float | None = None,
        beta: float = 0.0,
    ) -> tuple[float, float, float, float, float]:
        pred = self.predict_outputs(case, alpha, delta)
        cx = pred["cx"]
        cy = pred["cy"]
        mz_ref = pred["mz_ref"]
        mx_beta = pred["mx_beta"]
        my_beta = pred["my_beta"]
        return float(cx), float(cy), float(mz_ref), float(mx_beta), float(my_beta)

    def predict_outputs(self, case: dict[str, float], alpha: float, delta: float) -> dict[str, float]:
        return self.predict_outputs_batch(case, [alpha], [delta])[0]

    def predict_outputs_batch(self, case: dict[str, float], alpha: list[float] | np.ndarray, delta: list[float] | np.ndarray) -> list[dict[str, float]]:
        alpha_arr = np.asarray(alpha, dtype=float).reshape(-1)
        delta_arr = np.asarray(delta, dtype=float).reshape(-1)
        if alpha_arr.shape != delta_arr.shape:
            raise ValueError(f"alpha and delta batch shapes differ: {alpha_arr.shape} vs {delta_arr.shape}")
        vals = [case[k] for k in AERO_INPUT_COLS[:-2]]
        rows = []
        for a_i, d_i in zip(alpha_arr, delta_arr):
            row = list(vals)
            row.extend([float(a_i), float(d_i)])
            rows.append(row)
        x = np.asarray(rows, dtype=np.float32)
        with prof.timer("mlp_predict"):
            y_s = self.model.predict(self.sx.transform(x), verbose=0)
            y = self.sy.inverse_transform(y_s)
        cols = self.output_cols[: y.shape[1]]
        return [{name: float(row[i]) for i, name in enumerate(cols)} for row in y]


class SplitMlpBackend:
    def __init__(self, normal_model_dir: str | Path, duck_model_dir: str | Path):
        self.normal = MlpBackend(normal_model_dir)
        self.duck = MlpBackend(duck_model_dir)

    def select_backend(self, case: dict[str, float]) -> MlpBackend:
        return self.duck if float(case["a_S_rel"]) > 0.5 else self.normal

    def eval(
        self,
        case: dict[str, float],
        alpha: float,
        delta: float,
        kk: int,
        center_mass: float | None = None,
        beta: float = 0.0,
    ) -> tuple[float, float, float, float, float]:
        return self.select_backend(case).eval(case, alpha, delta, kk, center_mass=center_mass, beta=beta)

    def predict_outputs(self, case: dict[str, float], alpha: float, delta: float) -> dict[str, float]:
        return self.select_backend(case).predict_outputs(case, alpha, delta)

    def predict_outputs_batch(self, case: dict[str, float], alpha: list[float] | np.ndarray, delta: list[float] | np.ndarray) -> list[dict[str, float]]:
        return self.select_backend(case).predict_outputs_batch(case, alpha, delta)


def is_mlp_like_backend(backend: AeroBackend) -> bool:
    return isinstance(backend, (MlpBackend, SplitMlpBackend))


def mlp_batch_enabled() -> bool:
    return os.environ.get("NEW20_MLP_BATCH", "1").strip().lower() not in {"0", "false", "no", "off"}


def _k_from_prediction(pred: dict[str, float], cx: float, cy: float) -> float:
    use_direct = os.environ.get("NEW20_USE_DIRECT_K", "0").strip().lower() in {"1", "true", "yes", "on"}
    if use_direct and "K" in pred and np.isfinite(pred["K"]):
        return float(pred["K"])
    return float(cy / cx) if abs(cx) > 1e-12 else 0.0


def _cya_alpha(lift_surface_geo_char: np.ndarray, mach: float) -> float:
    aspect = float(lift_surface_geo_char[0])
    sweep = float(lift_surface_geo_char[1])
    if aspect <= 0.0:
        return 0.0
    span, root_chord, tip_chord = ac.lift_surface_def(lift_surface_geo_char)
    if abs(span) < 1e-12:
        return 0.0
    beta = math.sqrt(max(0.0, 1.0 - float(mach) ** 2))
    tan_khi_1to2 = math.tan(math.radians(abs(sweep))) - (root_chord - tip_chord) / span
    tg = math.sqrt(max(0.0, 0.9 * aspect ** 2 * (beta ** 2 + tan_khi_1to2) + 4.0))
    return float(2.0 * math.pi * aspect / (2.0 + tg))


def pitch_damping_omegaz(case: dict[str, float], center_mass: float) -> float:
    local = with_secondary_inputs(case, max(1.0, float(case["m0"])))
    f_geo, a_geo, _v_geo, _fuse_geo, _scheme_fuse = build_geom_from_row(local, ac)
    s_ref = float(local["S_ref"])
    if s_ref <= 0.0:
        return 0.0
    s2_rel = float(a_geo[6]) / s_ref
    if s2_rel <= 1e-12 or s2_rel >= 1.0 - 1e-12:
        return 0.0
    mac = ac.ref_dim_lift_surface(f_geo) if f_geo[6] >= a_geo[6] else ac.ref_dim_lift_surface(a_geo)
    mach, _re = ac.Mach_Reynolds_number(float(local["V"]), mac, float(local["H"]))
    _a_vol, l_arm = ac.vol_coeff(f_geo, a_geo, float(center_mass))
    s2_to_s1 = s2_rel / (1.0 - s2_rel)
    cya_f = _cya_alpha(f_geo, mach)
    cya_a = _cya_alpha(a_geo, mach)
    kv = float(os.environ.get("NEW20_DAMP_KV", "0.93"))
    if f_geo[6] >= a_geo[6]:
        mz_f = -cya_f / 4.0 * (1.0 - 2.0 * abs(float(center_mass))) ** 2 - (2.0 * math.pi - cya_f) / 16.0
        mz_a = -cya_a * s2_to_s1 * (float(l_arm) ** 2) * math.sqrt(kv)
    else:
        mz_f = -cya_f * (1.0 / s2_to_s1) * (float(l_arm) ** 2)
        mz_a = (-cya_a / 4.0 * (1.0 - 2.0 * abs(float(center_mass))) ** 2 - (2.0 * math.pi - cya_a) / 16.0) * math.sqrt(kv)
    return float(mz_f + 1.5 * mz_a)


def pitch_damping_population(population: np.ndarray, info_aircraft: np.ndarray) -> np.ndarray:
    pop = np.asarray(population, dtype=float)
    info = np.asarray(info_aircraft, dtype=float)
    out = []
    for i in range(pop.shape[0]):
        case = {name: float(pop[i, j]) for j, name in enumerate(INPUT_COLS_V2)}
        out.append(pitch_damping_omegaz(case, float(info[i, INFO_COLS.index("center_mass")])))
    return np.asarray(out, dtype=float)


def backend_direct_k(
    case: dict[str, float],
    backend: AeroBackend,
    alpha: float,
    delta: float,
    kk: int,
    center_mass: float,
    cx: float,
    cy: float,
) -> float:
    use_direct = os.environ.get("NEW20_USE_DIRECT_K", "0").strip().lower() in {"1", "true", "yes", "on"}
    if use_direct and isinstance(backend, (MlpBackend, SplitMlpBackend)):
        pred = backend.predict_outputs(case, alpha, delta)
        if "K" in pred and np.isfinite(pred["K"]):
            return float(pred["K"])
    return float(cy / cx) if abs(cx) > 1e-12 else 0.0


def cruise_trim(case: dict[str, float], backend: AeroBackend, kk: int) -> dict[str, float]:
    alpha_vals = parse_values(os.environ.get("TRIM_ALPHA_VALUES", ""), "-5,5")
    delta_vals = parse_values(os.environ.get("TRIM_DELTA_VALUES", ""), "-5,5")
    n_delta = len(delta_vals)
    n_alpha = len(alpha_vals)
    cy = np.zeros((n_delta, n_alpha))
    mz_ref = np.zeros((n_delta, n_alpha))
    aero_center = np.zeros(n_delta)
    mz0 = np.zeros(n_delta)
    mz_cg = np.zeros((n_delta, n_alpha))

    with prof.timer("trim_cruise", kk=kk):
        if is_mlp_like_backend(backend) and mlp_batch_enabled():
            batch_alpha = []
            batch_delta = []
            for delta in delta_vals:
                for alpha in alpha_vals:
                    batch_alpha.append(float(alpha))
                    batch_delta.append(float(delta))
            preds = backend.predict_outputs_batch(case, batch_alpha, batch_delta)  # type: ignore[attr-defined]
            pos = 0
            for i in range(n_delta):
                for j in range(n_alpha):
                    cy[i, j] = preds[pos]["cy"]
                    mz_ref[i, j] = preds[pos]["mz_ref"]
                    pos += 1

            for i in range(n_delta):
                coeff = np.polyfit(cy[i], mz_ref[i], deg=1)
                aero_center[i] = -float(coeff[0])
                mz0[i] = float(coeff[1])

            for i in range(n_delta):
                ac_i = aero_center[i]
                if abs(ac_i) < 1e-12:
                    mz_cg[i, :] = mz_ref[i, :]
                else:
                    mz_cg[i, :] = (mz_ref[i, :] - mz0[i]) * (-case["margin"]) / ac_i + mz0[i]

            coeff_mz = polyfit2d(alpha_vals, delta_vals, mz_cg)
            coeff_cy = polyfit2d(alpha_vals, delta_vals, cy)
            hs = np.array([[coeff_mz[1], coeff_mz[2]], [coeff_cy[1], coeff_cy[2]]], dtype=float)
            rhs = np.array([-coeff_mz[0], case["cy_req"] - coeff_cy[0]], dtype=float)
            alpha_bal, delta_bal = np.linalg.solve(hs, rhs)
            aero_center_est = float(np.interp(delta_bal, delta_vals, aero_center))
            dmz_dcy = -aero_center_est
            center_mass = float(aero_center_est + case["margin"])

            final_pred = backend.predict_outputs_batch(case, [float(alpha_bal)], [float(delta_bal)])[0]  # type: ignore[attr-defined]
            cx = float(final_pred["cx"])
            cy_final = float(final_pred["cy"])
            mz_final = float(coeff_mz[0] + coeff_mz[1] * alpha_bal + coeff_mz[2] * delta_bal)
            mx_beta = float(final_pred["mx_beta"])
            my_beta = float(final_pred["my_beta"])
            k_val = _k_from_prediction(final_pred, cx, cy_final)
            return {
                "alpha": float(alpha_bal),
                "delta": float(delta_bal),
                "aero_center": aero_center_est,
                "dmz_dcy": float(dmz_dcy),
                "center_mass": center_mass,
                "cx": cx,
                "cy": cy_final,
                "mz": mz_final,
                "mx_beta": mx_beta,
                "my_beta": my_beta,
                "K": float(k_val),
            }

        for i, delta in enumerate(delta_vals):
            for j, alpha in enumerate(alpha_vals):
                _, cy_ij, mz_ij, _, _ = backend.eval(case, float(alpha), float(delta), kk + 10 * i + j)
                cy[i, j] = cy_ij
                mz_ref[i, j] = mz_ij
            coeff = np.polyfit(cy[i], mz_ref[i], deg=1)
            aero_center[i] = -float(coeff[0])
            mz0[i] = float(coeff[1])

        for i in range(n_delta):
            ac_i = aero_center[i]
            if abs(ac_i) < 1e-12:
                mz_cg[i, :] = mz_ref[i, :]
            else:
                mz_cg[i, :] = (mz_ref[i, :] - mz0[i]) * (-case["margin"]) / ac_i + mz0[i]

        coeff_mz = polyfit2d(alpha_vals, delta_vals, mz_cg)
        coeff_cy = polyfit2d(alpha_vals, delta_vals, cy)
        hs = np.array([[coeff_mz[1], coeff_mz[2]], [coeff_cy[1], coeff_cy[2]]], dtype=float)
        rhs = np.array([-coeff_mz[0], case["cy_req"] - coeff_cy[0]], dtype=float)
        alpha_bal, delta_bal = np.linalg.solve(hs, rhs)
        aero_center_est = float(np.interp(delta_bal, delta_vals, aero_center))
        dmz_dcy = -aero_center_est
        center_mass = float(aero_center_est + case["margin"])

        cx, cy_final, mz_final, _, _ = backend.eval(
            case, float(alpha_bal), float(delta_bal), kk + 500, center_mass=center_mass
        )
        if is_mlp_like_backend(backend):
            mz_final = float(coeff_mz[0] + coeff_mz[1] * alpha_bal + coeff_mz[2] * delta_bal)

        mx_beta, my_beta = stability_beta(case, backend, kk + 700, float(alpha_bal), float(delta_bal), center_mass)
        k_val = backend_direct_k(case, backend, float(alpha_bal), float(delta_bal), kk + 500, center_mass, float(cx), float(cy_final))
    return {
        "alpha": float(alpha_bal),
        "delta": float(delta_bal),
        "aero_center": aero_center_est,
        "dmz_dcy": float(dmz_dcy),
        "center_mass": center_mass,
        "cx": float(cx),
        "cy": float(cy_final),
        "mz": float(mz_final),
        "mx_beta": float(mx_beta),
        "my_beta": float(my_beta),
        "K": float(k_val),
    }


def cy_target_from_cruise(case: dict[str, float], v_phase: float, theta: float) -> float:
    q_cr = ac.inputValuesAtmospheric(case["V"], case["H"])
    q_phase = ac.inputValuesAtmospheric(v_phase, case["H"])
    if abs(q_phase) < 1e-12:
        return case["cy_req"]
    return float(case["cy_req"] * q_cr * np.cos(np.deg2rad(theta)) / q_phase)


def alpha_search_phase(case: dict[str, float], backend: AeroBackend, kk: int, v_phase: float, theta: float) -> dict[str, float]:
    local = dict(case)
    local["V"] = float(v_phase)
    target = cy_target_from_cruise(case, v_phase, theta)
    alpha_vals = np.array([-10.0, 10.0], dtype=float)
    cy_vals = []
    mz_vals = []
    with prof.timer("alpha_search_phase", kk=kk, theta=float(theta)):
        if is_mlp_like_backend(backend) and mlp_batch_enabled():
            preds = backend.predict_outputs_batch(local, alpha_vals, np.zeros_like(alpha_vals))  # type: ignore[attr-defined]
            cy_vals = [float(p["cy"]) for p in preds]
            mz_vals = [float(p["mz_ref"]) for p in preds]
            alpha_bal = float(np.interp(target, cy_vals, alpha_vals))
            slope = (mz_vals[-1] - mz_vals[0]) / (cy_vals[-1] - cy_vals[0]) if abs(cy_vals[-1] - cy_vals[0]) > 1e-12 else 0.0
            center_mass = float(-slope + case["margin"])
            final_pred = backend.predict_outputs_batch(local, [alpha_bal], [0.0])[0]  # type: ignore[attr-defined]
            cx = float(final_pred["cx"])
            cy = float(final_pred["cy"])
            k_val = _k_from_prediction(final_pred, cx, cy)
            return {
                "alpha": alpha_bal,
                "delta": 0.0,
                "center_mass": center_mass,
                "cx": cx,
                "cy": cy,
                "mz": float(final_pred["mz_ref"]),
                "K": float(k_val),
                "cy_target": float(target),
            }

        for i, alpha in enumerate(alpha_vals):
            _, cy_i, mz_i, _, _ = backend.eval(local, float(alpha), 0.0, kk + i)
            cy_vals.append(cy_i)
            mz_vals.append(mz_i)
        alpha_bal = float(np.interp(target, cy_vals, alpha_vals))
        slope = (mz_vals[-1] - mz_vals[0]) / (cy_vals[-1] - cy_vals[0]) if abs(cy_vals[-1] - cy_vals[0]) > 1e-12 else 0.0
        center_mass = float(-slope + case["margin"])
        cx, cy, mz, _, _ = backend.eval(local, alpha_bal, 0.0, kk + 20, center_mass=center_mass)
        k_val = backend_direct_k(local, backend, alpha_bal, 0.0, kk + 20, center_mass, float(cx), float(cy))
    return {
        "alpha": alpha_bal,
        "delta": 0.0,
        "center_mass": center_mass,
        "cx": float(cx),
        "cy": float(cy),
        "mz": float(mz),
        "K": float(k_val),
        "cy_target": float(target),
    }


def stability_beta(case: dict[str, float], backend: AeroBackend, kk: int, alpha: float, delta: float, center_mass: float) -> tuple[float, float]:
    if is_mlp_like_backend(backend):
        _, _, _, mx_beta, my_beta = backend.eval(case, alpha, delta, kk, center_mass=center_mass)
        return mx_beta, my_beta
    _, _, _, mx0, my0 = backend.eval(case, alpha, delta, kk, center_mass=center_mass, beta=0.0)
    _, _, _, mx5, my5 = backend.eval(case, alpha, delta, kk + 1, center_mass=center_mass, beta=5.0)
    return (mx5 - mx0) / 5.0, (my5 - my0) / 5.0


def safe_p2w(v: float, k_val: float, alpha: float, theta: float, eff: float = 0.76) -> float:
    if not np.isfinite(k_val) or k_val <= 0:
        return -1.0 if theta < 0 else 1000.0
    val = sz.pwr_to_weight(v, k_val, alpha, eff, theta)
    if not np.isfinite(val):
        return 1000.0
    if val < 0 and theta >= 0:
        return 1000.0
    return float(val)


def phase_sizing(case: dict[str, float], phase: dict[str, float], m0: float, t: float, theta: float, w_rpm: float, gamma: float, ce: float, type_power: str) -> dict[str, float]:
    with prof.timer("mission_sizing_phase", theta=float(theta)):
        rho = 1.225 * (1 - case["H"] / 44300) ** 4.256
        s_ref = case["S_ref"]
        v = case["V"] if theta == 0 else 0.9 * case["V"]
        x_force = phase["cx"] * rho * v ** 2 * s_ref / 2
        thrust = (x_force + m0 * 9.81 * np.sin(np.deg2rad(theta))) / np.cos(np.deg2rad(phase["alpha"]))
        p2w = safe_p2w(v, phase["K"], phase["alpha"], theta)
        m_pow = sz.m_power(p2w, gamma)
        m_fue = sz.m_batery(0.2, p2w, t, 0.7, 0.7) if type_power == "eltr" else sz.m_fuel_cl(p2w, t, ce)
        thrust_for_sizing = thrust if np.isfinite(thrust) and thrust > 0 else 0.0
        m_v, d_v = sz.mvinta_DBC(thrust_for_sizing, w_rpm, case["H"])
    return {"m_V": float(m_v), "D_V": float(d_v), "m_pow": float(m_pow), "m_fue": float(m_fue), "p2w": float(p2w)}


def mission_phase_profile(case: dict[str, float], mission_l_km: float) -> dict[str, float]:
    with prof.timer("mission_profile"):
        v = float(case["V"])
        h = max(0.0, float(case["H"]))
        theta_cl = np.deg2rad(5.0)
        theta_dc = np.deg2rad(30.0)
        l_climb_km = h / np.tan(theta_cl) / 1000.0
        l_declimb_km = h / np.tan(theta_dc) / 1000.0
        l_cruise_km = float(mission_l_km) - l_climb_km - l_declimb_km
        if v <= 0.0 or l_cruise_km <= 0.0:
            raise ValueError(
                f"Invalid mission profile: V={v}, H={h}, L_cruise_km={l_cruise_km}"
            )
        t_cruise_h = l_cruise_km / (v * 3.6)
        t_climb_h = h / (0.9 * v * np.sin(theta_cl)) / 3600.0
        t_declimb_h = h / (0.9 * v * np.sin(theta_dc)) / 3600.0
    return {
        "L_mission_km": float(mission_l_km),
        "L_cruise_km": float(l_cruise_km),
        "L_climb_km": float(l_climb_km),
        "L_declimb_km": float(l_declimb_km),
        "t_cruise_h": float(t_cruise_h),
        "t_climb_h": float(t_climb_h),
        "t_declimb_h": float(t_declimb_h),
    }


def _candidate_core(
    row: dict[str, float],
    backend: AeroBackend,
    kk_base: int,
    m0: float,
    mission_l_km: float,
    w_rpm: float,
    gamma: float,
    ce1: float,
    ce2: float,
    type_power: str,
    mpay: float,
):
    with prof.timer("candidate_core", kk_base=kk_base):
        f_geo, a_geo, v_geo, fuse_geo, scheme_fuse = build_geom_from_row(row, ac)
        cruise = cruise_trim(row, backend, kk_base)
        climb = alpha_search_phase(row, backend, kk_base + 2000, 0.9 * row["V"], 5.0)
        declimb = alpha_search_phase(row, backend, kk_base + 3000, 0.9 * row["V"], -30.0)
        mission = mission_phase_profile(row, mission_l_km)

        with prof.timer("mission_sizing_total", kk_base=kk_base):
            cr_sz = phase_sizing(row, cruise, m0, mission["t_cruise_h"], 0.0, w_rpm, gamma, ce2, type_power)
            cl_sz = phase_sizing(row, climb, m0, mission["t_climb_h"], 5.0, w_rpm, gamma, ce1, type_power)
            dc_sz = phase_sizing(row, declimb, m0, mission["t_declimb_h"], -30.0, w_rpm, gamma, ce1, type_power)

    m_v = max(cr_sz["m_V"], cl_sz["m_V"], dc_sz["m_V"])
    d_v = cr_sz["D_V"]
    if cl_sz["m_V"] == m_v:
        d_v = cl_sz["D_V"]
    if dc_sz["m_V"] == m_v:
        d_v = dc_sz["D_V"]
    m_pu = max(cr_sz["m_pow"], cl_sz["m_pow"], dc_sz["m_pow"])
    p2w = max(cr_sz["p2w"], cl_sz["p2w"], dc_sz["p2w"])
    m_fue = cr_sz["m_fue"] + max(0.0, cl_sz["m_fue"]) + max(0.0, dc_sz["m_fue"])
    n_val = p2w * m0

    m_const_v = 0.0 if v_geo[6] <= 0 else float(sz.m_constr_surface(v_geo, m0))
    m_ss = float(sz.m_SS(m0))
    m_f = float(sz.m_constr_fueslage(fuse_geo, row["V"], scheme_fuse))
    m_const_f = float(sz.m_constr_surface(f_geo, m0))
    m_const_a = float(sz.m_constr_surface(a_geo, m0))
    m_equip = 0.08

    denom = 1.0 - m_fue - m_const_f - m_const_a - m_const_v - m_equip - m_pu
    mtow_out = 100000.0 if denom <= 0 or not np.isfinite(denom) else (mpay + m_v + m_ss + m_f) / denom
    if mtow_out <= 0 or not np.isfinite(mtow_out):
        mtow_out = 100000.0
    m_fue_kg = float(m_fue * mtow_out)
    q_fuel = float(m_fue_kg / (mpay * mission_l_km))
    q_g_per_ton_km = float(q_fuel * 1_000_000.0)
    freecad = geometry_freecad_outputs(row, f_geo, a_geo, v_geo, fuse_geo, scheme_fuse)
    return {
        "f_geo": f_geo,
        "a_geo": a_geo,
        "v_geo": v_geo,
        "fuse_geo": fuse_geo,
        "scheme_fuse": scheme_fuse,
        "cruise": cruise,
        "mission": mission,
        "m_v": float(m_v),
        "d_v": float(d_v),
        "m_pu": float(m_pu),
        "m_fue": float(m_fue),
        "p2w": float(p2w),
        "n_val": float(n_val),
        "m_const_f": float(m_const_f),
        "m_const_a": float(m_const_a),
        "m_const_v": float(m_const_v),
        "m_f": float(m_f),
        "m_ss": float(m_ss),
        "mtow_out": float(mtow_out),
        "m_fue_kg": float(m_fue_kg),
        "q_fuel": float(q_fuel),
        "q_g_per_ton_km": float(q_g_per_ton_km),
        "freecad": freecad,
    }


def _fixedpoint_enabled() -> bool:
    return os.environ.get("NEW20_FIXEDPOINT", "0").strip().lower() in {"1", "true", "yes", "on"}


def geometry_freecad_outputs(case: dict[str, float], f_geo: np.ndarray, a_geo: np.ndarray, v_geo: np.ndarray, fuse_geo: np.ndarray, scheme_fuse: int) -> list[float]:
    f_span, f_root, f_tip = ac.lift_surface_def(f_geo)
    a_span, a_root, a_tip = ac.lift_surface_def(a_geo)
    v_span, v_root, v_tip = ac.lift_surface_def(v_geo)
    distance_two_fuse = min(float(a_span), float(f_span))

    if case["a_S_rel"] <= 0.5:
        f_x, f_y, f_z, _, _ = ac.lift_wing_sec(f_geo)
        a_x, a_y, a_z, _, _ = ac.lift_h_tail_sec(a_geo, scheme_fuse)
    else:
        f_x, f_y, f_z, _, _ = ac.lift_h_tail_sec(f_geo, scheme_fuse)
        a_x, a_y, a_z, _, _ = ac.lift_wing_sec(a_geo)
    v_x, v_y, v_z, _, _ = ac.lift_v_tail_sec(v_geo)

    v_loc = float(case["a_x_loc"])
    fuse_d = float(fuse_geo[3])
    a_z_render = [float(a_z[0]), float(a_z[1])]
    a_dihedral_render = float(a_geo[4])
    if abs(a_dihedral_render) > 1e-12:
        a_half_span = abs(float(a_y[1]) - float(a_y[0]))
        a_z_render = [0.0, a_half_span * math.tan(math.radians(a_dihedral_render))]
    freecad_n_fuse = 2.0 if int(scheme_fuse) == 1 else 1.0
    return [
        float(f_geo[5]), float(f_geo[3]), float(f_root), float(f_tip),
        float(f_x[0]), float(f_y[0]), float(f_z[0]),
        float(f_x[1]), float(f_y[1]), float(f_z[1]),
        float(a_geo[5]), float(a_geo[3]), float(a_root), float(a_tip),
        float(a_x[0] + case["a_x_loc"]), float(a_y[0]), float(a_z_render[0]),
        float(a_x[1] + case["a_x_loc"]), float(a_y[1]), float(a_z_render[1]),
        float(v_root), float(v_tip),
        float(v_x[0] + v_loc), float(v_y[0]), float(v_z[0]),
        float(v_x[1] + v_loc), float(v_y[1]), float(v_z[1]),
        float(fuse_geo[0]), float(fuse_geo[1]), float(fuse_geo[2]),
        fuse_d, float(fuse_geo[5]), 1.0 if v_geo[6] > 0 else 0.0, freecad_n_fuse, distance_two_fuse,
    ]


def evaluate_candidate(
    row: dict[str, float],
    backend: AeroBackend,
    kk: int,
    mpay: float = 600.0,
    w_rpm: float = 5800.0,
    t: float = 25.0,
    type_power: str = "DBC",
    gamma: float = 0.87,
    ce1: float = 0.285,
    ce2: float = 0.27,
) -> tuple[float, ...]:
    with prof.timer("fixed_point_candidate", kk=int(kk)):
        mission_l_km = float(os.environ.get("NEW20_MISSION_L_KM", "3000"))
        avl_kk_base = int(kk) * 10000
        m0_guess = max(1.0, float(row["m0"]))
        final_row = with_secondary_inputs(row, m0_guess)
        core = _candidate_core(
            final_row, backend, avl_kk_base, m0_guess, mission_l_km,
            w_rpm, gamma, ce1, ce2, type_power, mpay,
        )

    f_geo = core["f_geo"]
    a_geo = core["a_geo"]
    cruise = core["cruise"]
    mission = core["mission"]
    mtow_out = core["mtow_out"]
    m_fue_kg = core["m_fue_kg"]
    q_fuel = core["q_fuel"]
    q_g_per_ton_km = core["q_g_per_ton_km"]
    freecad = core["freecad"]
    info = [
        float(mtow_out), float(core["m_v"] / mtow_out), float(core["d_v"]), float(core["m_pu"]), float(core["m_fue"]), float(core["p2w"]), float(core["n_val"]),
        float(core["m_const_f"]), float(core["m_const_a"]), float(core["m_const_v"]), float(core["m_f"] / mtow_out), float(core["m_ss"] / mtow_out),
        float(cruise["cx"]), float(cruise["cy"]), float(cruise["mz"]), float(cruise["mx_beta"]), float(cruise["my_beta"]),
        float(cruise["K"]), float(cruise["center_mass"]), float(cruise["alpha"]), float(cruise["delta"]), float(ac.vol_coeff(f_geo, a_geo, cruise["center_mass"])[0]),
        float(cruise["dmz_dcy"]),
        m_fue_kg, float(mission_l_km), mission["L_cruise_km"], mission["L_climb_km"],
        mission["L_declimb_km"], mission["t_cruise_h"], mission["t_climb_h"],
        mission["t_declimb_h"], q_fuel, q_g_per_ton_km,
    ]
    return tuple(info + freecad)


def failed_candidate_result(row: dict[str, float]) -> tuple[float, ...]:
    try:
        final_row = with_secondary_inputs(row, max(1.0, float(row.get("m0", 1.0))))
        f_geo, a_geo, v_geo, fuse_geo, scheme_fuse = build_geom_from_row(final_row, ac)
        freecad = geometry_freecad_outputs(final_row, f_geo, a_geo, v_geo, fuse_geo, scheme_fuse)
    except Exception:
        freecad = [0.0] * len(FREECAD_COLS)
    info = [
        100000.0, 1.0, 0.0, 1.0, 1.0, 1000.0, 0.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 999.0, 999.0, 999.0, 999.0,
        0.0, 0.0, 999.0, 999.0, 0.0, 0.0,
        100000.0, float(os.environ.get("NEW20_MISSION_L_KM", "3000")),
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 999.0, 999000000.0,
    ]
    return tuple(info + freecad)


def save_input_dataframe(plane) -> pd.DataFrame:
    return pd.DataFrame(plane, columns=INPUT_COLS_V2)


def save_output_dataframe(info_aircraft) -> pd.DataFrame:
    return pd.DataFrame(info_aircraft, columns=INFO_COLS)


def save_freecad_dataframe(plane) -> pd.DataFrame:
    return pd.DataFrame(plane, columns=FREECAD_COLS)
