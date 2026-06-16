from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from tensorflow import keras

from problem_v2_spec import AERO_INPUT_COLS


PROJECT_ROOT = Path(__file__).resolve().parent.parent
AVL_SRC = PROJECT_ROOT / "avl_optimize_portable" / "src_avl_full"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(AVL_SRC) not in sys.path:
    sys.path.insert(0, str(AVL_SRC))

from avl_optimize_portable.src.smoke_new17_case import solve_cruise as solve_cruise_avl  # noqa: E402


def polyfit2d(alpha_vals: np.ndarray, delta_vals: np.ndarray, values: np.ndarray) -> np.ndarray:
    alpha_grid, delta_grid = np.meshgrid(alpha_vals, delta_vals)
    a = np.array([np.ones(alpha_grid.size), alpha_grid.flatten(), delta_grid.flatten()]).T
    coeff, _, _, _ = np.linalg.lstsq(a, values.flatten(), rcond=None)
    return coeff


def parse_values(text: str) -> np.ndarray:
    return np.array([float(x.strip()) for x in text.split(",") if x.strip()], dtype=float)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=str, default="models/aero_mlp_v2")
    p.add_argument("--out", type=str, default="mlp_optimize_portable/output_real/mlp_case_result.json")
    return p.parse_args()


def feature(case: dict[str, float], alpha: float, delta: float) -> np.ndarray:
    vals = [case[k] for k in AERO_INPUT_COLS[:-2]]
    vals.extend([alpha, delta])
    return np.asarray(vals, dtype=np.float32)


def build_trim_plane(case, predict, alpha_vals, delta_vals):
    cy = np.zeros((2, 2))
    mz_ref = np.zeros((2, 2))
    aero_center = np.zeros(2)
    mz0 = np.zeros(2)
    mz_cg = np.zeros((2, 2))

    for i, delta in enumerate(delta_vals):
        for j, alpha in enumerate(alpha_vals):
            pred = predict(case, float(alpha), float(delta))
            cy[i, j] = float(pred[1])
            mz_ref[i, j] = float(pred[2])
        coeff_cy_to_mz = np.polyfit(cy[i], mz_ref[i], deg=1)
        aero_center[i] = -float(coeff_cy_to_mz[0])
        mz0[i] = float(coeff_cy_to_mz[1])

    for i in range(2):
        ac_i = float(aero_center[i])
        if abs(ac_i) < 1e-12:
            mz_cg[i, :] = mz_ref[i, :]
        else:
            mz_cg[i, :] = (mz_ref[i, :] - mz0[i]) * (-case["margin"]) / ac_i + mz0[i]

    coeff_mz = polyfit2d(alpha_vals, delta_vals, mz_cg)
    coeff_cy = polyfit2d(alpha_vals, delta_vals, cy)
    hs = np.array([[coeff_mz[1], coeff_mz[2]], [coeff_cy[1], coeff_cy[2]]], dtype=float)
    rhs = np.array([-coeff_mz[0], case["cy_req"] - coeff_cy[0]], dtype=float)
    alpha_bal, delta_bal = np.linalg.solve(hs, rhs)
    pred_final = predict(case, float(alpha_bal), float(delta_bal))
    aero_center_est = float(np.interp(delta_bal, delta_vals, aero_center))
    return {
        "alpha_bal_deg": float(alpha_bal),
        "delta_bal_deg": float(delta_bal),
        "aero_center_est": aero_center_est,
        "center_mass": float(aero_center_est + case["margin"]),
        "cx": float(pred_final[0]),
        "cy": float(pred_final[1]),
        "mz_ref": float(pred_final[2]),
        "mx_beta": float(pred_final[3]),
        "my_beta": float(pred_final[4]),
        "K": float(pred_final[5]) if len(pred_final) > 5 else None,
        "cy_error": float(pred_final[1] - case["cy_req"]),
    }


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir
    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    model = keras.models.load_model(model_dir / "aero_mlp_v2_best.keras", compile=False)
    sx = joblib.load(model_dir / "scaler_X.joblib")
    sy = joblib.load(model_dir / "scaler_Y.joblib")

    def predict(case: dict[str, float], alpha: float, delta: float) -> np.ndarray:
        x = feature(case, alpha, delta).reshape(1, -1)
        y_s = model.predict(sx.transform(x), verbose=0)
        return sy.inverse_transform(y_s)[0]

    case = {
        "f_aspect": 8.5,
        "f_sweep": 12.0,
        "f_taper": 2.0,
        "f_twist": -1.0,
        "a_aspect": 7.5,
        "a_sweep": -6.0,
        "a_taper": 1.8,
        "a_twist": 1.0,
        "a_x_loc": 3.8,
        "a_S_rel": 0.34,
        "v_aspect": 2.8,
        "v_S_rel": 0.14,
        "scheme_fuse": 0.5,
        "scheme_vertical": 0.5,
        "a_dihedral_mag": 6.0,
        "S_ref": 12.0,
        "V": 55.0,
        "H": 0.0,
        "margin": -0.1,
        "cy_req": 0.58,
    }

    trim_alpha_vals = parse_values(os.environ.get("TRIM_ALPHA_VALUES", "-5,5"))
    trim_delta_vals = parse_values(os.environ.get("TRIM_DELTA_VALUES", "-5,5"))
    best = build_trim_plane(case, predict, trim_alpha_vals, trim_delta_vals)

    result = {
        "case": case,
        "mlp_result": best,
        "avl_reference": solve_cruise_avl(case, 999000),
    }
    text = json.dumps(result, indent=2)
    out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
