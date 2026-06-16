from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from new20_sizing_eval import (
    AvlBackend,
    MlpBackend,
    SplitMlpBackend,
    candidate_row,
    evaluate_candidate,
    failed_candidate_result,
    save_freecad_dataframe,
    save_input_dataframe,
    save_output_dataframe,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _backend(model_dir: str | None = None):
    backend = os.environ.get("NEW20_BACKEND", "avl").strip().lower()
    if backend == "mlp":
        normal_model_dir = os.environ.get("NEW20_MODEL_DIR_NORMAL", "").strip()
        duck_model_dir = os.environ.get("NEW20_MODEL_DIR_DUCK", "").strip()
        if normal_model_dir and duck_model_dir:
            return SplitMlpBackend(normal_model_dir, duck_model_dir)
        model_root = model_dir or os.environ.get("NEW20_MODEL_DIR", "models/aero_mlp_v2")
        return MlpBackend(model_root)
    if backend == "avl":
        return AvlBackend()
    raise ValueError(f"Unsupported NEW20_BACKEND={backend!r}")


def m0_calc(
    des_par,
    mpay,
    w_rpm,
    t,
    margin,
    type_power,
    gamma,
    Ce1,
    Ce2,
    kk,
    H=0,
    model_dir: str | None = None,
):
    row = candidate_row(des_par, kk)
    # The new problem carries margin and H in the design vector.
    row["margin"] = float(row.get("margin", margin))
    row["H"] = float(row.get("H", H))
    try:
        return evaluate_candidate(
            row,
            _backend(model_dir),
            kk,
            mpay=mpay,
            w_rpm=w_rpm,
            t=t,
            type_power=type_power,
            gamma=gamma,
            ce1=Ce1,
            ce2=Ce2,
        )
    except Exception as exc:
        print(f"[WARN] candidate kk={kk} failed and was penalized: {exc!r}", flush=True)
        return failed_candidate_result(row)


def save_input_DataFrame(plane) -> pd.DataFrame:
    return save_input_dataframe(plane)


def save_output_DataFrame(info_aircraft) -> pd.DataFrame:
    return save_output_dataframe(info_aircraft)


def save_output_DataFrame_FreeCAD(plane) -> pd.DataFrame:
    return save_freecad_dataframe(plane)
