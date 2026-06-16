from __future__ import annotations

import numpy as np
import pandas as pd


def decode_scheme(value: float) -> int:
    if value <= 1.0 / 3.0:
        return 1
    if value >= 2.0 / 3.0:
        return 3
    return 2


def normalized_scheme_value(value: float | int) -> float:
    if value in (1, 2, 3):
        return {1: 0.0, 2: 0.5, 3: 1.0}[int(value)]
    return float(value)


def build_geom_from_row(row: dict[str, float] | pd.Series, ac_module):
    scheme_fuse = decode_scheme(float(row["scheme_fuse"]))
    scheme_vertical = decode_scheme(float(row["scheme_vertical"]))
    a_dihedral_mag = float(row["a_dihedral_mag"])
    a_s_rel = float(row["a_S_rel"])
    f_area = float(row["S_ref"]) * (1.0 - a_s_rel)
    a_area = float(row["S_ref"]) * a_s_rel

    if a_s_rel <= 0.5:
        aft_dihedral = {1: a_dihedral_mag, 2: 0.0, 3: -a_dihedral_mag}[scheme_vertical]
        a_aspect = float(row["a_aspect"])
        if abs(aft_dihedral) > 1e-12 and a_aspect > 7.0:
            a_aspect = 7.0
        f_geo = ac_module.input_lift_surface_data(
            float(row["f_aspect"]),
            float(row["f_sweep"]),
            float(row["f_taper"]),
            float(row["f_twist"]),
            0.0,
            0.0,
            f_area,
        )
        a_geo = ac_module.input_lift_surface_data(
            a_aspect,
            0.0,
            float(row["a_taper"]),
            float(row["a_twist"]),
            aft_dihedral,
            0.0,
            a_area,
            float(row["a_x_loc"]),
        )
        if abs(aft_dihedral) > 1e-12:
            v_aspect = 0.0
            v_area = 0.0
        else:
            v_aspect = float(row["v_aspect"])
            v_area = float(row["S_ref"]) * float(row["v_S_rel"])
    else:
        f_geo = ac_module.input_lift_surface_data(
            float(row["f_aspect"]),
            0.0,
            float(row["f_taper"]),
            float(row["f_twist"]),
            0.0,
            0.0,
            f_area,
        )
        a_geo = ac_module.input_lift_surface_data(
            float(row["a_aspect"]),
            float(row["a_sweep"]),
            float(row["a_taper"]),
            float(row["a_twist"]),
            a_dihedral_mag,
            0.0,
            a_area,
            float(row["a_x_loc"]),
        )
        v_aspect = float(row["v_aspect"])
        v_area = float(row["S_ref"]) * float(row["v_S_rel"])

    v_geo = ac_module.input_lift_surface_data(
        v_aspect,
        15.0,
        2.0,
        0.0,
        90.0,
        0.0,
        v_area,
        float(row["a_x_loc"]),
    )
    fuse_geo = ac_module.input_body_data(2.0, 4.0, 2.0, 0.76, 0.76, -2.0 * 0.76)
    return np.asarray(f_geo), np.asarray(a_geo), np.asarray(v_geo), np.asarray(fuse_geo), scheme_fuse
