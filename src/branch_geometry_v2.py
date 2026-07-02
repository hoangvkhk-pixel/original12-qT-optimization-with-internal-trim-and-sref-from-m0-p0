from __future__ import annotations

import math

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

    f_span, f_root_chord, _ = ac_module.lift_surface_def(f_geo)
    a_span, a_root_chord, a_tip_chord = ac_module.lift_surface_def(a_geo)
    _, v_root_chord, _ = ac_module.lift_surface_def(v_geo)
    distance_two_fuse = min(float(a_span), float(f_span))
    a_loc = float(row["a_x_loc"])
    v_loc = float(row["a_x_loc"])

    if scheme_fuse == 1:
        fuse_diameter = 0.76 / 2.5
    elif scheme_fuse == 2:
        fuse_diameter = 0.76 / 1.5
    else:
        fuse_diameter = 0.76
    nose_f_aspect = 2.0

    if scheme_fuse == 2:
        l_center = max(a_loc, v_loc)
        center_f_aspect = l_center / fuse_diameter if fuse_diameter else 0.0
        tg1 = a_loc + a_root_chord
        tg2 = v_loc + v_root_chord
        l_tail = max(tg1, tg2) - l_center
        tail_f_aspect = l_tail / fuse_diameter if fuse_diameter else 0.0
    elif scheme_fuse == 3:
        l_center = max(a_root_chord, f_root_chord)
        center_f_aspect = l_center / fuse_diameter if fuse_diameter else 0.0
        tail_f_aspect = 2.0
    else:
        l_center = max(a_loc, v_loc)
        center_f_aspect = l_center / fuse_diameter if fuse_diameter else 0.0
        chord_i = (a_tip_chord - a_root_chord) * distance_two_fuse / a_span + a_root_chord if abs(a_span) > 1e-12 else a_root_chord
        tg1 = a_loc + a_root_chord
        tg2 = v_loc + v_root_chord
        tg3 = a_loc + chord_i + distance_two_fuse * math.tan(math.radians(float(a_geo[1]))) / 2.0
        l_tail = max(tg1, tg2, tg3) - l_center
        tail_f_aspect = l_tail / fuse_diameter if fuse_diameter else 0.0

    l_nose = nose_f_aspect * fuse_diameter
    fuse_x_loc = a_loc - l_nose if (a_s_rel > 0.5 and scheme_fuse == 3) else -l_nose
    fuse_geo = ac_module.input_body_data(
        nose_f_aspect,
        center_f_aspect,
        tail_f_aspect,
        fuse_diameter,
        fuse_diameter,
        fuse_x_loc,
    )
    return np.asarray(f_geo), np.asarray(a_geo), np.asarray(v_geo), np.asarray(fuse_geo), scheme_fuse
