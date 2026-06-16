from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AVL_PROJECT_ROOT = PROJECT_ROOT.parent / "OptimizationNewMLP_original12_qmission_fixedpoint_avl10_experimental_portable"
AVL_SRC = PROJECT_ROOT / "avl_optimize_portable" / "src_avl_full"
if str(AVL_SRC) not in sys.path:
    sys.path.insert(0, str(AVL_SRC))

import AeroCoeff_AVL as ac  # type: ignore  # noqa: E402

from branch_geometry_v2 import build_geom_from_row
from new20_sizing_eval import FREECAD_COLS, geometry_freecad_outputs
from problem_v2_spec import INPUT_COLS_V2


BRANCHES = ["normal_1_1", "normal_1_2", "normal_1_3"]


def freecad_from_case(row: dict[str, float]) -> dict[str, float]:
    f_geo, a_geo, _, fuse_geo, scheme_fuse = build_geom_from_row(row, ac)
    vals = geometry_freecad_outputs(
        row,
        f_geo,
        a_geo,
        ac.input_lift_surface_data(0, 15, 2, 0, 90, 0, 0, float(row["a_x_loc"])),
        fuse_geo,
        scheme_fuse,
    )
    return {col: float(vals[i]) for i, col in enumerate(FREECAD_COLS)}


def normalize_to_origin(g: dict[str, float]) -> dict[str, float]:
    out = dict(g)
    x0 = float(out["f_x_loc_root_chord"])
    for key in [
        "f_x_loc_root_chord",
        "f_x_loc_tip_chord",
        "a_x_loc_root_chord",
        "a_x_loc_tip_chord",
        "v_x_loc_root_chord",
        "v_x_loc_tip_chord",
        "fuse_x_loc",
    ]:
        out[key] = float(out[key]) - x0
    return out


def surface_polygon(g: dict[str, float], prefix: str) -> np.ndarray:
    xr = float(g[f"{prefix}_x_loc_root_chord"])
    yr = float(g[f"{prefix}_y_loc_root_chord"])
    xt = float(g[f"{prefix}_x_loc_tip_chord"])
    yt = float(g[f"{prefix}_y_loc_tip_chord"])
    cr = float(g[f"{prefix}_root_chord"])
    ct = float(g[f"{prefix}_tip_chord"])
    right = [(xr, yr), (xt, yt), (xt + ct, yt), (xr + cr, yr)]
    left = [(xr + cr, -yr), (xt + ct, -yt), (xt, -yt), (xr, -yr)]
    return np.asarray(right + left, dtype=float)


def fuselage_polygons(g: dict[str, float]) -> list[np.ndarray]:
    d = float(g["fuse_diameter"])
    nose = float(g["nose_f_aspect"]) * d
    center = float(g["center_f_aspect"]) * d
    tail = float(g["tail_f_aspect"]) * d
    x0 = float(g["fuse_x_loc"])
    x1 = x0 + nose
    x2_nominal = x1 + center
    x3_nominal = x2_nominal + tail
    surface_rear = max(float(surface_polygon(g, "f")[:, 0].max()), float(surface_polygon(g, "a")[:, 0].max()))
    x3 = max(x3_nominal, surface_rear + 0.15 * d)
    x2 = x3 - tail
    half = 0.5 * d
    n_fuse = int(round(float(g["n_fuse"])))
    distance = float(g["distance_two_fuse"])
    centers = [0.0] if n_fuse != 2 else [-0.5 * distance, 0.5 * distance]
    polys = []
    for yc in centers:
        polys.append(
            np.asarray(
                [(x0, yc), (x1, yc + half), (x2, yc + half), (x3, yc), (x2, yc - half), (x1, yc - half)],
                dtype=float,
            )
        )
    return polys


def vertical_tail_polygons(g: dict[str, float]) -> list[np.ndarray]:
    if float(g["n_vertical"]) <= 0 or float(g["v_root_chord"]) <= 0 or float(g["v_tip_chord"]) <= 0:
        return []
    xr = float(g["v_x_loc_root_chord"])
    yr = float(g["v_y_loc_root_chord"])
    xt = float(g["v_x_loc_tip_chord"])
    yt = float(g["v_y_loc_tip_chord"])
    cr = float(g["v_root_chord"])
    ct = float(g["v_tip_chord"])
    width = max(0.08 * cr, 0.04)
    n_vertical = max(1, int(round(float(g["n_vertical"]))))
    centers = [-0.5 * float(g["distance_two_fuse"]), 0.5 * float(g["distance_two_fuse"])] if n_vertical == 2 else [yr]
    polys = []
    for yc in centers:
        y_root = yc
        y_tip = yc + yt
        polys.append(
            np.asarray([(xr, y_root - width), (xt, y_tip - width), (xt + ct, y_tip + width), (xr + cr, y_root + width)], dtype=float)
        )
    return polys


def all_points(g: dict[str, float]) -> np.ndarray:
    parts = [surface_polygon(g, "f"), surface_polygon(g, "a")]
    parts.extend(fuselage_polygons(g))
    parts.extend(vertical_tail_polygons(g))
    return np.vstack(parts)


def transform(poly: np.ndarray, dx: float, dy: float = 0.0) -> np.ndarray:
    out = poly.copy()
    out[:, 0] += dx
    out[:, 1] += dy
    return out


def mac_for_case(case: dict[str, float]) -> float:
    f_geo, a_geo, _, _, _ = build_geom_from_row(case, ac)
    ref_geo = f_geo if f_geo[6] >= a_geo[6] else a_geo
    return float(ac.ref_dim_lift_surface(ref_geo))


def draw_aircraft(ax: plt.Axes, g: dict[str, float], color: str, label: str, dx: float, dy: float = 0.0) -> None:
    for prefix in ("f", "a"):
        poly = transform(surface_polygon(g, prefix), dx, dy)
        ax.fill(poly[:, 0], poly[:, 1], facecolor=color, edgecolor=color, alpha=0.16, linewidth=1.0)
        ax.plot(*poly.T, color=color, linewidth=1.4)
    for poly0 in fuselage_polygons(g):
        poly = transform(poly0, dx, dy)
        ax.fill(poly[:, 0], poly[:, 1], facecolor=color, edgecolor=color, alpha=0.12, linewidth=1.0)
        ax.plot(*poly.T, color=color, linewidth=1.2)
    for poly0 in vertical_tail_polygons(g):
        poly = transform(poly0, dx, dy)
        ax.fill(poly[:, 0], poly[:, 1], facecolor=color, edgecolor=color, alpha=0.10, linewidth=1.0)
        ax.plot(*poly.T, color=color, linewidth=1.0)
    ax.scatter([dx], [dy], s=22, marker="+", color="black", linewidth=1.1, zorder=9)
    ax.text(dx + 0.08, dy + 0.08, "O", color="black", fontsize=8, ha="left", va="bottom")
    pts = transform(all_points(g), dx, dy)
    x_mid = 0.5 * float(pts[:, 0].min() + pts[:, 0].max())
    y_bottom = float(pts[:, 1].min())
    ax.text(x_mid, y_bottom - 0.32, label, color=color, fontsize=18, ha="center", va="top", fontweight="bold")


def stat_line(prefix: str, q: float, cy: float, K: float, mtow: float, V: float, S: float, dihedral: float) -> str:
    p0 = mtow / max(S, 1e-12)
    return f"{prefix}: q={q:.2f}, cy={cy:.3f}, K={K:.2f}, mtow={mtow:.0f}, p0={p0:.2f}, V={V:.2f}, S={S:.1f}, dihedral={dihedral:.2f}"


def load_mlp_cases() -> dict[str, pd.Series]:
    df = pd.read_csv(PROJECT_ROOT / "analysis_best_cases_qmission_summary.csv")
    return {str(r["branch"]): r for _, r in df.iterrows() if str(r["branch"]) in BRANCHES}


def load_avl_cases() -> dict[str, pd.Series]:
    rows = {}
    root = AVL_PROJECT_ROOT / "gen_avl_fixedpoint_qmission3000_H500_cy06"
    for branch in BRANCHES:
        branch_dir = root / branch
        gens = sorted(int(p.stem.rsplit("_", 1)[-1]) for p in branch_dir.glob("info_aircraft_*.xlsx"))
        gen = gens[-1]
        info = pd.read_excel(branch_dir / f"info_aircraft_{gen}.xlsx")
        px = pd.read_excel(branch_dir / f"Px_{gen}.xlsx")
        for col in info.columns:
            info[col] = pd.to_numeric(info[col], errors="coerce")
        for col in px.columns:
            px[col] = pd.to_numeric(px[col], errors="coerce")
        idx = info["q_g_per_ton_km"].astype(float).idxmin()
        rec = px.loc[idx].copy()
        for key in ["q_g_per_ton_km", "mtow_out", "cx", "cy", "K", "center_mass", "alpha_bal", "delta_bal"]:
            rec[key] = float(info.loc[idx, key])
        rec["generation"] = gen
        rec["row_index_0based"] = int(idx)
        rows[branch] = rec
    return rows


def render_branch(branch: str, avl_row: pd.Series, mlp_row: pd.Series, out_path: Path) -> None:
    avl_case = {name: float(avl_row[name]) for name in INPUT_COLS_V2}
    mlp_case = {name: float(mlp_row[name]) for name in INPUT_COLS_V2}
    avl_g = normalize_to_origin(freecad_from_case(avl_case))
    mlp_g = normalize_to_origin(freecad_from_case(mlp_case))
    avl_pts = all_points(avl_g)
    mlp_pts = all_points(mlp_g)
    avl_width = float(avl_pts[:, 0].max() - avl_pts[:, 0].min())
    mlp_width = float(mlp_pts[:, 0].max() - mlp_pts[:, 0].min())
    span = max(float(np.ptp(avl_pts[:, 1])), float(np.ptp(mlp_pts[:, 1])), 1.0)
    gap = max(0.35 * (avl_width + mlp_width), 1.0 * span)
    avl_dx = -0.5 * gap - 0.5 * avl_width - float(avl_pts[:, 0].min())
    mlp_dx = 0.5 * gap + 0.5 * mlp_width - float(mlp_pts[:, 0].max())

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    dy_aircraft = -0.55
    draw_aircraft(ax, avl_g, "#1f77b4", "AVL toi uu", avl_dx, dy_aircraft)
    draw_aircraft(ax, mlp_g, "#ff7f0e", "MLP best", mlp_dx, dy_aircraft)

    ax.scatter([avl_dx + float(avl_row["center_mass"]) * mac_for_case(avl_case)], [dy_aircraft], s=75, marker="o", color="red", edgecolor="white", linewidth=0.8, zorder=8)
    ax.scatter([mlp_dx + float(mlp_row["center_mass"]) * mac_for_case(mlp_case)], [dy_aircraft], s=75, marker="o", color="red", edgecolor="white", linewidth=0.8, zorder=8)

    fig.suptitle(branch, fontsize=24, y=0.965)
    fig.text(0.08, 0.86, stat_line("AVL toi uu", float(avl_row["q_g_per_ton_km"]), float(avl_row["cy"]), float(avl_row["K"]), float(avl_row["mtow_out"]), float(avl_row["V"]), float(avl_row["S_ref"]), float(avl_row["a_dihedral_mag"])), fontsize=16)
    fig.text(0.08, 0.82, stat_line("MLP best", float(mlp_row["q_g_per_ton_km"]), float(mlp_row["cy"]), float(mlp_row["K"]), float(mlp_row["mtow_out"]), float(mlp_row["V"]), float(mlp_row["S_ref"]), float(mlp_row["a_dihedral_mag"])), fontsize=16)

    pts = np.vstack([transform(avl_pts, avl_dx, dy_aircraft), transform(mlp_pts, mlp_dx, dy_aircraft)])
    pad_x = max(0.08 * np.ptp(pts[:, 0]), 0.6)
    pad_y = max(0.12 * np.ptp(pts[:, 1]), 0.6)
    ax.set_xlim(float(pts[:, 0].min() - pad_x), float(pts[:, 0].max() + pad_x))
    ax.set_ylim(float(pts[:, 1].min() - pad_y), float(pts[:, 1].max() + pad_y + 1.15))
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    out_root = PROJECT_ROOT / "analysis_topview_avlopt_vs_mlpbest_normal1x_20260608"
    out_branch = out_root / "top_view_by_branch"
    out_root.mkdir(parents=True, exist_ok=True)
    out_branch.mkdir(parents=True, exist_ok=True)
    mlp_map = load_mlp_cases()
    avl_map = load_avl_cases()
    summary_rows = []
    images = []
    for branch in BRANCHES:
        out_path = out_branch / f"{branch}_topview_avlopt_vs_mlpbest.png"
        render_branch(branch, avl_map[branch], mlp_map[branch], out_path)
        images.append(out_path)
        summary_rows.append(
            {
                "branch": branch,
                "q_avl_opt": float(avl_map[branch]["q_g_per_ton_km"]),
                "q_mlp_best": float(mlp_map[branch]["q_g_per_ton_km"]),
                "mtow_avl_opt": float(avl_map[branch]["mtow_out"]),
                "mtow_mlp_best": float(mlp_map[branch]["mtow_out"]),
                "S_ref_avl_opt": float(avl_map[branch]["S_ref"]),
                "S_ref_mlp_best": float(mlp_map[branch]["S_ref"]),
                "p0_avl_opt": float(avl_map[branch]["mtow_out"]) / max(float(avl_map[branch]["S_ref"]), 1e-12),
                "p0_mlp_best": float(mlp_map[branch]["mtow_out"]) / max(float(mlp_map[branch]["S_ref"]), 1e-12),
                "dihedral_avl_opt": float(avl_map[branch]["a_dihedral_mag"]),
                "dihedral_mlp_best": float(mlp_map[branch]["a_dihedral_mag"]),
            }
        )
    pd.DataFrame(summary_rows).to_csv(out_root / "topview_summary.csv", index=False)

    fig, axes = plt.subplots(3, 1, figsize=(16, 23))
    fig.suptitle("Top views: AVL toi uu vs MLP best | normal_1_x", fontsize=18, y=0.99, x=0.01, ha="left")
    for ax, img_path, branch in zip(axes, images, BRANCHES):
        ax.imshow(plt.imread(img_path))
        ax.set_title(branch, fontsize=14)
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_root / "topview_grid.png", dpi=170)
    plt.close(fig)
    print(f"Wrote {out_root}")


if __name__ == "__main__":
    main()
