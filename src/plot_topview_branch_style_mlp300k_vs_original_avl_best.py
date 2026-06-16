from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
AVL_SRC = PROJECT_ROOT / "avl_optimize_portable" / "src_avl_full"
if str(AVL_SRC) not in sys.path:
    sys.path.insert(0, str(AVL_SRC))

import AeroCoeff_AVL as ac  # type: ignore  # noqa: E402

from branch_geometry_v2 import build_geom_from_row
from new20_sizing_eval import FREECAD_COLS, geometry_freecad_outputs
from problem_v2_spec import INPUT_COLS_V2


BRANCH_ORDER = [
    "normal_1_1",
    "normal_1_2",
    "normal_1_3",
    "normal_2_1",
    "normal_2_2",
    "normal_2_3",
    "normal_3_1",
    "normal_3_2",
    "normal_3_3",
    "duck_1_x",
    "duck_2_x",
    "duck_3_x",
]


def freecad_from_case(row: dict[str, float]) -> dict[str, float]:
    f_geo, a_geo, _, fuse_geo, scheme_fuse = build_geom_from_row(row, ac)
    vals = geometry_freecad_outputs(row, f_geo, a_geo, ac.input_lift_surface_data(0, 15, 2, 0, 90, 0, 0, float(row["a_x_loc"])), fuse_geo, scheme_fuse)
    return {col: float(vals[i]) for i, col in enumerate(FREECAD_COLS)}


def x_origin_forward_wing_leading_edge(g: dict[str, float]) -> float:
    return float(g["f_x_loc_root_chord"])


def normalize_to_avl_origin(g: dict[str, float]) -> dict[str, float]:
    out = dict(g)
    x0 = x_origin_forward_wing_leading_edge(out)
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
        polys.append(np.asarray([(xr, y_root - width), (xt, y_tip - width), (xt + ct, y_tip + width), (xr + cr, y_root + width)], dtype=float))
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


def branch_line(prefix: str, q: float, cy: float, K: float, mtow: float, p0: float, V: float, S: float, feasible: str | None = None) -> str:
    base = f"{prefix}: q={q:.2f}, cy={cy:.3f}, K={K:.2f}, mtow={mtow:.0f}, p0={p0:.2f}, V={V:.2f}, S={S:.1f}"
    if feasible is not None:
        base += f", feasible={feasible}"
    return base


def render_branch(branch: str, orig_case: pd.Series, orig_stats: pd.Series, recheck_row: pd.Series, out_path: Path) -> None:
    orig_row = {name: float(orig_case[name]) for name in INPUT_COLS_V2}
    mlp_row = {name: float(recheck_row[name]) for name in INPUT_COLS_V2}
    avl_g = normalize_to_avl_origin(freecad_from_case(orig_row))
    mlp_g = normalize_to_avl_origin(freecad_from_case(mlp_row))
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
    draw_aircraft(ax, avl_g, "#1f77b4", "AVL goc", avl_dx, dy_aircraft)
    draw_aircraft(ax, mlp_g, "#ff7f0e", "MLP moi", mlp_dx, dy_aircraft)
    ax.scatter([avl_dx + float(orig_stats.get("center_mass", 0.0)) * mac_for_case(orig_row)], [dy_aircraft], s=75, marker="o", color="red", edgecolor="white", linewidth=0.8, zorder=8)
    ax.scatter([mlp_dx + float(recheck_row["mlp_center_mass"]) * mac_for_case(mlp_row)], [dy_aircraft], s=75, marker="o", color="red", edgecolor="white", linewidth=0.8, zorder=8)
    ax.scatter([mlp_dx + float(recheck_row["avl_center_mass"]) * mac_for_case(mlp_row)], [dy_aircraft], s=75, marker="x", color="purple", linewidth=1.8, zorder=8)

    title = branch
    line1 = branch_line(
        "AVL goc",
        float(orig_stats["q_avl_goc"]),
        float(orig_stats["cy_avl_goc"]),
        float(orig_stats["K_avl_goc"]),
        float(orig_stats["mtow_avl_goc"]),
        float(orig_stats["mtow_avl_goc"]) / max(float(orig_stats["S_ref_avl_goc"]), 1e-12),
        float(orig_stats["V_avl_goc"]),
        float(orig_stats["S_ref_avl_goc"]),
    )
    line2 = branch_line(
        "MLP moi",
        float(recheck_row["mlp_q_g_per_ton_km"]),
        float(recheck_row["mlp_cy"]),
        float(recheck_row["mlp_K"]),
        float(recheck_row["mlp_mtow_out"]),
        float(recheck_row["mlp_mtow_out"]) / max(float(recheck_row["S_ref"]), 1e-12),
        float(recheck_row["V"]),
        float(recheck_row["S_ref"]),
    )
    line3 = branch_line(
        "AVL recheck",
        float(recheck_row["avl_q_g_per_ton_km"]),
        float(recheck_row["avl_cy"]),
        float(recheck_row["avl_K"]),
        float(recheck_row["avl_mtow_out"]),
        float(recheck_row["avl_mtow_out"]) / max(float(recheck_row["S_ref"]), 1e-12),
        float(recheck_row["V"]),
        float(recheck_row["S_ref"]),
        feasible=str(bool(recheck_row["avl_feasible"])),
    )

    fig.suptitle(title, fontsize=24, y=0.965)
    fig.text(0.08, 0.86, line1, fontsize=16)
    fig.text(0.08, 0.82, line2, fontsize=16)
    fig.text(0.08, 0.78, line3, fontsize=16)

    pts = np.vstack([transform(avl_pts, avl_dx, dy_aircraft), transform(mlp_pts, mlp_dx, dy_aircraft)])
    pad_x = max(0.08 * np.ptp(pts[:, 0]), 0.6)
    pad_y = max(0.12 * np.ptp(pts[:, 1]), 0.6)
    ax.set_xlim(float(pts[:, 0].min() - pad_x), float(pts[:, 0].max() + pad_x))
    ax.set_ylim(float(pts[:, 1].min() - pad_y), float(pts[:, 1].max() + pad_y + 1.35))
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    src_summary = WORKSPACE_ROOT / "analysis_topview_mixed_normal200_duck230_20260604" / "mixed_topview_summary.csv"
    orig_cases_path = PROJECT_ROOT / "data" / "benchmarks" / "avl_cy06_full12_cases.csv"
    recheck_path = PROJECT_ROOT / "analysis_optimizer_300k_hard40k_avl_recheck.csv"
    out_root = PROJECT_ROOT / "analysis_topview_mlp300k_vs_original_avl_best_cadquery_20260605"
    out_branch = out_root / "top_view_by_branch"
    out_root.mkdir(parents=True, exist_ok=True)
    out_branch.mkdir(parents=True, exist_ok=True)

    orig_df = pd.read_csv(src_summary)
    orig_cases_df = pd.read_csv(orig_cases_path)
    re_df = pd.read_csv(recheck_path)
    orig_map = {str(r["branch"]): r for _, r in orig_df.iterrows()}
    orig_case_map = {str(r["branch"]): r for _, r in orig_cases_df.iterrows()}
    re_map = {str(r["branch"]): r for _, r in re_df.iterrows()}

    summary_rows: list[dict[str, float | str | bool]] = []
    image_paths: list[Path] = []

    for branch in BRANCH_ORDER:
        if branch not in orig_map or branch not in re_map or branch not in orig_case_map:
            continue
        out_path = out_branch / f"{branch}_topview_avl_vs_mlp300k.png"
        render_branch(branch, orig_case_map[branch], orig_map[branch], re_map[branch], out_path)
        image_paths.append(out_path)

        r = re_map[branch]
        o = orig_map[branch]
        summary_rows.append(
            {
                "branch": branch,
                "q_avl_goc": float(o["q_avl_goc"]),
                "K_avl_goc": float(o["K_avl_goc"]),
                "cy_avl_goc": float(o["cy_avl_goc"]),
                "mtow_avl_goc": float(o["mtow_avl_goc"]),
                "p0_avl_goc": float(o["mtow_avl_goc"]) / max(float(o["S_ref_avl_goc"]), 1e-12),
                "V_avl_goc": float(o["V_avl_goc"]),
                "S_ref_avl_goc": float(o["S_ref_avl_goc"]),
                "q_mlp": float(r["mlp_q_g_per_ton_km"]),
                "K_mlp": float(r["mlp_K"]),
                "cy_mlp": float(r["mlp_cy"]),
                "mtow_mlp": float(r["mlp_mtow_out"]),
                "p0_mlp": float(r["mlp_mtow_out"]) / max(float(r["S_ref"]), 1e-12),
                "V_mlp": float(r["V"]),
                "S_ref_mlp": float(r["S_ref"]),
                "q_recheck": float(r["avl_q_g_per_ton_km"]),
                "K_recheck": float(r["avl_K"]),
                "cy_recheck": float(r["avl_cy"]),
                "mtow_recheck": float(r["avl_mtow_out"]),
                "p0_recheck": float(r["avl_mtow_out"]) / max(float(r["S_ref"]), 1e-12),
                "feasible_recheck": bool(r["avl_feasible"]),
                "mlp_source": "300k_split",
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_root / "topview_summary.csv", index=False)

    fig, axes = plt.subplots(6, 2, figsize=(16, 34))
    axes_arr = axes.flatten()
    fig.suptitle(
        "Top views: original AVL best vs 300k split MLP\nLeft = AVL goc, right = new MLP branch optimum",
        fontsize=18,
        y=0.995,
        x=0.01,
        ha="left",
    )
    for ax, img_path, branch in zip(axes_arr, image_paths, BRANCH_ORDER):
        ax.imshow(plt.imread(img_path))
        ax.set_title(branch, fontsize=13)
        ax.axis("off")
    for ax in axes_arr[len(image_paths) :]:
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_root / "topview_grid.png", dpi=170)
    plt.close(fig)

    print(f"Wrote {out_root}")


if __name__ == "__main__":
    main()
