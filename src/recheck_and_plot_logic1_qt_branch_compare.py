from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AUTO_FULL_ROOT = PROJECT_ROOT / "avl_optimize_portable" / "runs_logic1_qt_avl_recheck_branch_compare"
os.environ.setdefault("AUTO_FULL_ROOT", str(DEFAULT_AUTO_FULL_ROOT))
AVL_SRC = PROJECT_ROOT / "avl_optimize_portable" / "src_avl_full"
if str(AVL_SRC) not in sys.path:
    sys.path.insert(0, str(AVL_SRC))

import AeroCoeff_AVL as ac  # type: ignore  # noqa: E402

from branch_geometry_v2 import build_geom_from_row
from new20_sizing_eval import FREECAD_COLS, INFO_COLS, INPUT_COLS_V2, AvlBackend, evaluate_candidate, geometry_freecad_outputs


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mlp-dir", default="gen_mlp_logic1_fixedpoint_qt")
    p.add_argument("--avl-dir", default="gen_avl_logic1_fixedpoint_qt")
    p.add_argument("--outdir", default="analysis_logic1_qt_avl_recheck_branch_compare")
    p.add_argument("--runs-dir", default="avl_optimize_portable/runs_logic1_qt_avl_recheck_branch_compare")
    p.add_argument("--kk-base", type=int, default=930)
    p.add_argument("--max-cy", type=float, default=0.6)
    p.add_argument("--bias-mz", type=float, default=0.001)
    return p.parse_args()


def resolve_dir(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def numeric_frame(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, header=None)
    df.columns = list(df.iloc[0])
    df = df.iloc[1:].reset_index(drop=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def vector_frame(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def last_generation(branch_dir: Path) -> int:
    files = sorted(branch_dir.glob("info_aircraft_*.xlsx"), key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
    if not files:
        raise FileNotFoundError(f"No info_aircraft_*.xlsx in {branch_dir}")
    return int(files[-1].stem.rsplit("_", 1)[-1])


def load_best_qt(branch_dir: Path) -> tuple[int, int, dict[str, float], dict[str, float]]:
    gen = last_generation(branch_dir)
    info_df = numeric_frame(branch_dir / f"info_aircraft_{gen}.xlsx")
    vec_df = vector_frame(branch_dir / f"Px_{gen}.xlsx")
    best_idx = int(info_df["q_g_per_ton_km"].astype(float).idxmin())
    info = {name: float(info_df.loc[best_idx, name]) for name in info_df.columns if pd.notna(info_df.loc[best_idx, name])}
    row = {name: float(vec_df.loc[best_idx, name]) for name in INPUT_COLS_V2}
    return gen, best_idx, row, info


def secondary_s_ref(row: dict[str, float]) -> float:
    p0 = float(row["p0"])
    return max(float(row["m0"]) / max(p0, 1e-12), 1e-12)


def augmented_case(row: dict[str, float]) -> dict[str, float]:
    out = dict(row)
    out["S_ref"] = secondary_s_ref(row)
    return out


def recheck_feasible(info: dict[str, float], max_cy: float, bias_mz: float) -> bool:
    return (
        abs(float(info["mz"])) <= bias_mz
        and float(info["mx_beta"]) <= 0.0
        and float(info["my_beta"]) <= 0.0
        and float(info["cy"]) <= max_cy
        and -5.0 <= float(info["delta_bal"]) <= 5.0
        and -10.0 <= float(info["alpha_bal"]) <= 10.0
        and 0.5 <= float(info["A"]) <= 1.2
    )


def freecad_from_case(row: dict[str, float]) -> dict[str, float]:
    row2 = augmented_case(row)
    f_geo, a_geo, _, fuse_geo, scheme_fuse = build_geom_from_row(row2, ac)
    vals = geometry_freecad_outputs(
        row2,
        f_geo,
        a_geo,
        ac.input_lift_surface_data(0, 15, 2, 0, 90, 0, 0, float(row2["a_x_loc"])),
        fuse_geo,
        scheme_fuse,
    )
    return {col: float(vals[i]) for i, col in enumerate(FREECAD_COLS)}


def normalize_to_forward_origin(g: dict[str, float]) -> dict[str, float]:
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
    polys: list[np.ndarray] = []
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
    polys: list[np.ndarray] = []
    for yc in centers:
        y_root = yc
        y_tip = yc + yt
        polys.append(
            np.asarray(
                [(xr, y_root - width), (xt, y_tip - width), (xt + ct, y_tip + width), (xr + cr, y_root + width)],
                dtype=float,
            )
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
    case2 = augmented_case(case)
    f_geo, a_geo, _, _, _ = build_geom_from_row(case2, ac)
    ref_geo = f_geo if f_geo[6] >= a_geo[6] else a_geo
    return float(ac.ref_dim_lift_surface(ref_geo))


def draw_aircraft(ax: plt.Axes, g: dict[str, float], color: str, label: str, dx: float, dy: float = 0.0) -> None:
    for prefix in ("f", "a"):
        poly = transform(surface_polygon(g, prefix), dx, dy)
        ax.fill(poly[:, 0], poly[:, 1], facecolor=color, edgecolor=color, alpha=0.16, linewidth=1.0)
        ax.plot(*poly.T, color=color, linewidth=1.4)
    for poly0 in fuselage_polygons(g):
        poly = transform(poly0, dx, dy)
        ax.fill(poly[:, 0], poly[:, 1], facecolor=color, edgecolor=color, alpha=0.10, linewidth=1.0)
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
    ax.text(x_mid, y_bottom - 0.30, label, color=color, fontsize=16, ha="center", va="top", fontweight="bold")


def metric_line(prefix: str, row: dict[str, float], info: dict[str, float], feasible: bool | None = None) -> str:
    text = (
        f"{prefix}: qT={float(info['q_g_per_ton_km']):.2f}, mtow={float(info['mtow_out']):.0f}, "
        f"cy={float(info['cy']):.3f}, K={float(info['K']):.2f}, p0={float(row['p0']):.2f}, "
        f"S_ref={secondary_s_ref(row):.2f}"
    )
    if feasible is not None:
        text += f", feasible={feasible}"
    return text


def render_branch(
    branch: str,
    avl_row: dict[str, float],
    avl_info: dict[str, float],
    mlp_row: dict[str, float],
    mlp_info: dict[str, float],
    recheck_info: dict[str, float],
    recheck_ok: bool,
    out_path: Path,
) -> None:
    avl_g = normalize_to_forward_origin(freecad_from_case(avl_row))
    mlp_g = normalize_to_forward_origin(freecad_from_case(mlp_row))
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
    dy_aircraft = -0.50

    draw_aircraft(ax, avl_g, "#1f77b4", "AVL best", avl_dx, dy_aircraft)
    draw_aircraft(ax, mlp_g, "#ff7f0e", "MLP best", mlp_dx, dy_aircraft)

    avl_mac = mac_for_case(avl_row)
    mlp_mac = mac_for_case(mlp_row)
    ax.scatter([avl_dx + float(avl_info["center_mass"]) * avl_mac], [dy_aircraft], s=70, marker="o", color="red", edgecolor="white", linewidth=0.8, zorder=8)
    ax.scatter([mlp_dx + float(mlp_info["center_mass"]) * mlp_mac], [dy_aircraft], s=70, marker="o", color="red", edgecolor="white", linewidth=0.8, zorder=8)
    ax.scatter([mlp_dx + float(recheck_info["center_mass"]) * mlp_mac], [dy_aircraft], s=70, marker="x", color="purple", linewidth=1.8, zorder=8)

    fig.suptitle(branch, fontsize=24, y=0.965)
    fig.text(0.06, 0.86, metric_line("AVL best", avl_row, avl_info), fontsize=15)
    fig.text(0.06, 0.82, metric_line("MLP best", mlp_row, mlp_info), fontsize=15)
    fig.text(0.06, 0.78, metric_line("AVL recheck on MLP", mlp_row, recheck_info, recheck_ok), fontsize=15)

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
    args = parse_args()
    mlp_dir = resolve_dir(args.mlp_dir)
    avl_dir = resolve_dir(args.avl_dir)
    outdir = resolve_dir(args.outdir)
    runs_dir = resolve_dir(args.runs_dir)
    out_branch = outdir / "top_view_by_branch"
    outdir.mkdir(parents=True, exist_ok=True)
    out_branch.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    backend = AvlBackend()
    rows: list[dict[str, float | str | bool | int]] = []
    image_paths: list[Path] = []

    for i, branch in enumerate(BRANCH_ORDER):
        avl_branch = avl_dir / branch
        mlp_branch = mlp_dir / branch
        if not avl_branch.is_dir() or not mlp_branch.is_dir():
            continue

        avl_gen, avl_idx, avl_row, avl_info = load_best_qt(avl_branch)
        mlp_gen, mlp_idx, mlp_row, mlp_info = load_best_qt(mlp_branch)
        avl_tuple = evaluate_candidate(mlp_row, backend, args.kk_base + i)
        recheck_info = dict(zip(INFO_COLS, avl_tuple[: len(INFO_COLS)]))
        recheck_ok = recheck_feasible(recheck_info, args.max_cy, args.bias_mz)

        out_path = out_branch / f"{branch}_topview_logic1_qt_avl_vs_mlp.png"
        render_branch(branch, avl_row, avl_info, mlp_row, mlp_info, recheck_info, recheck_ok, out_path)
        image_paths.append(out_path)

        rows.append(
            {
                "branch": branch,
                "avl_generation": avl_gen,
                "avl_row_index_0based": avl_idx,
                "mlp_generation": mlp_gen,
                "mlp_row_index_0based": mlp_idx,
                "avl_qT": float(avl_info["q_g_per_ton_km"]),
                "mlp_qT": float(mlp_info["q_g_per_ton_km"]),
                "recheck_qT": float(recheck_info["q_g_per_ton_km"]),
                "avl_mtow": float(avl_info["mtow_out"]),
                "mlp_mtow": float(mlp_info["mtow_out"]),
                "recheck_mtow": float(recheck_info["mtow_out"]),
                "avl_cy": float(avl_info["cy"]),
                "mlp_cy": float(mlp_info["cy"]),
                "recheck_cy": float(recheck_info["cy"]),
                "avl_K": float(avl_info["K"]),
                "mlp_K": float(mlp_info["K"]),
                "recheck_K": float(recheck_info["K"]),
                "avl_p0": float(avl_row["p0"]),
                "mlp_p0": float(mlp_row["p0"]),
                "recheck_p0": float(mlp_row["p0"]),
                "avl_S_ref": secondary_s_ref(avl_row),
                "mlp_S_ref": secondary_s_ref(mlp_row),
                "recheck_S_ref": secondary_s_ref(mlp_row),
                "avl_mz": float(avl_info["mz"]),
                "mlp_mz": float(mlp_info["mz"]),
                "recheck_mz": float(recheck_info["mz"]),
                "recheck_feasible": recheck_ok,
                "delta_qT_recheck_minus_mlp": float(recheck_info["q_g_per_ton_km"]) - float(mlp_info["q_g_per_ton_km"]),
                "delta_K_recheck_minus_mlp": float(recheck_info["K"]) - float(mlp_info["K"]),
                "delta_cy_recheck_minus_mlp": float(recheck_info["cy"]) - float(mlp_info["cy"]),
                "delta_qT_recheck_minus_avl": float(recheck_info["q_g_per_ton_km"]) - float(avl_info["q_g_per_ton_km"]),
                "delta_K_recheck_minus_avl": float(recheck_info["K"]) - float(avl_info["K"]),
                "delta_cy_recheck_minus_avl": float(recheck_info["cy"]) - float(avl_info["cy"]),
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(outdir / "logic1_qt_branch_compare_summary.csv", index=False)

    fig, axes = plt.subplots(6, 2, figsize=(16, 34))
    axes_arr = axes.flatten()
    fig.suptitle("Logic1 qT | AVL best vs MLP best with AVL recheck", fontsize=18, y=0.995, x=0.01, ha="left")
    for ax, branch, img_path in zip(axes_arr, BRANCH_ORDER, image_paths):
        ax.imshow(plt.imread(img_path))
        ax.set_title(branch, fontsize=13)
        ax.axis("off")
    for ax in axes_arr[len(image_paths):]:
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(outdir / "topview_grid.png", dpi=170)
    plt.close(fig)

    print(f"Wrote {outdir}")


if __name__ == "__main__":
    main()
