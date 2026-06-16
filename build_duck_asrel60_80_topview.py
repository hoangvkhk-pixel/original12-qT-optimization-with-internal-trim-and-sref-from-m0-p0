from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon


ROOT = Path(r"D:\OptimizationNewMLP")
REPO = Path(__file__).resolve().parent
OUT = REPO / "topview_duck_asrel60_80_20260616"
OUT_BRANCH = OUT / "top_view_by_branch"

BRANCHES = ["duck_1_x", "duck_2_x", "duck_3_x"]
SOURCES = {
    "Original AVL": ROOT
    / "OptimizationNewMLP_original12_logic1_fixedpoint_qt_dual10_portable"
    / "gen_avl_logic1_fixedpoint_qt",
    "Duck-clean MLP": ROOT
    / "OptimizationNewMLP_logic1_qt_duckonly_asrel60_80_portable"
    / "gen_mlp_logic1_qt_duckonly_asrel60_80",
    "Duck-clean AVL": ROOT
    / "OptimizationNewMLP_logic1_qt_duckonly_asrel60_80_portable"
    / "gen_avl_logic1_qt_duckonly_asrel60_80",
}
COLORS = {
    "Original AVL": "#666666",
    "Duck-clean MLP": "#1f77b4",
    "Duck-clean AVL": "#d95f02",
}
INFO_RE = re.compile(r"info_aircraft_(\d+)\.xlsx$")


def latest_generation(branch_dir: Path) -> int | None:
    gens: list[int] = []
    for path in branch_dir.glob("info_aircraft_*.xlsx"):
        match = INFO_RE.search(path.name)
        if match:
            gens.append(int(match.group(1)))
    return max(gens) if gens else None


def best_record(root: Path, branch: str) -> tuple[dict[str, float], dict[str, float]] | None:
    branch_dir = root / branch
    gen = latest_generation(branch_dir)
    if gen is None:
        return None

    info = pd.read_excel(branch_dir / f"info_aircraft_{gen}.xlsx")
    lx_path = branch_dir / f"Lx{gen}.npy"
    if lx_path.exists():
        idx = int(np.nanargmin(np.load(lx_path)))
    else:
        idx = int(pd.to_numeric(info["q_g_per_ton_km"], errors="coerce").idxmin())

    rec: dict[str, float] = {"generation": float(gen), "idx": float(idx)}
    if idx < len(info):
        for key, value in info.loc[idx].items():
            rec[str(key)] = value

    px_path = branch_dir / f"Px_{gen}.xlsx"
    if px_path.exists():
        px = pd.read_excel(px_path)
        if idx < len(px):
            for key, value in px.loc[idx].items():
                rec[str(key)] = value

    freecad = pd.read_excel(branch_dir / "info_to_FreeCAD.xlsx")
    geom = freecad.loc[idx].to_dict() if idx < len(freecad) else freecad.iloc[0].to_dict()
    return rec, geom


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
    x0 = float(g["fuse_x_loc"])
    nose = float(g["nose_f_aspect"]) * d
    center = float(g["center_f_aspect"]) * d
    tail = float(g["tail_f_aspect"]) * d
    x1 = x0 + nose
    x2 = x1 + center
    x3 = x2 + tail
    rear = max(float(surface_polygon(g, "f")[:, 0].max()), float(surface_polygon(g, "a")[:, 0].max()))
    x3 = max(x3, rear + 0.15 * d)
    x2 = x3 - tail
    half = 0.5 * d
    n_fuse = int(round(float(g.get("n_fuse", 1))))
    distance = float(g.get("distance_two_fuse", 0.0))
    centers = [0.0] if n_fuse != 2 else [-0.5 * distance, 0.5 * distance]
    return [
        np.asarray([(x0, yc), (x1, yc + half), (x2, yc + half), (x3, yc), (x2, yc - half), (x1, yc - half)], dtype=float)
        for yc in centers
    ]


def vertical_tail_polygons(g: dict[str, float]) -> list[np.ndarray]:
    if float(g.get("n_vertical", 0)) <= 0 or float(g.get("v_root_chord", 0)) <= 0:
        return []
    xr = float(g["v_x_loc_root_chord"])
    yr = float(g["v_y_loc_root_chord"])
    xt = float(g["v_x_loc_tip_chord"])
    yt = float(g["v_y_loc_tip_chord"])
    cr = float(g["v_root_chord"])
    ct = float(g["v_tip_chord"])
    width = max(0.08 * cr, 0.04)
    n_vertical = max(1, int(round(float(g.get("n_vertical", 1)))))
    distance = float(g.get("distance_two_fuse", 0.0))
    centers = [-0.5 * distance, 0.5 * distance] if n_vertical == 2 else [yr]
    return [
        np.asarray([(xr, yc - width), (xt, yc + yt - width), (xt + ct, yc + yt + width), (xr + cr, yc + width)], dtype=float)
        for yc in centers
    ]


def normalize_to_origin(g: dict[str, float]) -> dict[str, float]:
    out = dict(g)
    x0 = float(out["f_x_loc_root_chord"])
    for key in list(out):
        if key.endswith("_x_loc_root_chord") or key.endswith("_x_loc_tip_chord") or key == "fuse_x_loc":
            out[key] = float(out[key]) - x0
    return out


def all_points(g: dict[str, float]) -> np.ndarray:
    parts = [surface_polygon(g, "f"), surface_polygon(g, "a")]
    parts.extend(fuselage_polygons(g))
    parts.extend(vertical_tail_polygons(g))
    return np.vstack(parts)


def draw_aircraft(ax: plt.Axes, geom: dict[str, float], color: str, title: str, meta: str) -> None:
    g = normalize_to_origin(geom)
    for prefix in ("f", "a"):
        poly = surface_polygon(g, prefix)
        ax.add_patch(Polygon(poly, closed=True, facecolor=color, edgecolor=color, alpha=0.16, linewidth=1.2))
        ax.plot(*poly.T, color=color, linewidth=1.2)
    for poly in fuselage_polygons(g):
        ax.add_patch(Polygon(poly, closed=True, facecolor=color, edgecolor=color, alpha=0.10, linewidth=1.0))
        ax.plot(*poly.T, color=color, linewidth=1.0)
    for poly in vertical_tail_polygons(g):
        ax.add_patch(Polygon(poly, closed=True, facecolor=color, edgecolor=color, alpha=0.10, linewidth=0.9))
        ax.plot(*poly.T, color=color, linewidth=0.9)

    pts = all_points(g)
    pad_x = max(0.08 * np.ptp(pts[:, 0]), 0.5)
    pad_y = max(0.12 * np.ptp(pts[:, 1]), 0.5)
    ax.set_aspect("equal", "box")
    ax.set_xlim(float(pts[:, 0].min()) - pad_x, float(pts[:, 0].max()) + pad_x)
    ax.set_ylim(float(pts[:, 1].min()) - pad_y, float(pts[:, 1].max()) + pad_y)
    ax.axis("off")
    ax.set_title(title, fontsize=11)
    ax.text(
        0.02,
        0.98,
        meta,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )


def metric_text(branch: str, rec: dict[str, float]) -> str:
    return (
        f"{branch}\n"
        f"qT={float(rec.get('q_g_per_ton_km', np.nan)):.1f}, mtow={float(rec.get('mtow_out', np.nan)):.0f}\n"
        f"cy={float(rec.get('cy', np.nan)):.3f}, K={float(rec.get('K', np.nan)):.1f}, "
        f"aS={float(rec.get('a_S_rel', np.nan)):.3f}"
    )


def main() -> None:
    OUT_BRANCH.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, float | str]] = []

    for branch in BRANCHES:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=170)
        for ax, (label, root) in zip(axes, SOURCES.items()):
            data = best_record(root, branch)
            if data is None:
                ax.axis("off")
                ax.set_title(f"{label} missing")
                continue
            rec, geom = data
            draw_aircraft(ax, geom, COLORS[label], label, metric_text(branch, rec))
            row: dict[str, float | str] = {"branch": branch, "source": label}
            for key in [
                "generation",
                "idx",
                "q_g_per_ton_km",
                "mtow_out",
                "cy",
                "mz",
                "K",
                "A",
                "a_S_rel",
                "S_ref",
                "V",
                "H",
                "alpha_bal",
                "delta_bal",
            ]:
                row[key] = rec.get(key, np.nan)
            summary.append(row)
        fig.suptitle(f"{branch}: original vs duck-clean a_S_rel 0.6..0.8", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(OUT_BRANCH / f"{branch}_topview_original_vs_duck_asrel60_80.png")
        plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(12, 10), dpi=160)
    for row_idx, branch in enumerate(BRANCHES):
        for col_idx, (label, root) in enumerate(SOURCES.items()):
            ax = axes[row_idx, col_idx]
            data = best_record(root, branch)
            if data is None:
                ax.axis("off")
                continue
            rec, geom = data
            draw_aircraft(ax, geom, COLORS[label], label if row_idx == 0 else "", metric_text(branch, rec))
    fig.suptitle("Logic1 qT duck branches: original AVL vs duck-clean MLP/AVL", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "topview_duck_asrel60_80_grid.png")
    plt.close(fig)

    pd.DataFrame(summary).to_csv(OUT / "duck_asrel60_80_recheck_summary.csv", index=False)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
