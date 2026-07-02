from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(r"D:\OptimizationNewMLP")
BASE_SCRIPT_DIR = ROOT / "analysis_oldbranch_qt_topview_true_fuselage_boom_20260701"
if str(BASE_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_SCRIPT_DIR))

import plot_oldbranch_qt_topview_true_fuselage_boom as plotter  # type: ignore  # noqa: E402

plotter.WORK = ROOT / "analysis_oldbranch_qt_topview_true_fuselage_boom_20260701"
plotter.WORK.mkdir(parents=True, exist_ok=True)

plotter.AVL_ROOT = (
    ROOT
    / "OptimizationNewMLP_original12_oldbranch_avl_dampmz20_portable"
    / "gen_avl_original12_oldbranch_qt_np140_dampmz20"
)
plotter.MLP_ROOT = (
    ROOT
    / "OptimizationNewMLP_original12_oldbranch_mlp_batch_dampmz20_portable"
    / "gen_mlp_original12_oldbranch_qt_np140_dampmz20"
    / "gen_mlp_original12_oldbranch_qt_np140_dampmz20"
)


def main() -> None:
    plotter.render_grid(
        plotter.NORMAL_BRANCHES,
        plotter.WORK / "oldbranch_qt_damp20_topview_normal_avl_vs_mlp_true_fuselage_boom.png",
        (15, 12),
    )
    plotter.render_grid(
        plotter.DUCK_BRANCHES,
        plotter.WORK / "oldbranch_qt_damp20_topview_canard_avl_vs_mlp_true_fuselage_boom.png",
        (15, 4.2),
    )


if __name__ == "__main__":
    main()
