# Original12 Qmission Optimization with Internal Trim

This repository contains the Original12 12-branch `q_g_per_ton_km`
optimization workflow using the fixed-point/internal-trim logic.

Core fixed-point logic:

`m0_guess -> cy_req = 9.81 * m0_guess / (q * S_ref) -> internal trim -> sizing -> mtow_out -> update m0_guess`

The current repository configuration uses a mixed branch definition:

- `normal_*`: unchanged original branch ranges, `a_S_rel in [0.2, 0.5]`
- `duck_*`: duck-clean branch ranges, `a_S_rel in [0.6, 0.8]`

The duck-clean setting prevents the duck branches from collapsing toward
`a_S_rel ~= 0.5`, which produced lower qT in some AVL runs but no longer
represented a clearly canard-like configuration.

## Included Workflows

- `mlp_portable/`: MLP split normal/duck backend
- `avl_portable/`: AVL backend

Both portable workflows include:

- runnable `src/`
- regenerated `init_h500/` initial populations matching the mixed branch ranges
- launcher `.bat` files
- AVL backend support files
- MLP model folders where needed

## Run

MLP:

```bat
mlp_portable\run_mlp_original12_fixedpoint_split300k_qmission3000_H500_cy06_optimize.bat
```

AVL:

```bat
avl_portable\run_avl_original12_fixedpoint_qmission3000_H500_cy06_optimize.bat
```

## Branch Logic

Normal branches remain unchanged:

| Branch group | `a_S_rel` | `scheme_fuse` | `scheme_vertical` |
|---|---:|---:|---:|
| `normal_1_*` | `0.2..0.5` | `0.0` | `0.0 / 0.5 / 1.0` |
| `normal_2_*` | `0.2..0.5` | `0.5` | `0.0 / 0.5 / 1.0` |
| `normal_3_*` | `0.2..0.5` | `1.0` | `0.0 / 0.5 / 1.0` |

Duck branches are tightened:

| Branch | `a_S_rel` | `scheme_fuse` | `scheme_vertical` |
|---|---:|---:|---:|
| `duck_1_x` | `0.6..0.8` | `0.0` | `0.5` |
| `duck_2_x` | `0.6..0.8` | `0.5` | `0.5` |
| `duck_3_x` | `0.6..0.8` | `1.0` | `0.5` |

## Duck AVL Recheck

The duck-clean rerun was compared against the original AVL duck result. The
important point is not that the duck-clean result always lowers qT; it does not.
The point is that it keeps the duck branches in a clearer canard-like geometry.

| Branch | Source | qT | mtow | cy | K | `a_S_rel` |
|---|---|---:|---:|---:|---:|---:|
| `duck_1_x` | Original AVL | 90.23 | 1464 | 0.570 | 28.14 | 0.800 |
| `duck_1_x` | Duck-clean MLP | 85.15 | 1380 | 0.599 | 28.14 | 0.800 |
| `duck_1_x` | Duck-clean AVL | 86.08 | 1421 | 0.564 | 28.64 | 0.800 |
| `duck_2_x` | Original AVL | 65.54 | 1254 | 0.598 | 33.26 | 0.500 |
| `duck_2_x` | Duck-clean MLP | 73.45 | 1193 | 0.599 | 28.21 | 0.800 |
| `duck_2_x` | Duck-clean AVL | 72.56 | 1264 | 0.524 | 30.24 | 0.800 |
| `duck_3_x` | Original AVL | 67.06 | 1269 | 0.574 | 32.89 | 0.501 |
| `duck_3_x` | Duck-clean MLP | 73.42 | 1189 | 0.598 | 28.15 | 0.800 |
| `duck_3_x` | Duck-clean AVL | 71.24 | 1249 | 0.534 | 30.43 | 0.800 |

Interpretation:

- `duck_1_x` improves with the duck-clean range.
- `duck_2_x` and `duck_3_x` lose qT versus the original AVL optimum because the
  original optimum exploited `a_S_rel ~= 0.5`.
- The duck-clean variants are geometrically more consistent with a true duck
  layout, but they are not the qT-minimum if the optimizer is allowed to use the
  `0.5` boundary.

## Duck Topview

Duck-clean comparison gallery:

- Full grid: `topview_duck_asrel60_80_20260616/topview_duck_asrel60_80_grid.png`
- Per-branch views: `topview_duck_asrel60_80_20260616/top_view_by_branch/`
- Recheck summary: `topview_duck_asrel60_80_20260616/duck_asrel60_80_recheck_summary.csv`

Preview:

![Duck-clean topview comparison](topview_duck_asrel60_80_20260616/topview_duck_asrel60_80_grid.png)

## Original Fixed-Point Topview

The older partial fixed-point gallery is kept for provenance:

- `topview_fixedpoint_partial_20260610/topview_grid.png`
- `topview_fixedpoint_partial_20260610/top_view_by_branch/`
- `topview_fixedpoint_partial_20260610/topview_summary.csv`

![Original fixed-point topview partial gallery](topview_fixedpoint_partial_20260610/topview_grid.png)

## Notes

This repository is intended to preserve the runnable qmission fixed-point setup
and the duck-clean comparison evidence. It intentionally excludes bulky previous
`gen_*` outputs and temporary run folders, while keeping the compact topview and
summary artifacts needed to understand the design choice.
