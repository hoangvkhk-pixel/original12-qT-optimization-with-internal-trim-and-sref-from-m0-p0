from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BranchSpec:
    name: str
    a_s_range: tuple[float, float]
    scheme_fuse: float
    scheme_vertical: float
    a_dihedral_range: tuple[float, float]


BRANCHES: list[BranchSpec] = [
    BranchSpec("normal_1_1", (0.2, 0.5), 0.0, 0.0, (30.0, 50.0)),
    BranchSpec("normal_1_2", (0.2, 0.5), 0.0, 0.5, (0.0, 0.0)),
    BranchSpec("normal_1_3", (0.2, 0.5), 0.0, 1.0, (30.0, 50.0)),
    BranchSpec("normal_2_1", (0.2, 0.5), 0.5, 0.0, (30.0, 50.0)),
    BranchSpec("normal_2_2", (0.2, 0.5), 0.5, 0.5, (0.0, 0.0)),
    BranchSpec("normal_2_3", (0.2, 0.5), 0.5, 1.0, (30.0, 50.0)),
    BranchSpec("normal_3_1", (0.2, 0.5), 1.0, 0.0, (30.0, 50.0)),
    BranchSpec("normal_3_2", (0.2, 0.5), 1.0, 0.5, (0.0, 0.0)),
    BranchSpec("normal_3_3", (0.2, 0.5), 1.0, 1.0, (30.0, 50.0)),
    BranchSpec("duck_1_x", (0.5, 0.8), 0.0, 0.5, (0.0, 10.0)),
    BranchSpec("duck_2_x", (0.5, 0.8), 0.5, 0.5, (0.0, 10.0)),
    BranchSpec("duck_3_x", (0.5, 0.8), 1.0, 0.5, (0.0, 10.0)),
]

TRAIN_H_BROAD_RANGE = (200.0, 5000.0)
TRAIN_H_FOCUS_RANGE = (450.0, 550.0)
TRAIN_H_FOCUS_FRACTION = 0.20


def branch_by_name(name: str) -> BranchSpec:
    for branch in BRANCHES:
        if branch.name == name:
            return branch
    raise KeyError(f"Unknown branch: {name}")
