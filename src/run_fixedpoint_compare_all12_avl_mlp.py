from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd

from smoke_fixedpoint_compare_branch_avl_mlp import (
    CASE_PATH,
    full_fixedpoint_avl_parallel,
    full_fixedpoint_mlp,
    select_case,
    select_mlp,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = PROJECT_ROOT / "analysis_fixedpoint_compare_all12_avl_mlp.json"
OUT_CSV = PROJECT_ROOT / "analysis_fixedpoint_compare_all12_avl_mlp_summary.csv"


def summarize(branch: str, kind: str, result: dict[str, object], elapsed: float) -> dict[str, float | str]:
    cruise = result["cruise_final"]
    climb = result["climb_final"]
    declimb = result["declimb_final"]
    history = result["history"]
    last = history[-1]
    return {
        "branch": branch,
        "kind": kind,
        "time_sec": elapsed,
        "iters": len(history),
        "m0_final_for_trim": result["m0_final_for_trim"],
        "cy_req_final": result["cy_req_final"],
        "last_diff_abs": last["diff_abs"],
        "last_diff_rel": last["diff_rel"],
        "cruise_alpha": cruise["alpha"],
        "cruise_delta": cruise["delta"],
        "cruise_cx": cruise["cx"],
        "cruise_cy": cruise["cy"],
        "cruise_K": cruise["K"],
        "climb_cy": climb["cy"],
        "climb_cy_target": climb["cy_target"],
        "declimb_cy": declimb["cy"],
        "declimb_cy_target": declimb["cy_target"],
    }


def main() -> None:
    df = pd.read_csv(CASE_PATH)
    branches = df["branch"].tolist()
    all_rows: list[dict[str, float | str]] = []
    all_detail: dict[str, object] = {"branches": []}

    for i, branch in enumerate(branches):
        case = select_case(branch)

        t0 = time.perf_counter()
        avl = full_fixedpoint_avl_parallel(case, 20000 + i * 100000, workers=6)
        t1 = time.perf_counter()

        os.environ["OMP_NUM_THREADS"] = "6"
        os.environ["TF_NUM_INTRAOP_THREADS"] = "6"
        os.environ["TF_NUM_INTEROP_THREADS"] = "6"
        mlp_backend = select_mlp(case)
        mlp = full_fixedpoint_mlp(case, mlp_backend, 30000 + i * 100000)
        t2 = time.perf_counter()

        avl_row = summarize(branch, "avl_parallel6", avl, t1 - t0)
        mlp_row = summarize(branch, "mlp_threads6", mlp, t2 - t1)
        all_rows.extend([avl_row, mlp_row])
        all_detail["branches"].append(
            {
                "branch": branch,
                "case": case,
                "timing_sec": {"avl_parallel6": t1 - t0, "mlp_threads6": t2 - t1},
                "avl_parallel6": avl,
                "mlp_threads6": mlp,
            }
        )
        print(
            f"{branch}: AVL {t1 - t0:.3f}s iters={avl_row['iters']} "
            f"m0={avl_row['m0_final_for_trim']:.3f} | "
            f"MLP {t2 - t1:.3f}s iters={mlp_row['iters']} "
            f"m0={mlp_row['m0_final_for_trim']:.3f}"
        )

    pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps(all_detail, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
