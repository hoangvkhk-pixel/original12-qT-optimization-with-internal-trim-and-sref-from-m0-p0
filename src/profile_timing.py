from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def enabled() -> bool:
    return os.environ.get("NEW20_PROFILE", "0").strip().lower() in {"1", "true", "yes", "on"}


def _profile_dir() -> Path:
    raw = os.environ.get("NEW20_PROFILE_DIR", "").strip()
    if raw:
        path = Path(raw)
    else:
        path = Path(os.environ.get("NEW20_PROFILE_GEN_PATH", ".")) / "_profile"
    path.mkdir(parents=True, exist_ok=True)
    return path


def record(section: str, seconds: float, **meta) -> None:
    if not enabled():
        return
    row = {
        "ts": time.time(),
        "pid": os.getpid(),
        "section": section,
        "seconds": float(seconds),
    }
    row.update(meta)
    path = _profile_dir() / f"profile_{os.getpid()}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


@contextmanager
def timer(section: str, **meta) -> Iterator[None]:
    if not enabled():
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        record(section, time.perf_counter() - t0, **meta)
