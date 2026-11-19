"""Shared helpers: project paths, ASCII-safe logging, seeding."""
from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ART_DIR = ROOT / "artifacts"
SEED = 42

_T0 = time.time()


def ascii_safe(s) -> str:
    """Return a string that can always be printed on a Windows console."""
    return str(s).encode("ascii", "replace").decode("ascii")


def log(msg) -> None:
    print(f"[{time.time() - _T0:7.1f}s] {ascii_safe(msg)}", flush=True)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self) -> float:
        return time.time() - self.t0
