"""Runtime setup for a stable torchdrug import."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path


def _resolve_cache_dir() -> Path:
    # Project-local cache: <DSCrisk>/tmp
    return Path(__file__).resolve().parents[1] / "tmp"


def setup_torchdrug_env() -> Path:
    cache_dir = _resolve_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    ext_dir = cache_dir / "torch_extensions"
    ext_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(ext_dir))

    # Stub broken lmdb to avoid noisy failed cffi builds on import.
    if "lmdb" not in sys.modules:
        fake = types.ModuleType("lmdb")

        def _open(*args, **kwargs):
            raise RuntimeError("lmdb stub: DSCrisk does not use lmdb")

        fake.open = _open
        sys.modules["lmdb"] = fake

    return cache_dir
