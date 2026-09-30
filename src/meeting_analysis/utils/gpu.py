"""GPU hygiene helpers.

The 4 GB RTX 3050 cannot hold the ASR model and the diarization pipeline at the
same time, so stages unload themselves and release VRAM between runs.
"""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def register_cuda_dlls() -> None:
    """Make torch's bundled CUDA DLLs discoverable by CTranslate2.

    faster-whisper runs on CTranslate2, which loads cuBLAS/cuDNN at runtime and
    does not search ``site-packages/torch/lib`` by default. Without this the
    first GPU encode fails with::

        RuntimeError: Library cublas64_12.dll is not found or cannot be loaded
    """
    try:
        import torch
    except Exception:  # pragma: no cover
        return

    lib_dir = Path(torch.__file__).parent / "lib"
    if not lib_dir.is_dir():
        return

    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(str(lib_dir))
        except OSError:  # pragma: no cover
            logger.debug("add_dll_directory failed for %s", lib_dir, exc_info=True)

    os.environ["PATH"] = str(lib_dir) + os.pathsep + os.environ.get("PATH", "")


def free_cuda() -> None:
    """Collect garbage and release cached CUDA memory."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:  # pragma: no cover - best effort only
        logger.debug("CUDA cache release skipped", exc_info=True)


def vram_report() -> str:
    """Human-readable VRAM usage, for logging."""
    try:
        import torch

        if not torch.cuda.is_available():
            return "cuda:unavailable"
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return f"cuda allocated={allocated:.2f}GB reserved={reserved:.2f}GB"
    except Exception:  # pragma: no cover
        return "cuda:unknown"
