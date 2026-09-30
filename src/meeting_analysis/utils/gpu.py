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


def free_vram_gb() -> float | None:
    """Free GPU memory in GB, or ``None`` when CUDA is unavailable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        return free_bytes / 1e9
    except Exception:  # pragma: no cover
        return None


def ensure_vram_headroom(min_free_gb: float = 1.2) -> None:
    """Fail early, and clearly, when the GPU is already occupied.

    cuDNN reports a genuinely misleading error when it cannot reserve
    convolution workspace::

        RuntimeError: GET was unable to find an engine to execute this computation

    That is really "out of memory", usually caused by an earlier run that
    crashed before releasing its models. Checking up front turns it into an
    actionable message.
    """
    free = free_vram_gb()
    if free is None:
        return

    if free < min_free_gb:
        free_cuda()
        free = free_vram_gb()

    if free is not None and free < min_free_gb:
        raise RuntimeError(
            f"Only {free:.2f} GB of GPU memory is free, but about "
            f"{min_free_gb:.2f} GB is needed. Another process is most likely "
            "still holding the GPU — typically a previous run that failed "
            "before releasing its models. Check `nvidia-smi`, close the stale "
            "python.exe, or run with --device cpu."
        )


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
