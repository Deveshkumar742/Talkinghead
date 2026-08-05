"""Where are we running, and what does the host already provide?

This project is deliberately cloud-first. Kaggle and Colab sessions ship with
FFmpeg preinstalled and provide a free GPU, so the intended runtime for every
compute stage is a notebook session -- not the laptop. The laptop's job is to
edit scripts and watch finished videos.

Two practical consequences encoded here:

* FFmpeg is treated as a *host-provided* dependency. We never install it, and a
  missing binary points the user at a cloud session rather than at a package
  manager.
* Path layout differs per host. Kaggle mounts datasets read-only at
  ``/kaggle/input`` and only ``/kaggle/working`` is writable; Colab mounts Drive
  at ``/content/drive``. Both are resolved here so nothing else has to care.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class Host(StrEnum):
    """Which environment the process is running in."""

    KAGGLE = "kaggle"
    COLAB = "colab"
    LOCAL = "local"


@dataclass(frozen=True)
class HostInfo:
    """Detected host, plus what it provides."""

    host: Host
    has_ffmpeg: bool
    has_gpu: bool
    #: Read-only mount for uploaded assets and cached weights, if the host has one.
    input_root: Path | None
    #: Writable scratch directory.
    work_root: Path

    @property
    def is_cloud(self) -> bool:
        return self.host is not Host.LOCAL

    def describe(self) -> str:
        gpu = "GPU" if self.has_gpu else "CPU only"
        ffmpeg = "ffmpeg present" if self.has_ffmpeg else "NO ffmpeg"
        return f"{self.host} ({gpu}, {ffmpeg})"

    def available_mounts(self) -> list[str]:
        """Paths of the datasets actually mounted, relative to the input root.

        Two reasons this is not a simple ``iterdir``:

        1. Kaggle derives the mount directory from a *slugified* dataset title,
           so it frequently is not the string you typed.
        2. Kaggle has more than one layout. ``/kaggle/input/<slug>`` is the
           classic one, but current sessions also use
           ``/kaggle/input/datasets/<owner>/<slug>``. Listing only the top level
           reports a useless ``datasets`` entry and hides the real name.

        When a lookup misses, showing what is genuinely there is far more useful
        than repeating the name that failed.
        """
        if self.input_root is None or not self.input_root.exists():
            return []

        names: list[str] = []
        for entry in sorted(self.input_root.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name == "datasets":
                # Descend the owner level to reach the actual dataset slugs.
                for owner in sorted(entry.iterdir()):
                    if owner.is_dir():
                        names.extend(
                            f"datasets/{owner.name}/{d.name}"
                            for d in sorted(owner.iterdir())
                            if d.is_dir()
                        )
            else:
                names.append(entry.name)
        return names


def _detect_host() -> Host:
    """Identify the runtime.

    Kaggle sets ``KAGGLE_KERNEL_RUN_TYPE`` in its sessions and mounts
    ``/kaggle/working``; either signal is sufficient. Colab is identified by its
    importable ``google.colab`` module, which is the check Colab's own docs use.
    """
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/working").exists():
        return Host.KAGGLE
    if "google.colab" in sys.modules:
        return Host.COLAB
    try:
        import google.colab  # noqa: F401

        return Host.COLAB
    except ImportError:
        pass
    return Host.LOCAL


def _detect_gpu() -> bool:
    """True if a CUDA GPU looks usable.

    Checks for ``nvidia-smi`` rather than importing torch, so this stays cheap
    and works before the ML extras are installed.
    """
    return shutil.which("nvidia-smi") is not None


def detect() -> HostInfo:
    """Inspect the current environment. Cheap enough to call freely."""
    host = _detect_host()
    has_ffmpeg = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

    if host is Host.KAGGLE:
        input_root = Path("/kaggle/input")
        work_root = Path("/kaggle/working")
    elif host is Host.COLAB:
        drive = Path("/content/drive/MyDrive")
        input_root = drive if drive.exists() else None
        work_root = Path("/content")
    else:
        input_root = None
        work_root = Path.cwd()

    return HostInfo(
        host=host,
        has_ffmpeg=has_ffmpeg,
        has_gpu=_detect_gpu(),
        input_root=input_root if input_root and input_root.exists() else None,
        work_root=work_root,
    )


#: Guidance shown when FFmpeg is absent. Leads with the cloud path, because that
#: is the supported way to run this project -- a local install is a convenience
#: for running the test suite, not a requirement.
FFMPEG_MISSING_HELP = """\
FFmpeg was not found, and this project does not install it.

The compute stages are meant to run in a cloud notebook session, where FFmpeg is
already present:

  * Kaggle  -- ffmpeg is preinstalled. Open notebooks/kaggle_render.ipynb.
                CPU sessions are unmetered; only GPU sessions draw against the
                30 hours/week quota, so audio assembly costs you nothing.
  * Colab   -- ffmpeg is preinstalled. Same notebook works.

If you specifically want to run the test suite on this machine, FFmpeg is the
only local prerequisite:

  Windows:  winget install Gyan.FFmpeg   (then restart your shell)
  macOS:    brew install ffmpeg
  Linux:    apt-get install ffmpeg
"""
