"""The lipsync seam.

Both implementations behind this protocol repaint only the mouth region of a
real video, which is why ``sync`` returns a file at the *source* resolution --
callers depend on that to render 1080p without an upscale step.

The bake-off between LatentSync and MuseTalk is a genuine open question that
documentation cannot settle, so this interface exists specifically to make
swapping them a one-flag change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


class LipsyncError(RuntimeError):
    """Lipsync inference failed."""


@runtime_checkable
class LipsyncProvider(Protocol):
    """Rewrites a talking face's mouth to match new audio."""

    #: Key into ``config.TRUSTED_SOURCES``. Implementations must gate their
    #: model load through ``assert_commercially_licensed(model_key)``.
    model_key: str

    #: Face-crop resolution the model operates at. Callers use this to warn when
    #: the face in frame is larger than the crop, since that is when the
    #: regenerated mouth starts looking soft against the untouched frame.
    face_crop: int

    #: Whether this provider preserves the input resolution. Must be True for
    #: any provider used on the 1080p profile; full-frame generative models
    #: (InfiniteTalk, Wan-S2V) would set this False and require an upscale.
    preserves_resolution: bool

    def sync(self, video_path: Path, audio_path: Path, out_path: Path) -> Path:
        """Repaint the mouth in ``video_path`` to match ``audio_path``.

        The output must match the input frame count and resolution. Raises
        :class:`LipsyncError` on failure -- including the case where no face is
        detected, which otherwise fails silently by passing frames through
        untouched.
        """
        ...

    def estimated_vram_gb(self) -> float:
        """Peak VRAM this provider expects to need.

        Checked against the available device before a long render starts, so an
        out-of-memory failure surfaces in seconds rather than twenty minutes in.
        """
        ...
