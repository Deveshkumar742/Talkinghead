"""The TTS seam.

One method, one responsibility: text plus a reference clip in, a WAV out. The
reference clip is passed on every call rather than held as provider state,
because voice consistency across a multi-segment script depends on every segment
seeing the *same* reference -- making that an explicit argument means a caller
cannot accidentally drift.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


class TTSError(RuntimeError):
    """Synthesis failed."""


@runtime_checkable
class TTSProvider(Protocol):
    """Synthesizes speech in a cloned voice."""

    #: Key into ``config.TRUSTED_SOURCES``. Implementations must gate their
    #: model load through ``assert_commercially_licensed(model_key)``.
    model_key: str

    #: Sample rate of the WAVs this provider emits.
    sample_rate: int

    def synthesize(self, text: str, reference_wav: Path, out_path: Path) -> Path:
        """Speak ``text`` in the voice of ``reference_wav``, writing to ``out_path``.

        Must be deterministic enough that re-running a segment produces
        comparable output, and must raise :class:`TTSError` rather than emitting
        a silent or truncated file.
        """
        ...

    def warmup(self) -> None:
        """Load weights ahead of the first call.

        Called once before a batch so the cost of a multi-gigabyte model load is
        not attributed to segment zero.
        """
        ...
