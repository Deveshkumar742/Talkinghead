"""Shared fixtures.

Media fixtures are generated with ffmpeg rather than committed as binaries: they
stay tiny, they exercise the same code path the pipeline uses, and the repo never
carries sample video.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from talkinghead import media

PROJECT_ROOT = Path(__file__).resolve().parent.parent


#: Applied to tests that shell out to ffmpeg.
#:
#: FFmpeg is a host-provided dependency here -- the pipeline is meant to run in a
#: Kaggle or Colab session where it is preinstalled. So these tests skip rather
#: than fail on a bare local machine, and the pure-logic suite still runs
#: everywhere. In a cloud session the whole suite runs.
requires_ffmpeg = pytest.mark.skipif(
    not media.ffmpeg_available(),
    reason="ffmpeg/ffprobe not on PATH (expected in a Kaggle/Colab session)",
)


@pytest.fixture(scope="session")
def sample_script_text() -> str:
    return (PROJECT_ROOT / "samples" / "test_script.txt").read_text(encoding="utf-8")


@pytest.fixture
def tone_wav(tmp_path: Path):
    """A 1-second mono tone. Stand-in for a synthesized segment."""
    return media.make_tone(1.0, tmp_path / "tone.wav")


@pytest.fixture
def short_tone_wav(tmp_path: Path):
    """A 0.5-second tone at a different pitch, for concat-order checks."""
    return media.make_tone(0.5, tmp_path / "short.wav", freq=880)


@pytest.fixture
def test_video(tmp_path: Path):
    """A 2-second silent test pattern. Stand-in for the base loop."""
    return media.make_test_video(2.0, tmp_path / "base.mp4")
