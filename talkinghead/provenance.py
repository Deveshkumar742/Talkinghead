"""Provenance marking.

This is Devesh's own face and voice, which is the clean consent case -- but a
synthesized video should still never be mistakable for unedited camera footage.
Two layers, neither of which is a watermark in the anti-piracy sense:

1. Container metadata, always applied. Cheap, survives most copies, and readable
   with ``ffprobe``.
2. An optional visible corner label, for anything leaving the company.

Chatterbox additionally embeds PerTh neural watermarking in the *audio* by
default, so the soundtrack carries its own provenance signal without any work
here.
"""

from __future__ import annotations

from talkinghead import __version__

#: Applied to every render.
SYNTHETIC_TAG = "AI-generated talking-head video (synthesized lip motion and voice)"


def build_metadata(
    script_hash: str,
    lipsync_model: str,
    profile: str,
    created: str | None = None,
) -> dict[str, str]:
    """Metadata tags describing how a video was produced.

    Args:
        script_hash: identifies the source script without embedding its text.
        lipsync_model: which provider generated the mouth motion.
        profile: resolution profile name.
        created: ISO timestamp. Passed in rather than generated here so callers
            control it and the function stays deterministic for tests.
    """
    tags = {
        "comment": SYNTHETIC_TAG,
        "encoder": f"talkinghead {__version__}",
        "description": (
            f"Synthesized with lipsync={lipsync_model}, profile={profile}, "
            f"script={script_hash}. Face and voice are the speaker's own."
        ),
    }
    if created:
        tags["creation_time"] = created
    return tags


def watermark_filter(
    text: str = "AI-generated",
    font_size: int = 24,
    margin: int = 24,
    opacity: float = 0.75,
) -> str:
    """An ffmpeg ``drawtext`` filter placing a label in the bottom-right corner.

    Returned as a filter string rather than applied directly so it can be
    composed into the single scale/pad/encode pass instead of costing an extra
    generation.

    Note: ``drawtext`` requires an ffmpeg built with libfreetype. The Gyan.FFmpeg
    full build used here includes it.
    """
    safe = text.replace(":", r"\:").replace("'", r"\'")
    return (
        f"drawtext=text='{safe}'"
        f":fontsize={font_size}"
        f":fontcolor=white@{opacity:.2f}"
        f":box=1:boxcolor=black@0.35:boxborderw=8"
        f":x=w-tw-{margin}:y=h-th-{margin}"
    )
