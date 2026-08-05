"""Split a script into TTS segments.

Why segment at all: Chatterbox drifts in pitch and prosody over long
single-pass synthesis. Chunking and synthesizing each piece against the *same*
reference clip keeps the voice stable, and it means one bad sentence can be
re-rolled without redoing the whole script.

The pause values attached to each segment are also what a future captions
feature would need, which is why they are recorded per-segment rather than
applied blindly at concat time.

Pure functions, no I/O, no ML dependencies -- so this is testable on a laptop.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

# Sentence-final punctuation followed by whitespace. Deliberately conservative:
# a wrong split costs a slightly odd pause, while a missed split only makes a
# segment longer (and the length cap catches that anyway).
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+")

# Abbreviations that end in a period but do not end a sentence. Not exhaustive
# by design -- these are the ones that actually show up in business scripts.
_ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st",
        "vs", "etc", "eg", "ie", "approx", "dept", "est",
        "inc", "ltd", "co", "corp", "no", "fig", "vol",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
        "oct", "nov", "dec",
    }
)

_WHITESPACE = re.compile(r"[ \t]+")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n+")

# Straight-quote and dash normalisation. TTS models mispronounce or stumble on
# typographic characters that arrive via copy-paste from docs and email.
_REPLACEMENTS = {
    "‘": "'", "’": "'",      # single quotes
    "“": '"', "”": '"',      # double quotes
    "–": "-", "—": " - ",    # en/em dash
    "…": "...",                    # ellipsis
    " ": " ",                      # non-breaking space
    "‑": "-",                      # non-breaking hyphen
}


@dataclass(frozen=True)
class Segment:
    """One unit of synthesis.

    ``pause_after_ms`` is the silence inserted *after* this segment when the
    audio is concatenated. Paragraph boundaries get a longer pause than
    sentence boundaries, which is what makes multi-paragraph scripts sound
    deliberate rather than breathless.
    """

    index: int
    text: str
    pause_after_ms: int
    is_paragraph_end: bool = False

    @property
    def slug(self) -> str:
        """Stable filename stem, so the artifact cache can key on it."""
        return f"seg_{self.index:03d}"

    def cache_key(self) -> str:
        """Hash of the content that determines this segment's audio.

        Changing wording invalidates just that segment; changing a pause does
        not invalidate the synthesised speech at all, since pauses are applied
        during concat rather than baked into the WAV.
        """
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]


@dataclass
class PreparedScript:
    """A script broken into segments, plus the hash that keys its cache dir."""

    segments: list[Segment] = field(default_factory=list)
    source_text: str = ""

    @property
    def script_hash(self) -> str:
        return hashlib.sha256(self.source_text.encode("utf-8")).hexdigest()[:12]

    @property
    def total_chars(self) -> int:
        return sum(len(s.text) for s in self.segments)

    def estimated_duration_s(self, words_per_minute: float = 150.0) -> float:
        """Rough runtime estimate, including pauses.

        Used to warn before a render rather than to make decisions -- actual
        duration comes from probing the synthesised audio.
        """
        words = sum(len(s.text.split()) for s in self.segments)
        speech_s = (words / words_per_minute) * 60.0
        pause_s = sum(s.pause_after_ms for s in self.segments) / 1000.0
        return speech_s + pause_s

    def __len__(self) -> int:
        return len(self.segments)


def normalize_text(text: str) -> str:
    """Collapse whitespace and replace typographic characters.

    Paragraph breaks are preserved as ``\\n\\n`` because they carry pause
    information; every other run of whitespace collapses to a single space.
    """
    for old, new in _REPLACEMENTS.items():
        text = text.replace(old, new)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Mark paragraph breaks before collapsing, then restore them.
    text = _PARAGRAPH_BREAK.sub("\x00", text)
    text = text.replace("\n", " ")
    text = _WHITESPACE.sub(" ", text)
    paragraphs = [p.strip() for p in text.split("\x00")]
    return "\n\n".join(p for p in paragraphs if p)


def _ends_with_abbreviation(fragment: str) -> bool:
    """True if the fragment's final token is a known abbreviation."""
    tail = fragment.rstrip().rstrip(".")
    if not tail:
        return False
    last = re.split(r"[\s(\[\"']", tail)[-1].lower()

    # Strip internal periods so multi-period forms match their entry in the set:
    # "e.g" -> "eg", "i.e" -> "ie". These are common in business scripts, which
    # is the input this pipeline is built for.
    collapsed = last.replace(".", "")
    if collapsed in _ABBREVIATIONS:
        return True
    # Single initials such as "J." in "J. Smith".
    return len(collapsed) == 1 and collapsed.isalpha()


def split_sentences(paragraph: str) -> list[str]:
    """Split a paragraph into sentences, rejoining false splits."""
    raw = _SENTENCE_END.split(paragraph)
    sentences: list[str] = []
    for piece in raw:
        piece = piece.strip()
        if not piece:
            continue
        if sentences and _ends_with_abbreviation(sentences[-1]):
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)
    return sentences


def _split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    """Break an over-long sentence at the best available boundary.

    Preference order: clause punctuation, then conjunctions, then a hard word
    split. A hard split is audible, so it is the last resort -- but leaving the
    segment over-length is worse, because that is where Chatterbox drifts.
    """
    if len(sentence) <= max_chars:
        return [sentence]

    # Try clause boundaries first: semicolon, colon, then comma.
    for delimiter in ("; ", ": ", ", "):
        if delimiter not in sentence:
            continue
        parts = sentence.split(delimiter)
        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = f"{current}{delimiter}{part}" if current else part
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = part
        if current:
            chunks.append(current)
        # Only accept if this actually solved the problem.
        if all(len(c) <= max_chars for c in chunks) and len(chunks) > 1:
            return [c.strip() for c in chunks if c.strip()]

    # Fall back to packing whole words up to the cap.
    words = sentence.split()
    chunks = []
    current = ""
    for word in words:
        candidate = f"{current} {word}" if current else word
        if len(candidate) <= max_chars or not current:
            current = candidate
        else:
            chunks.append(current)
            current = word
    if current:
        chunks.append(current)
    return chunks


def prepare_script(
    text: str,
    max_chars: int = 280,
    sentence_pause_ms: int = 180,
    paragraph_pause_ms: int = 450,
) -> PreparedScript:
    """Turn raw script text into an ordered list of synthesis segments.

    Raises:
        ValueError: if the text contains no speakable content.
    """
    normalized = normalize_text(text)
    if not normalized:
        raise ValueError("Script is empty after normalization -- nothing to synthesize.")

    paragraphs = [p for p in normalized.split("\n\n") if p.strip()]

    segments: list[Segment] = []
    for para_idx, paragraph in enumerate(paragraphs):
        is_last_paragraph = para_idx == len(paragraphs) - 1
        sentences = split_sentences(paragraph)

        # Expand any over-long sentence into cap-respecting pieces, tracking
        # which piece is the true end of its sentence so pauses land correctly.
        pieces: list[tuple[str, bool]] = []
        for sentence in sentences:
            sub = _split_long_sentence(sentence, max_chars)
            for i, part in enumerate(sub):
                pieces.append((part, i == len(sub) - 1))

        for piece_idx, (piece_text, is_sentence_end) in enumerate(pieces):
            is_last_piece = piece_idx == len(pieces) - 1
            is_para_end = is_last_piece and not is_last_paragraph

            if is_last_piece and is_last_paragraph:
                pause = 0  # no trailing silence on the final segment
            elif is_para_end:
                pause = paragraph_pause_ms
            elif is_sentence_end:
                pause = sentence_pause_ms
            else:
                # Mid-sentence split: keep it tight so the seam is inaudible.
                pause = 0

            segments.append(
                Segment(
                    index=len(segments),
                    text=piece_text.strip(),
                    pause_after_ms=pause,
                    is_paragraph_end=is_para_end,
                )
            )

    return PreparedScript(segments=segments, source_text=normalized)
