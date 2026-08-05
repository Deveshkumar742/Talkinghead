"""Chatterbox TTS provider — clones a voice from a short reference clip.

Chatterbox (Resemble AI, MIT) is the only permissively-licensed zero-shot voice
cloner in this space; see ``config.REJECTED_MODELS`` for why XTTS-v2 and
VibeVoice are not options for client work.

Torch and chatterbox are imported lazily inside methods rather than at module
scope. That is deliberate: it keeps this module importable on a laptop with no
ML dependencies installed, which is what lets the pipeline and its tests be
verified without a GPU.

API confirmed against resemble-ai/chatterbox:

    from chatterbox.tts import ChatterboxTTS
    model = ChatterboxTTS.from_pretrained(device="cuda")
    wav = model.generate(text, audio_prompt_path="ref.wav")
    torchaudio.save("out.wav", wav, model.sr)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from talkinghead.config import assert_commercially_licensed
from talkinghead.tts.base import TTSError

log = logging.getLogger("talkinghead")

#: Chatterbox emits 24 kHz. Declared up front because the pipeline needs a rate
#: to assemble audio with before any model is loaded; corrected from ``model.sr``
#: once the weights are in memory.
DEFAULT_SAMPLE_RATE = 24_000


class ChatterboxTTS:
    """Synthesizes speech in a cloned voice via Chatterbox.

    One model instance is reused across every segment of a script, and every
    call passes the *same* reference clip. That is what keeps the voice stable:
    Chatterbox conditions on the reference per call, so varying it mid-script
    would make the speaker drift.
    """

    model_key = "chatterbox"

    def __init__(
        self,
        device: str = "cpu",
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
    ) -> None:
        """
        Args:
            device: ``cuda`` or ``cpu``. CPU works and needs no GPU session, but
                runs slower than realtime.
            exaggeration: Chatterbox's expressiveness control. The 0.5 default is
                neutral; higher is more dramatic, which reads as unprofessional
                for client explainers.
            cfg_weight: how tightly to adhere to the reference speaker. Higher
                tracks the reference more closely at some cost to naturalness.
        """
        assert_commercially_licensed(self.model_key)
        self.device = device
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight
        self._model: Any = None
        self._sample_rate = DEFAULT_SAMPLE_RATE

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def warmup(self) -> None:
        """Load the weights. Idempotent, so calling it repeatedly is free."""
        if self._model is not None:
            return

        try:
            import torch  # noqa: F401
            from chatterbox.tts import ChatterboxTTS as _Chatterbox
        except ImportError as exc:
            raise TTSError(
                "Chatterbox is not installed. In a Kaggle session:\n"
                "    pip install chatterbox-tts\n"
                "Note this pulls in torch, so it belongs in a notebook session "
                "rather than on a laptop."
            ) from exc

        log.info("loading Chatterbox on %s", self.device)
        self._model = _Chatterbox.from_pretrained(device=self.device)

        # Trust the model over the constant.
        reported = getattr(self._model, "sr", None)
        if isinstance(reported, int) and reported > 0:
            if reported != self._sample_rate:
                log.info(
                    "Chatterbox reports %d Hz (expected %d) -- using the model's",
                    reported,
                    self._sample_rate,
                )
            self._sample_rate = reported

    def synthesize(self, text: str, reference_wav: Path, out_path: Path) -> Path:
        """Speak ``text`` in the voice of ``reference_wav``.

        Raises:
            TTSError: if the reference is missing, generation fails, or the
                result is empty. An empty or near-silent file must raise rather
                than pass, since the pipeline would otherwise assemble silence
                into the middle of a video.
        """
        if not text.strip():
            raise TTSError("Refusing to synthesize empty text.")
        if not reference_wav.exists():
            raise TTSError(f"Voice reference not found: {reference_wav}")

        self.warmup()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import torchaudio

            wav = self._model.generate(
                text,
                audio_prompt_path=str(reference_wav),
                exaggeration=self.exaggeration,
                cfg_weight=self.cfg_weight,
            )
            torchaudio.save(str(out_path), wav.cpu(), self._sample_rate)
        except Exception as exc:  # noqa: BLE001 - upstream raises bare Exception
            raise TTSError(
                f"Chatterbox failed on {text[:60]!r}: {exc}"
            ) from exc

        if not out_path.exists() or out_path.stat().st_size == 0:
            raise TTSError(
                f"Chatterbox produced an empty file for {text[:60]!r}. "
                f"Silence here would be assembled into the video unnoticed."
            )
        return out_path
