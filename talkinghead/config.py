"""Configuration, pinned model revisions, and resolution profiles.

Two things in here are load-bearing and should not be casually edited:

1. ``TRUSTED_SOURCES`` — every model is pulled from its vendor's own GitHub or
   HuggingFace org. The web is full of lookalike aggregator domains and
   third-party re-uploads; those are not acceptable sources for weights that end
   up in client-facing work product.

2. ``REJECTED_MODELS`` — the popular defaults in this space are mostly
   non-commercial. They are recorded here by name so nobody reintroduces one
   from a tutorial six months from now.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from talkinghead.runtime import Host, detect

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Licensing: what we may use, and what we may not
# --------------------------------------------------------------------------

class TrustedSource(BaseModel):
    """A first-party source for a model, with its verified license."""

    name: str
    repo: str
    hf_repo: str | None = None
    revision: str
    license: str


#: Only these. Pinned by revision so upstream churn cannot silently change what
#: we ship. Revisions are deliberately set to ``main`` until Phase 0 confirms a
#: working commit, then frozen to that SHA -- see ``freeze_revisions`` below.
TRUSTED_SOURCES: dict[str, TrustedSource] = {
    "latentsync": TrustedSource(
        name="LatentSync 1.6",
        repo="https://github.com/bytedance/LatentSync",
        hf_repo="ByteDance/LatentSync-1.6",
        revision="main",
        license="Apache-2.0",
    ),
    "musetalk": TrustedSource(
        name="MuseTalk",
        repo="https://github.com/TMElyralab/MuseTalk",
        hf_repo="TMElyralab/MuseTalk",
        revision="main",
        license="MIT",
    ),
    "chatterbox": TrustedSource(
        name="Chatterbox",
        repo="https://github.com/resemble-ai/chatterbox",
        hf_repo="ResembleAI/chatterbox",
        revision="main",
        license="MIT",
    ),
    "gfpgan": TrustedSource(
        name="GFPGAN",
        repo="https://github.com/TencentARC/GFPGAN",
        hf_repo=None,
        revision="main",
        license="Apache-2.0",
    ),
}

#: Do NOT reintroduce these. Each is a popular default in this space and each
#: forbids the commercial use this project is for. Mapped name -> reason.
REJECTED_MODELS: dict[str, str] = {
    "wav2lip": (
        "Weights trained on the LRS2 dataset; commercial use strictly "
        "prohibited. Use LatentSync or MuseTalk instead."
    ),
    "codeformer": (
        "NTU S-Lab License 1.0 — non-commercial. Use GFPGAN (Apache-2.0) "
        "instead."
    ),
    "xtts-v2": (
        "Coqui Public Model License (CPML) — non-commercial. Use Chatterbox "
        "(MIT) instead."
    ),
    "vibevoice": (
        "Microsoft removed the TTS code and scopes the model to research and "
        "development, not commercial deployment. Use Chatterbox (MIT) instead."
    ),
}


class LicenseError(RuntimeError):
    """Raised when something would pull in a non-commercial model."""


def assert_commercially_licensed(model_key: str) -> TrustedSource:
    """Gate every model load through this.

    Turns a licensing mistake into an import-time crash rather than a legal
    problem discovered after a video ships to a client.
    """
    key = model_key.lower().strip()
    if key in REJECTED_MODELS:
        raise LicenseError(
            f"{model_key!r} is not usable in this project: {REJECTED_MODELS[key]}"
        )
    if key not in TRUSTED_SOURCES:
        raise LicenseError(
            f"{model_key!r} is not in TRUSTED_SOURCES. Add it only after "
            f"verifying its weights (not just its code) permit commercial use, "
            f"and only with a first-party repo URL."
        )
    return TRUSTED_SOURCES[key]


# --------------------------------------------------------------------------
# Resolution profiles
# --------------------------------------------------------------------------

class LipsyncModel(StrEnum):
    LATENTSYNC = "latentsync"
    MUSETALK = "musetalk"


class Profile(BaseModel):
    """A resolution/quality pairing validated as a unit.

    The face-crop size and the output height are coupled: rendering taller than
    roughly the crop size wastes nothing but *does* make the regenerated mouth
    visibly softer than the untouched rest of the frame.
    """

    name: str
    output_width: int
    output_height: int
    face_crop: int
    checkpoint: str
    notes: str


#: Primary. LatentSync 1.6 was retrained on 512x512 video specifically to fix
#: the blurry teeth/lips of 1.5, which is what makes 1080p viable here.
PROFILE_1080P = Profile(
    name="1080p",
    output_width=1920,
    output_height=1080,
    face_crop=512,
    checkpoint="latentsync_unet.pt",
    notes="LatentSync 1.6 @ 512. Requires Phase 0 VRAM confirmation on a 16GB GPU.",
)

#: Fallback if 512 inference does not fit in Kaggle's 16GB. Downscaling the full
#: frame to 720p pulls the whole image toward the mouth's true resolution, which
#: hides most of the 256-crop softness.
PROFILE_720P = Profile(
    name="720p",
    output_width=1280,
    output_height=720,
    face_crop=256,
    checkpoint="latentsync_unet.pt",
    notes="LatentSync 1.5 @ 256. Fallback; fits comfortably in 16GB.",
)

PROFILES: dict[str, Profile] = {p.name: p for p in (PROFILE_1080P, PROFILE_720P)}


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

class Settings(BaseSettings):
    """Runtime settings. Override via env vars (``TH_`` prefix) or ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="TH_", env_file=".env", extra="ignore"
    )

    # Where things live
    project_root: Path = PROJECT_ROOT
    assets_dir: Path = PROJECT_ROOT / "assets"
    work_dir: Path = PROJECT_ROOT / "work"
    out_dir: Path = PROJECT_ROOT / "out"
    weights_dir: Path = PROJECT_ROOT / "weights"

    base_loop: Path = PROJECT_ROOT / "assets" / "base_loop.mp4"
    reference_wav: Path = PROJECT_ROOT / "assets" / "reference.wav"

    # Compute
    device: str = "cpu"
    profile: str = "1080p"
    lipsync_model: LipsyncModel = LipsyncModel.LATENTSYNC

    # TTS segmentation. Chatterbox drifts in pitch and prosody over long
    # single-pass synthesis, so the script is chunked and each chunk is
    # synthesized against the same reference clip.
    max_chars_per_segment: int = Field(default=280, ge=40, le=1000)
    sentence_pause_ms: int = Field(default=180, ge=0, le=2000)
    paragraph_pause_ms: int = Field(default=450, ge=0, le=3000)

    # Lipsync windowing. Kaggle sessions time out, so long videos are rendered
    # in overlapping windows and crossfaded at the seams.
    lipsync_window_s: float = Field(default=25.0, gt=1.0)
    lipsync_overlap_s: float = Field(default=0.5, ge=0.0)

    # Output encoding
    fps: int = 25
    video_crf: int = Field(default=18, ge=0, le=51)
    audio_bitrate: str = "192k"
    loudness_lufs: float = -16.0

    watermark: bool = False
    use_cache: bool = True

    #: Name of the private Kaggle Dataset holding the base loop, voice
    #: reference, and cached model weights. Mounted read-only at
    #: ``/kaggle/input/<this>``. Keep the dataset PRIVATE -- it is a likeness and
    #: a voice, and a public dataset makes both freely downloadable.
    kaggle_dataset: str = "talkinghead-assets"

    @property
    def active_profile(self) -> Profile:
        if self.profile not in PROFILES:
            raise ValueError(
                f"Unknown profile {self.profile!r}. Choose from {sorted(PROFILES)}."
            )
        return PROFILES[self.profile]

    def kaggle_mode(self) -> bool:
        """True when running inside a Kaggle notebook session."""
        return detect().host is Host.KAGGLE


#: How deep to look for an asset inside a mounted dataset. Kaggle preserves the
#: directory structure of whatever was uploaded, so files often arrive nested one
#: level down -- particularly from a ZIP, or from dragging in a folder.
_ASSET_SEARCH_DEPTH = 2


def find_asset(mount: Path, filename: str) -> Path:
    """Locate ``filename`` inside a mounted dataset.

    Checks the mount root first, then progressively deeper, and returns the
    shallowest match. Falls back to ``mount / filename`` when nothing is found,
    so callers still produce a sensible "missing file" message rather than a
    confusing ``None``.

    This exists because a dataset that *looks* correctly uploaded can still have
    its files one directory down, and the resulting "base_loop.mp4 not found" is
    baffling when you can plainly see the file in Kaggle's file browser.
    """
    if not mount.exists():
        return mount / filename

    direct = mount / filename
    if direct.is_file():
        return direct

    for depth in range(2, _ASSET_SEARCH_DEPTH + 1):
        pattern = "/".join(["*"] * (depth - 1) + [filename])
        for candidate in sorted(mount.glob(pattern)):
            if candidate.is_file():
                return candidate

    return direct


def load_settings(**overrides: object) -> Settings:
    """Build settings, then adapt paths to whichever host we are on.

    This project runs its compute stages in the cloud, so the common case is a
    notebook session with a read-only asset mount and a single writable scratch
    directory. Explicit overrides are re-applied last so a caller can always
    force a path regardless of host.

    Kaggle: assets and weights arrive via a private Dataset at
    ``/kaggle/input/<kaggle_dataset>``; only ``/kaggle/working`` is writable.

    Colab: assets come from Google Drive once mounted; ``/content`` is writable.
    """
    settings = Settings(**overrides)  # type: ignore[arg-type]
    host = detect()

    if host.host is Host.KAGGLE:
        mount = Path("/kaggle/input") / settings.kaggle_dataset
        if mount.exists():
            settings.assets_dir = mount
            settings.base_loop = find_asset(mount, "base_loop.mp4")
            settings.reference_wav = find_asset(mount, "reference.wav")
            settings.weights_dir = mount / "weights"
        settings.work_dir = host.work_root / "work"
        settings.out_dir = host.work_root / "out"

    elif host.host is Host.COLAB:
        if host.input_root is not None:
            mount = host.input_root / "talkinghead"
            if mount.exists():
                settings.assets_dir = mount
                settings.base_loop = find_asset(mount, "base_loop.mp4")
                settings.reference_wav = find_asset(mount, "reference.wav")
                settings.weights_dir = mount / "weights"
        settings.work_dir = host.work_root / "work"
        settings.out_dir = host.work_root / "out"

    # A GPU session should not silently run on CPU, and vice versa.
    if "device" not in overrides:
        settings.device = "cuda" if host.has_gpu else "cpu"

    # Caller-supplied paths win over host defaults.
    for key in ("work_dir", "out_dir", "assets_dir", "base_loop", "reference_wav",
                "weights_dir", "device"):
        if key in overrides:
            setattr(settings, key, overrides[key])

    return settings
