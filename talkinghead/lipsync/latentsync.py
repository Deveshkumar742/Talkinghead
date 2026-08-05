"""LatentSync provider — repaints the mouth on real footage to match new audio.

Driven as a subprocess (`python -m scripts.inference`), because that is the only
interface upstream offers. Arguments verified against ByteDance's own
``inference.sh`` at pinned commit a229c394.

Three things here exist because of measurements, not preference:

* **Weights are resolved from a cache before downloading.** The checkpoint set is
  9.8 GB, and re-fetching it costs 5-10 minutes of every session. Pointing
  ``cache_dir`` at a mounted Kaggle Dataset removes that entirely and is the
  single largest speedup available.
* **``num_frames`` is configurable and matters twice over.** It sets peak VRAM
  (the VAE encodes all frames as one batch) *and* the audio context per window.
  512 needs 8 to fit a 15 GB card, but 8 may sync worse than the upstream 16 --
  a genuine tension, not a free win.
* **``inference_steps`` is exposed.** Upstream defaults to 20 and documents
  20-50. Time scales roughly linearly with it, so lowering it is the most direct
  way to cut render time.
"""

from __future__ import annotations

import copy
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from talkinghead.config import Profile, assert_commercially_licensed
from talkinghead.lipsync.base import LipsyncError

log = logging.getLogger("talkinghead")

#: The commit Phase 0 actually exercised.
PINNED_SHA = "a229c3948406bc2cf6eaf4873e662e70c6a04746"
REPO_URL = "https://github.com/bytedance/LatentSync"

#: Files the inference script needs present under ``checkpoints/``.
REQUIRED_WEIGHTS = ("latentsync_unet.pt", "whisper/tiny.pt")


class LatentSync:
    """Mouth-region inpainting on an existing video.

    Preserves input resolution by construction: only the face crop is
    regenerated and composited back, so a 1080p source yields 1080p output. That
    property is why this can render 1080p without an upscale step, and the
    pipeline asserts it independently rather than trusting the flag.
    """

    model_key = "latentsync"
    preserves_resolution = True

    def __init__(
        self,
        profile: Profile,
        repo_dir: Path,
        cache_dir: Path | None = None,
        device: str = "cuda",
        inference_steps: int = 20,
        num_frames: int | None = None,
        seed: int = 1247,
        enable_deepcache: bool = True,
    ) -> None:
        """
        Args:
            profile: supplies the face crop, the matching checkpoint repo, and
                the validated ``num_frames`` for that crop size.
            repo_dir: where the LatentSync source lives (cloned if absent).
            cache_dir: read-only directory holding pre-downloaded weights,
                typically a mounted Kaggle Dataset. Skips the 9.8 GB download.
            device: informational only -- upstream picks CUDA itself.
            inference_steps: denoising steps. Lower is faster and slightly
                softer; upstream documents 20-50.
            num_frames: overrides the profile's value. Only set this when
                deliberately trading sync quality against VRAM.
            seed: fixed so reruns are comparable when judging quality by eye.
            enable_deepcache: upstream's own inference.sh passes this; it is a
                straight speedup, so leaving it off means rendering slower than
                necessary.
        """
        assert_commercially_licensed(self.model_key)
        self.profile = profile
        self.repo_dir = Path(repo_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.device = device
        self.inference_steps = inference_steps
        self.num_frames = num_frames if num_frames is not None else profile.num_frames
        self.seed = seed
        self.enable_deepcache = enable_deepcache

    @property
    def face_crop(self) -> int:
        return self.profile.face_crop

    def estimated_vram_gb(self) -> float:
        """Peak VRAM, from the Phase 0 measurement rather than a guess.

        512 at num_frames=16 measured 13.65 GB on a T4. The VAE encodes all
        frames in one batch, so the frame-dependent part scales roughly linearly
        while the resident model does not.
        """
        if self.face_crop >= 512:
            base, per_frame = 6.0, 0.48   # 6.0 + 16*0.48 ~= 13.7
        else:
            base, per_frame = 3.5, 0.12   # quarter the pixels per frame
        return base + per_frame * self.num_frames

    # ----------------------------------------------------------------
    # Setup
    # ----------------------------------------------------------------

    def ensure_repo(self) -> Path:
        """Clone the pinned revision if the repo is not already present."""
        if (self.repo_dir / "scripts" / "inference.py").is_file():
            return self.repo_dir

        self.repo_dir.parent.mkdir(parents=True, exist_ok=True)
        log.info("cloning LatentSync at %s", PINNED_SHA[:12])
        subprocess.run(
            ["git", "clone", "--quiet", REPO_URL, str(self.repo_dir)], check=True
        )
        # Pinned rather than tracking main, so upstream churn cannot silently
        # change behaviour between renders.
        subprocess.run(
            ["git", "checkout", "--quiet", PINNED_SHA],
            cwd=self.repo_dir,
            check=True,
        )
        return self.repo_dir

    def ensure_weights(self) -> Path:
        """Make the checkpoints available under ``repo_dir/checkpoints``.

        Order of preference:
          1. already present -- do nothing
          2. symlink the cache (instant; a mounted dataset is read-only, which
             is fine because inference only reads)
          3. download from HuggingFace (5-10 minutes)

        Raises:
            LipsyncError: if the weights cannot be obtained or are incomplete.
        """
        ckpt_dir = self.repo_dir / "checkpoints"

        if self._weights_complete(ckpt_dir):
            log.info("weights already present at %s", ckpt_dir)
            return ckpt_dir

        if self.cache_dir and self._weights_complete(self.cache_dir):
            log.info("linking cached weights from %s", self.cache_dir)
            if ckpt_dir.is_symlink() or ckpt_dir.is_file():
                ckpt_dir.unlink()
            elif ckpt_dir.is_dir():
                shutil.rmtree(ckpt_dir)
            try:
                ckpt_dir.symlink_to(self.cache_dir, target_is_directory=True)
            except OSError:
                # Windows without developer mode, or a filesystem that refuses
                # symlinks. Copying is slow but correct.
                log.warning("symlink refused; copying weights instead")
                shutil.copytree(self.cache_dir, ckpt_dir)
            return ckpt_dir

        log.info(
            "downloading %s (~9.8 GB). Cache this into a Kaggle Dataset to skip "
            "it next time -- it is the biggest single speedup available.",
            self.profile.hf_repo,
        )
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise LipsyncError(
                "huggingface_hub is required to fetch weights: pip install huggingface_hub"
            ) from exc

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=self.profile.hf_repo, local_dir=str(ckpt_dir))

        if not self._weights_complete(ckpt_dir):
            missing = [
                name for name in REQUIRED_WEIGHTS if not (ckpt_dir / name).is_file()
            ]
            raise LipsyncError(
                f"Weights incomplete after download from {self.profile.hf_repo}. "
                f"Missing: {missing}"
            )
        return ckpt_dir

    @staticmethod
    def _weights_complete(directory: Path) -> bool:
        if not directory.exists():
            return False
        return all((directory / name).is_file() for name in REQUIRED_WEIGHTS)

    def _write_config(self, ckpt_dir: Path) -> Path:
        """Emit a config variant carrying this provider's resolution and frames.

        Upstream ships stage2.yaml (256) and stage2_512.yaml (512). stage1
        variants are *training* configs and would silently do the wrong thing, so
        only stage2 is ever selected.
        """
        try:
            import yaml
        except ImportError as exc:
            raise LipsyncError("pyyaml is required: pip install pyyaml") from exc

        unet_dir = self.repo_dir / "configs" / "unet"
        preferred = "stage2_512.yaml" if self.face_crop >= 512 else "stage2.yaml"
        source = unet_dir / preferred

        if not source.is_file():
            available = sorted(p.name for p in unet_dir.glob("stage2*.yaml"))
            raise LipsyncError(
                f"{preferred} not found in {unet_dir}. Available stage2 configs: "
                f"{available}"
            )

        cfg = yaml.safe_load(source.read_text())
        cfg = copy.deepcopy(cfg)
        cfg.setdefault("data", {})
        cfg["data"]["resolution"] = self.face_crop
        cfg["data"]["num_frames"] = self.num_frames

        out = self.repo_dir / f"_th_unet_{self.face_crop}_nf{self.num_frames}.yaml"
        out.write_text(yaml.safe_dump(cfg, sort_keys=False))
        return out

    # ----------------------------------------------------------------
    # Inference
    # ----------------------------------------------------------------

    def sync(self, video_path: Path, audio_path: Path, out_path: Path) -> Path:
        """Repaint the mouth in ``video_path`` to match ``audio_path``.

        Raises:
            LipsyncError: on any failure, with the memory case named explicitly
                since that has a different remedy from every other error.
        """
        for label, path in (("video", video_path), ("audio", audio_path)):
            if not Path(path).exists():
                raise LipsyncError(f"Lipsync {label} input not found: {path}")

        self.ensure_repo()
        ckpt_dir = self.ensure_weights()
        config_path = self._write_config(ckpt_dir)
        unet_ckpt = ckpt_dir / self.profile.checkpoint

        out_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, "-m", "scripts.inference",
            "--unet_config_path", str(config_path),
            "--inference_ckpt_path", str(unet_ckpt),
            "--inference_steps", str(self.inference_steps),
            "--guidance_scale", "1.5",
            "--seed", str(self.seed),
            "--video_path", str(video_path),
            "--audio_path", str(audio_path),
            "--video_out_path", str(out_path),
        ]
        if self.enable_deepcache:
            cmd.append("--enable_deepcache")

        # Recommended by torch's own OOM message; only reclaims fragmentation,
        # so it is a small win rather than a fix for being genuinely short.
        env = dict(os.environ, PYTORCH_ALLOC_CONF="expandable_segments:True")

        log.info(
            "lipsync: %dpx crop, num_frames=%d, %d steps (est. %.1f GB VRAM)",
            self.face_crop, self.num_frames, self.inference_steps,
            self.estimated_vram_gb(),
        )
        proc = subprocess.run(
            cmd, cwd=self.repo_dir, capture_output=True, text=True, env=env
        )

        if proc.returncode != 0 or not out_path.exists():
            combined = (proc.stdout + proc.stderr).lower()
            tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
            if "out of memory" in combined:
                raise LipsyncError(
                    f"CUDA out of memory at {self.face_crop}px with "
                    f"num_frames={self.num_frames}.\n"
                    f"Lower num_frames, or switch to the 720p profile "
                    f"(TH_PROFILE=720p) which uses a 256 crop.\n\n{tail}"
                )
            raise LipsyncError(
                f"LatentSync failed (exit {proc.returncode}). This is not a "
                f"memory problem, so changing profile will not help.\n\n{tail}"
            )
        return out_path
