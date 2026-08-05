"""Tests for the Chatterbox and LatentSync providers.

Neither model is loaded here. The point is to verify the parts that are ours —
licence gating, weight resolution, config generation, error classification — on a
laptop with no GPU and no torch. Getting a model to produce good output is a
judgement call made by watching video; getting these paths right is not.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from talkinghead.config import PROFILES, LicenseError
from talkinghead.lipsync.base import LipsyncError
from talkinghead.lipsync.latentsync import PINNED_SHA, REQUIRED_WEIGHTS, LatentSync
from talkinghead.tts.base import TTSError
from talkinghead.tts.chatterbox_local import ChatterboxTTS


class TestChatterboxConstruction:
    def test_declares_the_licensed_model_key(self):
        assert ChatterboxTTS().model_key == "chatterbox"

    def test_passes_the_licence_gate(self):
        # Constructing must not raise -- Chatterbox is MIT.
        ChatterboxTTS()

    def test_declares_a_sample_rate_before_any_model_load(self):
        # The pipeline needs a rate to assemble audio with before warmup.
        assert ChatterboxTTS().sample_rate == 24_000

    def test_defaults_to_cpu(self):
        # CPU means script iteration needs no GPU session.
        assert ChatterboxTTS().device == "cpu"

    def test_expressiveness_knobs_are_settable(self):
        provider = ChatterboxTTS(exaggeration=0.3, cfg_weight=0.7)
        assert (provider.exaggeration, provider.cfg_weight) == (0.3, 0.7)


class TestChatterboxSynthesizeGuards:
    """Input validation, which runs before any model would be touched."""

    def test_empty_text_is_refused(self, tmp_path):
        ref = tmp_path / "ref.wav"
        ref.write_bytes(b"x")
        with pytest.raises(TTSError, match="empty text"):
            ChatterboxTTS().synthesize("   ", ref, tmp_path / "out.wav")

    def test_missing_reference_is_reported(self, tmp_path):
        with pytest.raises(TTSError, match="reference not found"):
            ChatterboxTTS().synthesize("hello", tmp_path / "gone.wav",
                                       tmp_path / "out.wav")

    def test_missing_dependency_explains_where_to_install(self, tmp_path, monkeypatch):
        # On a laptop chatterbox is absent, and the message should point at a
        # notebook session rather than suggesting a local install.
        ref = tmp_path / "ref.wav"
        ref.write_bytes(b"x")
        provider = ChatterboxTTS()

        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) \
            else __builtins__.__import__

        def blocked(name, *args, **kwargs):
            if name.startswith("chatterbox"):
                raise ImportError("no chatterbox")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", blocked)
        with pytest.raises(TTSError, match="Kaggle session"):
            provider.synthesize("hello", ref, tmp_path / "out.wav")


class TestChatterboxSynthesize:
    """Behaviour with a stand-in model, exercising our code around it."""

    def _provider_with_fake_model(self, tmp_path, monkeypatch, *, emit_bytes=b"RIFFdata"):
        provider = ChatterboxTTS()

        class FakeWav:
            def cpu(self):
                return self

        class FakeModel:
            sr = 24_000

            def generate(self, text, audio_prompt_path, **kwargs):
                self.last = (text, audio_prompt_path, kwargs)
                return FakeWav()

        provider._model = FakeModel()

        saved: dict = {}

        class FakeTorchaudio:
            @staticmethod
            def save(path, wav, rate):
                saved["path"] = path
                saved["rate"] = rate
                Path(path).write_bytes(emit_bytes)

        monkeypatch.setitem(__import__("sys").modules, "torchaudio", FakeTorchaudio)
        return provider, saved

    def test_writes_a_file_at_the_model_sample_rate(self, tmp_path, monkeypatch):
        provider, saved = self._provider_with_fake_model(tmp_path, monkeypatch)
        ref = tmp_path / "ref.wav"
        ref.write_bytes(b"x")

        out = provider.synthesize("Hello there.", ref, tmp_path / "seg.wav")
        assert out.exists()
        assert saved["rate"] == 24_000

    def test_passes_the_reference_clip_through(self, tmp_path, monkeypatch):
        # Voice consistency depends on every segment seeing the same reference.
        provider, _ = self._provider_with_fake_model(tmp_path, monkeypatch)
        ref = tmp_path / "ref.wav"
        ref.write_bytes(b"x")

        provider.synthesize("Hello.", ref, tmp_path / "seg.wav")
        text, prompt, kwargs = provider._model.last
        assert text == "Hello."
        assert prompt == str(ref)
        assert kwargs["exaggeration"] == 0.5

    def test_empty_output_raises_rather_than_passing_silence(
        self, tmp_path, monkeypatch
    ):
        # Silence assembled into the middle of a video is the worst failure mode
        # here, because nothing errors and the defect is only found on playback.
        provider, _ = self._provider_with_fake_model(
            tmp_path, monkeypatch, emit_bytes=b""
        )
        ref = tmp_path / "ref.wav"
        ref.write_bytes(b"x")

        with pytest.raises(TTSError, match="empty file"):
            provider.synthesize("Hello.", ref, tmp_path / "seg.wav")


class TestLatentSyncConstruction:
    def test_declares_the_licensed_model_key(self):
        provider = LatentSync(profile=PROFILES["720p"], repo_dir=Path("/tmp/ls"))
        assert provider.model_key == "latentsync"

    def test_preserves_resolution(self):
        # The 1080p profile depends on this being true; the pipeline also
        # verifies it by probing, but the declaration must agree.
        provider = LatentSync(profile=PROFILES["1080p"], repo_dir=Path("/tmp/ls"))
        assert provider.preserves_resolution is True

    def test_takes_num_frames_from_the_profile(self):
        provider = LatentSync(profile=PROFILES["1080p"], repo_dir=Path("/tmp/ls"))
        assert provider.num_frames == PROFILES["1080p"].num_frames == 8

    def test_num_frames_can_be_overridden(self):
        provider = LatentSync(
            profile=PROFILES["1080p"], repo_dir=Path("/tmp/ls"), num_frames=16
        )
        assert provider.num_frames == 16

    def test_face_crop_follows_the_profile(self):
        assert LatentSync(PROFILES["1080p"], Path("/tmp/ls")).face_crop == 512
        assert LatentSync(PROFILES["720p"], Path("/tmp/ls")).face_crop == 256

    def test_deepcache_is_on_by_default(self):
        # Upstream's own inference.sh enables it; leaving it off renders slower
        # for no benefit.
        assert LatentSync(PROFILES["720p"], Path("/tmp/ls")).enable_deepcache is True

    def test_pinned_sha_is_the_commit_phase_0_exercised(self):
        assert PINNED_SHA.startswith("a229c394")


class TestVramEstimate:
    def test_512_at_16_frames_matches_the_measurement(self):
        # Phase 0 measured 13.65 GB on a T4 at these settings.
        provider = LatentSync(PROFILES["1080p"], Path("/tmp/ls"), num_frames=16)
        assert 13.0 <= provider.estimated_vram_gb() <= 14.5

    def test_halving_frames_reduces_the_estimate(self):
        big = LatentSync(PROFILES["1080p"], Path("/tmp/ls"), num_frames=16)
        small = LatentSync(PROFILES["1080p"], Path("/tmp/ls"), num_frames=8)
        assert small.estimated_vram_gb() < big.estimated_vram_gb()

    def test_256_is_cheaper_than_512_at_equal_frames(self):
        big = LatentSync(PROFILES["1080p"], Path("/tmp/ls"), num_frames=16)
        small = LatentSync(PROFILES["720p"], Path("/tmp/ls"), num_frames=16)
        assert small.estimated_vram_gb() < big.estimated_vram_gb()

    def test_256_at_upstream_frames_fits_a_15gb_card(self):
        provider = LatentSync(PROFILES["720p"], Path("/tmp/ls"), num_frames=16)
        assert provider.estimated_vram_gb() < 13.0


class TestWeightResolution:
    """The cache path is what removes 8-12 minutes from every session."""

    def _make_weights(self, directory: Path) -> Path:
        for name in REQUIRED_WEIGHTS:
            target = directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"weights")
        return directory

    def test_recognises_a_complete_weight_set(self, tmp_path):
        assert LatentSync._weights_complete(self._make_weights(tmp_path))

    def test_rejects_a_partial_weight_set(self, tmp_path):
        # A half-finished download must not be treated as usable.
        (tmp_path / "latentsync_unet.pt").write_bytes(b"weights")
        assert not LatentSync._weights_complete(tmp_path)

    def test_rejects_a_missing_directory(self, tmp_path):
        assert not LatentSync._weights_complete(tmp_path / "nope")

    def test_existing_weights_are_used_without_touching_the_cache(self, tmp_path):
        repo = tmp_path / "repo"
        self._make_weights(repo / "checkpoints")
        provider = LatentSync(
            PROFILES["720p"], repo, cache_dir=tmp_path / "does-not-exist"
        )
        assert provider.ensure_weights() == repo / "checkpoints"

    def test_cache_is_linked_when_the_repo_has_none(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        cache = self._make_weights(tmp_path / "cache")

        provider = LatentSync(PROFILES["720p"], repo, cache_dir=cache)
        resolved = provider.ensure_weights()

        assert LatentSync._weights_complete(resolved)
        assert (resolved / "latentsync_unet.pt").read_bytes() == b"weights"

    def test_incomplete_cache_is_not_linked(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        repo.mkdir()
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "latentsync_unet.pt").write_bytes(b"partial")  # whisper missing

        provider = LatentSync(PROFILES["720p"], repo, cache_dir=cache)

        # A fake module rather than monkeypatching the real one: huggingface_hub
        # is not installed on a laptop, and this test must still run there.
        called: dict = {}

        class FakeHub:
            @staticmethod
            def snapshot_download(repo_id, local_dir):
                called["repo_id"] = repo_id
                raise RuntimeError("download attempted")

        monkeypatch.setitem(
            __import__("sys").modules, "huggingface_hub", FakeHub
        )

        # Must fall through to downloading rather than link a partial cache.
        with pytest.raises(RuntimeError, match="download attempted"):
            provider.ensure_weights()
        assert called["repo_id"] == PROFILES["720p"].hf_repo


class TestConfigGeneration:
    def _repo_with_configs(self, tmp_path) -> Path:
        repo = tmp_path / "repo"
        unet = repo / "configs" / "unet"
        unet.mkdir(parents=True)
        for name in ("stage1.yaml", "stage1_512.yaml", "stage2.yaml", "stage2_512.yaml"):
            res = 512 if "512" in name else 256
            (unet / name).write_text(
                f"data:\n  resolution: {res}\n  num_frames: 16\n"
                f"  batch_size: 1\nrun:\n  inference_steps: 20\n"
            )
        return repo

    def test_512_profile_selects_stage2_512(self, tmp_path):
        repo = self._repo_with_configs(tmp_path)
        provider = LatentSync(PROFILES["1080p"], repo)
        written = provider._write_config(repo / "checkpoints")

        import yaml
        cfg = yaml.safe_load(written.read_text())
        assert cfg["data"]["resolution"] == 512

    def test_256_profile_selects_stage2(self, tmp_path):
        repo = self._repo_with_configs(tmp_path)
        provider = LatentSync(PROFILES["720p"], repo)
        written = provider._write_config(repo / "checkpoints")

        import yaml
        cfg = yaml.safe_load(written.read_text())
        assert cfg["data"]["resolution"] == 256

    def test_num_frames_is_written_into_the_config(self, tmp_path):
        repo = self._repo_with_configs(tmp_path)
        provider = LatentSync(PROFILES["1080p"], repo, num_frames=4)
        written = provider._write_config(repo / "checkpoints")

        import yaml
        assert yaml.safe_load(written.read_text())["data"]["num_frames"] == 4

    def test_never_selects_a_stage1_training_config(self, tmp_path):
        # stage1 configs declare the same resolutions but are training stages.
        # Selecting one measures and renders the wrong thing, silently.
        repo = self._repo_with_configs(tmp_path)
        for profile in (PROFILES["1080p"], PROFILES["720p"]):
            written = LatentSync(profile, repo)._write_config(repo / "checkpoints")
            assert "stage1" not in written.name

    def test_the_source_config_is_not_modified(self, tmp_path):
        repo = self._repo_with_configs(tmp_path)
        source = repo / "configs" / "unet" / "stage2_512.yaml"
        before = source.read_text()

        LatentSync(PROFILES["1080p"], repo, num_frames=4)._write_config(
            repo / "checkpoints"
        )
        assert source.read_text() == before

    def test_missing_config_lists_what_is_available(self, tmp_path):
        repo = tmp_path / "repo"
        (repo / "configs" / "unet").mkdir(parents=True)
        (repo / "configs" / "unet" / "stage2_other.yaml").write_text("data: {}\n")

        with pytest.raises(LipsyncError, match="Available stage2 configs"):
            LatentSync(PROFILES["1080p"], repo)._write_config(repo / "checkpoints")


class TestSyncGuards:
    def test_missing_video_is_reported(self, tmp_path):
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"x")
        provider = LatentSync(PROFILES["720p"], tmp_path / "repo")
        with pytest.raises(LipsyncError, match="video input not found"):
            provider.sync(tmp_path / "gone.mp4", audio, tmp_path / "out.mp4")

    def test_missing_audio_is_reported(self, tmp_path):
        video = tmp_path / "v.mp4"
        video.write_bytes(b"x")
        provider = LatentSync(PROFILES["720p"], tmp_path / "repo")
        with pytest.raises(LipsyncError, match="audio input not found"):
            provider.sync(video, tmp_path / "gone.wav", tmp_path / "out.mp4")


class TestFailureClassification:
    """An OOM has a different remedy from every other error, so they must differ."""

    def _provider(self, tmp_path, stderr: str):
        repo = tmp_path / "repo"
        unet = repo / "configs" / "unet"
        unet.mkdir(parents=True)
        (unet / "stage2.yaml").write_text(
            "data:\n  resolution: 256\n  num_frames: 16\n"
        )
        ckpt = repo / "checkpoints"
        for name in REQUIRED_WEIGHTS:
            target = ckpt / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"w")
        (repo / "scripts").mkdir()
        (repo / "scripts" / "inference.py").write_text("")

        provider = LatentSync(PROFILES["720p"], repo)

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 1, "", stderr)

        return provider, fake_run

    def test_oom_names_the_profile_remedy(self, tmp_path, monkeypatch):
        provider, fake_run = self._provider(
            tmp_path, "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate"
        )
        monkeypatch.setattr(subprocess, "run", fake_run)

        video = tmp_path / "v.mp4"; video.write_bytes(b"x")
        audio = tmp_path / "a.wav"; audio.write_bytes(b"x")

        with pytest.raises(LipsyncError, match="TH_PROFILE=720p"):
            provider.sync(video, audio, tmp_path / "out.mp4")

    def test_non_memory_failure_says_profile_will_not_help(self, tmp_path, monkeypatch):
        provider, fake_run = self._provider(
            tmp_path, "ModuleNotFoundError: No module named 'insightface'"
        )
        monkeypatch.setattr(subprocess, "run", fake_run)

        video = tmp_path / "v.mp4"; video.write_bytes(b"x")
        audio = tmp_path / "a.wav"; audio.write_bytes(b"x")

        with pytest.raises(LipsyncError, match="not a memory problem"):
            provider.sync(video, audio, tmp_path / "out.mp4")


class TestLicenceGateStillBites:
    def test_a_rejected_model_cannot_be_constructed(self):
        # Defence against someone adding a Wav2Lip provider later.
        from talkinghead.config import assert_commercially_licensed

        with pytest.raises(LicenseError, match="LRS2"):
            assert_commercially_licensed("wav2lip")
