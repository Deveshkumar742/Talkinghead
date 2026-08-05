"""Tests for host detection.

Detection decides where assets are read from and where output is written, so a
wrong answer silently sends a render to the wrong filesystem. The Kaggle and
Colab branches are exercised by faking their environment signals, since CI for
this project is a laptop.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from talkinghead import runtime
from talkinghead.config import load_settings
from talkinghead.runtime import Host, detect


class TestHostDetection:
    def test_local_machine_detected_by_default(self, monkeypatch):
        monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
        monkeypatch.setattr(runtime.Path, "exists", lambda self: False)
        monkeypatch.setitem(sys.modules, "google.colab", None)
        # With google.colab present-but-None the import check still succeeds, so
        # exclude it explicitly to isolate the local branch.
        monkeypatch.delitem(sys.modules, "google.colab", raising=False)
        assert runtime._detect_host() in {Host.LOCAL, Host.COLAB}

    def test_kaggle_detected_from_env_var(self, monkeypatch):
        monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
        assert runtime._detect_host() is Host.KAGGLE

    def test_kaggle_env_var_wins_over_missing_directory(self, monkeypatch):
        # A Kaggle session always sets the env var; the directory check is a
        # belt-and-braces fallback.
        monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
        assert runtime._detect_host() is Host.KAGGLE

    def test_detect_returns_usable_info(self):
        info = detect()
        assert isinstance(info.has_ffmpeg, bool)
        assert isinstance(info.has_gpu, bool)
        assert isinstance(info.work_root, Path)

    def test_is_cloud_is_false_locally(self, monkeypatch):
        monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
        info = detect()
        if info.host is Host.LOCAL:
            assert info.is_cloud is False

    def test_describe_mentions_ffmpeg_state(self):
        assert "ffmpeg" in detect().describe()


class TestKaggleSettings:
    """Path redirection when a Kaggle session is detected."""

    @pytest.fixture
    def fake_kaggle(self, monkeypatch, tmp_path):
        mount = tmp_path / "kaggle" / "input" / "talkinghead-assets"
        mount.mkdir(parents=True)
        (mount / "base_loop.mp4").touch()
        (mount / "reference.wav").touch()
        work = tmp_path / "kaggle" / "working"
        work.mkdir(parents=True)

        monkeypatch.setattr(
            runtime,
            "detect",
            lambda: runtime.HostInfo(
                host=Host.KAGGLE,
                has_ffmpeg=True,
                has_gpu=True,
                input_root=mount.parent,
                work_root=work,
            ),
        )
        # config imported detect by name, so patch it there too.
        import talkinghead.config as config_module

        monkeypatch.setattr(config_module, "detect", runtime.detect)
        return work

    def test_work_and_out_go_to_the_writable_root(self, fake_kaggle):
        settings = load_settings()
        assert settings.work_dir == fake_kaggle / "work"
        assert settings.out_dir == fake_kaggle / "out"

    def test_device_defaults_to_cuda_when_gpu_present(self, fake_kaggle):
        assert load_settings().device == "cuda"

    def test_explicit_device_override_wins(self, fake_kaggle):
        assert load_settings(device="cpu").device == "cpu"

    def test_explicit_path_override_wins_over_host_default(self, fake_kaggle, tmp_path):
        forced = tmp_path / "forced"
        assert load_settings(work_dir=forced).work_dir == forced

    def test_kaggle_mode_reports_true(self, fake_kaggle):
        assert load_settings().kaggle_mode() is True


class TestFfmpegGuidance:
    def test_help_leads_with_kaggle(self):
        # The first mentioned option should be the supported one.
        help_text = runtime.FFMPEG_MISSING_HELP
        assert help_text.index("Kaggle") < help_text.index("winget")

    def test_help_notes_cpu_sessions_are_unmetered(self):
        # This is the reason running assembly in the cloud is free.
        assert "unmetered" in runtime.FFMPEG_MISSING_HELP

    def test_help_states_the_project_does_not_install_ffmpeg(self):
        assert "does not install it" in runtime.FFMPEG_MISSING_HELP
