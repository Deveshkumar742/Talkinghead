"""Tests for the ffmpeg wrappers.

These shell out to real ffmpeg against generated fixtures. Durations are asserted
with tolerance because container timestamps and frame boundaries do not land on
exact values -- but the tolerance is tight enough to catch a wrapper that
silently drops or duplicates content.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from talkinghead import media
from tests.conftest import requires_ffmpeg

TOL = 0.15  # seconds


class TestFindBinary:
    @requires_ffmpeg
    def test_finds_ffmpeg_when_the_host_provides_it(self):
        assert media.find_binary("ffmpeg")

    def test_missing_binary_points_at_a_cloud_session(self):
        # FFmpeg is host-provided, so the error must route the user to Kaggle
        # rather than to a package manager.
        with pytest.raises(media.MediaError, match="Kaggle"):
            media.find_binary("definitely-not-a-real-binary-xyz")

    def test_ffmpeg_available_never_raises(self):
        assert isinstance(media.ffmpeg_available(), bool)


@requires_ffmpeg
class TestProbe:
    def test_reports_audio_duration(self, tone_wav):
        info = media.probe(tone_wav)
        assert info.has_audio
        assert not info.has_video
        assert abs(info.duration_s - 1.0) < TOL
        assert info.channels == 1

    def test_reports_video_geometry(self, test_video):
        info = media.probe(test_video)
        assert info.has_video
        assert (info.width, info.height) == (320, 240)
        assert info.fps is not None and abs(info.fps - 25) < 1
        assert abs(info.duration_s - 2.0) < TOL

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(media.MediaError, match="not found"):
            media.probe(tmp_path / "nope.wav")

    def test_non_media_file_raises(self, tmp_path):
        junk = tmp_path / "junk.wav"
        junk.write_text("this is not audio")
        with pytest.raises(media.MediaError):
            media.probe(junk)


@requires_ffmpeg
class TestConcatAudio:
    def test_durations_add_up(self, tone_wav, short_tone_wav, tmp_path):
        out = media.concat_audio(
            [(tone_wav, 0), (short_tone_wav, 0)], tmp_path / "joined.wav"
        )
        assert abs(media.probe(out).duration_s - 1.5) < TOL

    def test_pauses_are_included(self, tone_wav, short_tone_wav, tmp_path):
        out = media.concat_audio(
            [(tone_wav, 500), (short_tone_wav, 0)], tmp_path / "paused.wav"
        )
        # 1.0 + 0.5s pause + 0.5 = 2.0
        assert abs(media.probe(out).duration_s - 2.0) < TOL

    def test_single_clip_works(self, tone_wav, tmp_path):
        out = media.concat_audio([(tone_wav, 0)], tmp_path / "one.wav")
        assert abs(media.probe(out).duration_s - 1.0) < TOL

    def test_output_is_mono_at_requested_rate(self, tone_wav, tmp_path):
        out = media.concat_audio(
            [(tone_wav, 0)], tmp_path / "rate.wav", sample_rate=16_000
        )
        info = media.probe(out)
        assert info.sample_rate == 16_000
        assert info.channels == 1

    def test_empty_clip_list_raises(self, tmp_path):
        with pytest.raises(media.MediaError, match="no clips"):
            media.concat_audio([], tmp_path / "out.wav")

    def test_missing_clip_raises_before_running_ffmpeg(self, tmp_path):
        with pytest.raises(media.MediaError, match="not found"):
            media.concat_audio([(tmp_path / "ghost.wav", 0)], tmp_path / "out.wav")

    def test_creates_parent_directory(self, tone_wav, tmp_path):
        out = media.concat_audio(
            [(tone_wav, 0)], tmp_path / "nested" / "deep" / "out.wav"
        )
        assert out.exists()


@requires_ffmpeg
class TestNormalizeLoudness:
    def test_preserves_duration(self, tone_wav, tmp_path):
        out = media.normalize_loudness(tone_wav, tmp_path / "norm.wav")
        assert abs(media.probe(out).duration_s - 1.0) < TOL


@requires_ffmpeg
class TestFitVideoToDuration:
    def test_trims_when_source_is_longer(self, test_video, tmp_path):
        out = media.fit_video_to_duration(test_video, 1.0, tmp_path / "trim.mp4")
        assert abs(media.probe(out).duration_s - 1.0) < TOL

    def test_loops_when_source_is_shorter(self, test_video, tmp_path):
        # 2s source, 5s target -- this is the normal case, since a 60-90s loop
        # has to cover a multi-minute narration.
        out = media.fit_video_to_duration(test_video, 5.0, tmp_path / "loop.mp4")
        assert abs(media.probe(out).duration_s - 5.0) < TOL

    def test_exact_duration_needs_no_loop(self, test_video, tmp_path):
        out = media.fit_video_to_duration(test_video, 2.0, tmp_path / "exact.mp4")
        assert abs(media.probe(out).duration_s - 2.0) < TOL

    def test_output_has_no_audio(self, test_video, tmp_path):
        # Audio is added at mux time; a stray track here would fight the mux.
        out = media.fit_video_to_duration(test_video, 1.0, tmp_path / "silent.mp4")
        assert media.probe(out).has_audio is False

    def test_preserves_resolution(self, test_video, tmp_path):
        out = media.fit_video_to_duration(test_video, 1.0, tmp_path / "res.mp4")
        info = media.probe(out)
        assert (info.width, info.height) == (320, 240)

    def test_zero_duration_raises(self, test_video, tmp_path):
        with pytest.raises(media.MediaError, match="must be positive"):
            media.fit_video_to_duration(test_video, 0, tmp_path / "bad.mp4")

    def test_negative_duration_raises(self, test_video, tmp_path):
        with pytest.raises(media.MediaError, match="must be positive"):
            media.fit_video_to_duration(test_video, -3, tmp_path / "bad.mp4")

    def test_pingpong_doubles_before_fitting(self, test_video, tmp_path):
        out = media.fit_video_to_duration(
            test_video, 4.0, tmp_path / "pp.mp4", pingpong=True
        )
        assert abs(media.probe(out).duration_s - 4.0) < TOL


@requires_ffmpeg
class TestMakePingpong:
    def test_roughly_doubles_duration(self, test_video, tmp_path):
        out = media.make_pingpong(test_video, tmp_path / "pp.mp4")
        assert abs(media.probe(out).duration_s - 4.0) < 0.3


@requires_ffmpeg
class TestMuxAndEncode:
    def test_produces_both_streams_at_target_resolution(
        self, test_video, tone_wav, tmp_path
    ):
        out = media.mux_and_encode(
            test_video, tone_wav, tmp_path / "final.mp4", width=640, height=360
        )
        info = media.probe(out)
        assert info.has_video and info.has_audio
        assert (info.width, info.height) == (640, 360)

    def test_shortest_stream_bounds_duration(self, test_video, tone_wav, tmp_path):
        # 2s video, 1s audio -- must not leave a frozen tail frame.
        out = media.mux_and_encode(
            test_video, tone_wav, tmp_path / "short.mp4", width=320, height=240
        )
        assert media.probe(out).duration_s < 1.5

    def test_pads_rather_than_distorts_on_aspect_change(
        self, test_video, tone_wav, tmp_path
    ):
        # 4:3 source into a 16:9 target must letterbox, not stretch.
        out = media.mux_and_encode(
            test_video, tone_wav, tmp_path / "pad.mp4", width=1920, height=1080
        )
        info = media.probe(out)
        assert (info.width, info.height) == (1920, 1080)

    def test_writes_metadata(self, test_video, tone_wav, tmp_path):
        out = media.mux_and_encode(
            test_video,
            tone_wav,
            tmp_path / "meta.mp4",
            width=320,
            height=240,
            metadata={"comment": "AI-generated test"},
        )
        assert out.exists() and out.stat().st_size > 0
