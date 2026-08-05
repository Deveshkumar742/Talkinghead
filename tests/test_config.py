"""Tests for configuration and the licensing gate.

The gate is the mechanism that stops a non-commercial model reaching client
work. It is cheap to test and expensive to get wrong, so it is tested
thoroughly.
"""

from __future__ import annotations

import pytest

from talkinghead.config import (
    PROFILES,
    REJECTED_MODELS,
    TRUSTED_SOURCES,
    LicenseError,
    Settings,
    assert_commercially_licensed,
    find_asset,
    find_dataset_mount,
    load_settings,
)


class TestLicenseGate:
    @pytest.mark.parametrize("key", sorted(TRUSTED_SOURCES))
    def test_approved_models_pass(self, key):
        assert assert_commercially_licensed(key).name

    @pytest.mark.parametrize("key", sorted(REJECTED_MODELS))
    def test_rejected_models_raise(self, key):
        with pytest.raises(LicenseError):
            assert_commercially_licensed(key)

    def test_wav2lip_rejection_explains_the_dataset_problem(self):
        # Wav2Lip is the single most likely mistake here: it is the default
        # tutorial choice and its restriction comes from the training data, not
        # the code license, so it is easy to miss.
        with pytest.raises(LicenseError, match="LRS2"):
            assert_commercially_licensed("wav2lip")

    def test_codeformer_rejection_points_at_gfpgan(self):
        with pytest.raises(LicenseError, match="GFPGAN"):
            assert_commercially_licensed("codeformer")

    def test_unknown_model_is_refused_by_default(self):
        # Fail closed: an unlisted model is rejected rather than assumed fine.
        with pytest.raises(LicenseError, match="not in TRUSTED_SOURCES"):
            assert_commercially_licensed("some-new-model")

    def test_gate_is_case_insensitive(self):
        assert assert_commercially_licensed("LatentSync")
        with pytest.raises(LicenseError):
            assert_commercially_licensed("Wav2Lip")

    def test_gate_tolerates_surrounding_whitespace(self):
        assert assert_commercially_licensed("  chatterbox  ")

    def test_rejected_and_trusted_sets_do_not_overlap(self):
        assert not set(REJECTED_MODELS) & set(TRUSTED_SOURCES)


class TestTrustedSources:
    @pytest.mark.parametrize("key,src", sorted(TRUSTED_SOURCES.items()))
    def test_every_source_is_first_party_github(self, key, src):
        # Guards against a mirror or re-upload creeping in.
        assert src.repo.startswith("https://github.com/"), src.repo

    @pytest.mark.parametrize("key,src", sorted(TRUSTED_SOURCES.items()))
    def test_every_source_declares_a_permissive_license(self, key, src):
        assert src.license in {"MIT", "Apache-2.0"}, f"{key}: {src.license}"

    @pytest.mark.parametrize("key,src", sorted(TRUSTED_SOURCES.items()))
    def test_every_source_is_pinned(self, key, src):
        assert src.revision, f"{key} has no pinned revision"


class TestFindDatasetMount:
    """Kaggle has more than one mount layout, confirmed the hard way.

    A real session mounted the dataset at
    ``/kaggle/input/datasets/deveshkumar742/talkinghead-assets/`` rather than the
    documented ``/kaggle/input/talkinghead-assets/``. Checking only the classic
    path silently found nothing.
    """

    def test_finds_classic_layout(self, tmp_path):
        (tmp_path / "talkinghead-assets").mkdir()
        found = find_dataset_mount(tmp_path, "talkinghead-assets")
        assert found == tmp_path / "talkinghead-assets"

    def test_finds_owner_nested_layout(self, tmp_path):
        nested = tmp_path / "datasets" / "deveshkumar742" / "talkinghead-assets"
        nested.mkdir(parents=True)
        assert find_dataset_mount(tmp_path, "talkinghead-assets") == nested

    def test_prefers_classic_over_nested(self, tmp_path):
        classic = tmp_path / "talkinghead-assets"
        classic.mkdir()
        (tmp_path / "datasets" / "someone" / "talkinghead-assets").mkdir(parents=True)
        assert find_dataset_mount(tmp_path, "talkinghead-assets") == classic

    def test_returns_none_when_absent(self, tmp_path):
        (tmp_path / "unrelated-dataset").mkdir()
        assert find_dataset_mount(tmp_path, "talkinghead-assets") is None

    def test_returns_none_when_input_root_missing(self, tmp_path):
        assert find_dataset_mount(tmp_path / "nope", "talkinghead-assets") is None

    def test_ignores_a_file_with_the_slug_name(self, tmp_path):
        (tmp_path / "talkinghead-assets").write_text("not a directory")
        assert find_dataset_mount(tmp_path, "talkinghead-assets") is None

    def test_does_not_search_unbounded_depth(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c" / "talkinghead-assets"
        deep.mkdir(parents=True)
        assert find_dataset_mount(tmp_path, "talkinghead-assets") is None


class TestFindAsset:
    """Kaggle preserves upload structure, so assets are often nested one level.

    A dataset that looks correctly uploaded in Kaggle's file browser can still
    have its files a directory down -- from a ZIP, or from dragging a folder in.
    The resulting "base_loop.mp4 not found" is baffling when you can see the file
    on screen, so the lookup tolerates it.
    """

    def test_finds_asset_at_mount_root(self, tmp_path):
        (tmp_path / "base_loop.mp4").write_bytes(b"x")
        assert find_asset(tmp_path, "base_loop.mp4") == tmp_path / "base_loop.mp4"

    def test_finds_asset_nested_one_level(self, tmp_path):
        nested = tmp_path / "assets"
        nested.mkdir()
        (nested / "base_loop.mp4").write_bytes(b"x")
        assert find_asset(tmp_path, "base_loop.mp4") == nested / "base_loop.mp4"

    def test_prefers_root_over_nested(self, tmp_path):
        (tmp_path / "reference.wav").write_bytes(b"x")
        nested = tmp_path / "sub"
        nested.mkdir()
        (nested / "reference.wav").write_bytes(b"y")
        assert find_asset(tmp_path, "reference.wav") == tmp_path / "reference.wav"

    def test_falls_back_to_root_path_when_absent(self, tmp_path):
        # Returns a sensible path so callers can report what was missing.
        assert find_asset(tmp_path, "nope.mp4") == tmp_path / "nope.mp4"

    def test_falls_back_when_mount_does_not_exist(self, tmp_path):
        ghost = tmp_path / "not-mounted"
        assert find_asset(ghost, "base_loop.mp4") == ghost / "base_loop.mp4"

    def test_ignores_a_directory_with_the_target_name(self, tmp_path):
        (tmp_path / "base_loop.mp4").mkdir()
        # A directory is not an asset; fall through rather than returning it.
        assert find_asset(tmp_path, "base_loop.mp4") == tmp_path / "base_loop.mp4"
        assert not find_asset(tmp_path, "base_loop.mp4").is_file()

    def test_does_not_search_arbitrarily_deep(self, tmp_path):
        # Bounded depth: an unrelated deep match should not be silently adopted.
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        (deep / "reference.wav").write_bytes(b"x")
        assert find_asset(tmp_path, "reference.wav") == tmp_path / "reference.wav"


class TestProfiles:
    def test_both_profiles_exist(self):
        assert set(PROFILES) == {"1080p", "720p"}

    def test_1080p_uses_the_512_crop(self):
        # This is what makes 1080p viable -- LatentSync 1.6 retrained at 512.
        assert PROFILES["1080p"].face_crop == 512
        assert PROFILES["1080p"].output_height == 1080

    def test_720p_fallback_uses_the_256_crop(self):
        assert PROFILES["720p"].face_crop == 256
        assert PROFILES["720p"].output_height == 720

    def test_each_crop_size_names_its_matching_checkpoint(self):
        # 1.6 was trained at 512 and 1.5 at 256; pairing a checkpoint with the
        # wrong resolution is a silent mismatch rather than an error, so the
        # association is pinned here.
        assert PROFILES["1080p"].hf_repo == "ByteDance/LatentSync-1.6"
        assert PROFILES["720p"].hf_repo == "ByteDance/LatentSync-1.5"

    def test_1080p_reduces_num_frames_below_upstream_default(self):
        # Phase 0 measured the upstream num_frames=16 at 13.65GB peak, OOMing on
        # a 14.56GiB T4. Reverting this to 16 reintroduces that failure.
        assert PROFILES["1080p"].num_frames == 8

    def test_720p_keeps_the_upstream_num_frames(self):
        # A quarter the pixels per frame, so 16 fits comfortably.
        assert PROFILES["720p"].num_frames == 16

    @pytest.mark.parametrize("name,profile", sorted(PROFILES.items()))
    def test_num_frames_is_positive(self, name, profile):
        assert profile.num_frames > 0, name

    @pytest.mark.parametrize("name,profile", sorted(PROFILES.items()))
    def test_output_is_16_by_9(self, name, profile):
        ratio = profile.output_width / profile.output_height
        assert abs(ratio - 16 / 9) < 0.01, f"{name} is not 16:9"


class TestSettings:
    def test_defaults_to_1080p_on_cpu(self):
        settings = Settings()
        assert settings.profile == "1080p"
        assert settings.device == "cpu"

    def test_active_profile_resolves(self):
        assert Settings(profile="720p").active_profile.output_height == 720

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError, match="Unknown profile"):
            _ = Settings(profile="4k").active_profile

    def test_overrides_apply(self):
        assert load_settings(device="cuda", profile="720p").device == "cuda"

    def test_segment_cap_is_bounded(self):
        # An unbounded cap would defeat the point of chunking.
        with pytest.raises(ValueError):
            Settings(max_chars_per_segment=5000)

    def test_crf_is_bounded(self):
        with pytest.raises(ValueError):
            Settings(video_crf=99)

    def test_env_vars_are_read_with_prefix(self, monkeypatch):
        monkeypatch.setenv("TH_PROFILE", "720p")
        monkeypatch.setenv("TH_DEVICE", "cuda")
        settings = Settings()
        assert settings.profile == "720p"
        assert settings.device == "cuda"
