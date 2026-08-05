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
