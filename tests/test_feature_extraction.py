"""Unit tests for feature_extraction.py against fixed dummy images.

Checks shapes/ranges only — not real face data, no integration with the dataset.
"""

import numpy as np
import pytest

import config
from feature_extraction import (
    compute_dft_radial_profile,
    compute_lbp_histogram,
    compute_noise_residual_stats,
    extract_features,
)

EXPECTED_LEN = config.DFT_RADIAL_BINS + config.LBP_N_BINS + 5


@pytest.fixture
def flat_gray_image():
    # Degenerate case: constant image -> zero noise residual, minimal texture variation
    return np.full((128, 128), 128, dtype=np.uint8)


@pytest.fixture
def random_gray_image():
    rng = np.random.default_rng(config.RANDOM_SEED)
    return rng.integers(0, 256, size=(128, 128), dtype=np.uint8)


@pytest.fixture
def random_bgr_face():
    rng = np.random.default_rng(config.RANDOM_SEED)
    return rng.integers(0, 256, size=(200, 180, 3), dtype=np.uint8)


def test_dft_radial_profile_shape_and_range(random_gray_image):
    profile = compute_dft_radial_profile(random_gray_image)
    assert profile.shape == (config.DFT_RADIAL_BINS,)
    assert np.all(np.isfinite(profile))
    assert np.all(profile >= 0)  # log1p of a magnitude is always non-negative


def test_lbp_histogram_shape_and_normalization(random_gray_image):
    hist = compute_lbp_histogram(random_gray_image)
    assert hist.shape == (config.LBP_N_BINS,)
    assert np.all(np.isfinite(hist))
    assert np.all(hist >= 0)
    # density=True histogram integrates to ~1 over the bin width (bin width = 1 here)
    assert hist.sum() == pytest.approx(1.0, abs=1e-6)


def test_lbp_histogram_flat_image_is_degenerate(flat_gray_image):
    # A constant image has zero local variation -> LBP is dominated by a single
    # pattern (border-pixel edge handling in skimage puts a small remainder elsewhere)
    hist = compute_lbp_histogram(flat_gray_image)
    assert hist.max() > 0.9


def test_noise_residual_stats_shape_and_finite(random_gray_image):
    stats = compute_noise_residual_stats(random_gray_image)
    assert stats.shape == (5,)
    assert np.all(np.isfinite(stats))
    mean, std, _, _, energy = stats
    assert std >= 0
    assert energy >= 0


def test_noise_residual_stats_flat_image_is_zero(flat_gray_image):
    # Constant image -> Gaussian blur changes nothing -> residual is exactly zero
    stats = compute_noise_residual_stats(flat_gray_image)
    mean, std, skewness, kurt, energy = stats
    assert mean == pytest.approx(0.0, abs=1e-9)
    assert std == pytest.approx(0.0, abs=1e-9)
    assert energy == pytest.approx(0.0, abs=1e-9)


def test_extract_features_fused_shape_and_finite(random_bgr_face):
    features = extract_features(random_bgr_face)
    assert features.shape == (EXPECTED_LEN,)
    assert np.all(np.isfinite(features))


def test_extract_features_resizes_arbitrary_input(random_bgr_face):
    # Different input dims should still produce the same fixed-length output
    small = random_bgr_face[:50, :60]
    features = extract_features(small)
    assert features.shape == (EXPECTED_LEN,)
