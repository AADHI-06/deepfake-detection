"""Handcrafted feature extraction: DFT radial profile + LBP texture + noise residual stats.

Each branch operates on a grayscale face crop resized to config.IMAGE_SIZE.
The three branches are concatenated into a single fused feature vector (section 6.4).
"""

import cv2
import numpy as np
from scipy.stats import kurtosis, skew
from skimage.feature import local_binary_pattern

import config


def compute_dft_magnitude(gray_image):
    """DFT -> shift -> log magnitude spectrum. Split out from the radial profile
    below so app.py's explainability view can reuse it for a visual, not just
    the reduced radial vector."""
    f = np.fft.fft2(gray_image.astype(np.float64))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    return np.log1p(magnitude)  # log1p avoids log(0) and compresses dynamic range


def compute_dft_radial_profile(gray_image, n_bins=config.DFT_RADIAL_BINS):
    """log magnitude spectrum -> radially-averaged profile.

    Radial averaging collapses the 2D spectrum into a 1D fingerprint of how
    energy is distributed across frequencies, which is what differs between
    real camera images and GAN-upsampled ones (section 6.1).
    """
    h, w = gray_image.shape
    magnitude_log = compute_dft_magnitude(gray_image)

    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_r = r.max()

    bin_edges = np.linspace(0, max_r, n_bins + 1)
    bin_indices = np.clip(np.digitize(r.ravel(), bin_edges) - 1, 0, n_bins - 1)

    sums = np.bincount(bin_indices, weights=magnitude_log.ravel(), minlength=n_bins)
    counts = np.bincount(bin_indices, minlength=n_bins)
    radial_profile = sums / np.maximum(counts, 1)  # avoid divide-by-zero for empty bins

    return radial_profile


def compute_lbp_histogram(gray_image):
    """Uniform LBP -> normalized histogram of pattern labels (section 6.2)."""
    lbp = local_binary_pattern(
        gray_image, P=config.LBP_POINTS, R=config.LBP_RADIUS, method=config.LBP_METHOD
    )
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=config.LBP_N_BINS,
        range=(0, config.LBP_N_BINS),
        density=True,
    )
    return hist


def compute_noise_residual_stats(gray_image):
    """High-pass residual (image minus its own Gaussian-blurred denoise) -> summary stats.

    Real sensor noise and GAN output noise differ in this residual's statistics
    (section 6.3) — a lightweight stand-in for full PRNU analysis.
    """
    gray_f = gray_image.astype(np.float64)
    denoised = cv2.GaussianBlur(gray_image, (3, 3), 0).astype(np.float64)
    residual = gray_f - denoised
    flat = residual.ravel()

    return np.array([
        flat.mean(),
        flat.std(),
        skew(flat),
        kurtosis(flat),
        np.mean(flat ** 2),  # energy
    ])


def extract_features(face_image_bgr):
    """Fuse all three branches into one feature vector for a single face crop."""
    resized = cv2.resize(face_image_bgr, config.IMAGE_SIZE)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    dft_features = compute_dft_radial_profile(gray)
    lbp_features = compute_lbp_histogram(gray)
    residual_features = compute_noise_residual_stats(gray)

    return np.concatenate([dft_features, lbp_features, residual_features])
