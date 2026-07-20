"""Builds the intermediate images shown in the API's "explainability" field
(Build_Specifications_Checklist.md section 4). Only used by app.py, for the one
uploaded video being analyzed — not part of the bulk evaluate.py path, so it's
fine to do a little redundant per-request work here.
"""

import base64
import io

import cv2
import matplotlib

matplotlib.use("Agg")  # no display available in a server process
import matplotlib.pyplot as plt
import numpy as np

import config
from feature_extraction import compute_dft_magnitude, compute_dft_radial_profile


def _encode_image(image):
    ok, buf = cv2.imencode(".png", image)
    return base64.b64encode(buf).decode("ascii")


def _encode_plot(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def build_explainability(original_frame_bgr, face_crop_bgr):
    """original_frame_bgr: the raw sampled frame. face_crop_bgr: the detected
    face region from that same frame, before feature_extraction's own resize."""
    resized = cv2.resize(face_crop_bgr, config.IMAGE_SIZE)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    magnitude_log = compute_dft_magnitude(gray)
    magnitude_display = cv2.normalize(magnitude_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    radial_profile = compute_dft_radial_profile(gray)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(radial_profile)
    ax.set_xlabel("Radial frequency bin")
    ax.set_ylabel("Log magnitude")
    ax.set_title("DFT Radial Profile")

    return {
        "original_frame": _encode_image(original_frame_bgr),
        "face_crop": _encode_image(face_crop_bgr),
        "grayscale": _encode_image(gray),
        "dft_magnitude": _encode_image(magnitude_display),
        "radial_profile_plot": _encode_plot(fig),
    }
