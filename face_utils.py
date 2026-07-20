"""Frame sampling + face detection/quality-filtering, shared by train.py
(feature extraction) and inference.py (prediction) so both use identical
preprocessing."""

import cv2
import numpy as np
from facenet_pytorch import MTCNN

import config

# facenet-pytorch defaults to GPU if available; this project assumes CPU-only
# (see Build_Specifications_Checklist.md section 1)
_mtcnn = MTCNN(keep_all=False, device="cpu")


def sample_frame_indices(total_frames, n_frames):
    """Uniform temporal sampling; if the video is short, take every frame it has
    (documentation section 7 step 4: sample fewer, don't fail)."""
    if total_frames <= n_frames:
        return set(range(total_frames))
    return set(np.linspace(0, total_frames - 1, n_frames, dtype=int).tolist())


def laplacian_variance(gray_image):
    return cv2.Laplacian(gray_image, cv2.CV_64F).var()


def detect_and_filter_face(frame_bgr):
    """MTCNN detect -> discard low-confidence or blurry crops -> return the face
    crop (still full color, resizing/grayscale happens in feature_extraction)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    boxes, probs = _mtcnn.detect(rgb)
    if boxes is None or probs[0] is None or probs[0] < config.FACE_CONFIDENCE_THRESHOLD:
        return None

    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = boxes[0]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    face = frame_bgr[y1:y2, x1:x2]
    if face.size == 0:
        return None

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    if laplacian_variance(gray) < config.BLUR_THRESHOLD:
        return None

    return face
