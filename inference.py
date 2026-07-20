"""Video-level inference (documentation section 8): sample frames -> detect and
quality-filter faces -> extract fused features -> predict per-frame probability
-> confidence-weighted aggregation into one video-level label.

Shared by evaluate.py (offline test-set scoring) and app.py (live API), so both
use the exact same prediction logic.
"""

import cv2
import joblib

import config
from face_utils import detect_and_filter_face, sample_frame_indices
from feature_extraction import extract_features

LABEL_REAL = 0
LABEL_FAKE = 1
LABEL_NAMES = {LABEL_REAL: "REAL", LABEL_FAKE: "FAKE"}


def load_pipeline(path=config.MODEL_OUTPUT_PATH):
    return joblib.load(path)


def predict_video(video_path, pipeline, n_frames=config.FRAMES_PER_VIDEO_TEST):
    """Returns (label, confidence, frame_results).

    frame_results is a list of one dict per sampled frame:
      {"frame_index": int, "used_in_vote": bool, "probability_fake": float}
    ("probability_fake" is omitted for frames dropped by face/quality filtering.)

    If no sampled frame yields a usable face, label and confidence are both None
    — callers must handle this as the "no face detected" case, not a crash.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted_indices = sample_frame_indices(total, n_frames)

    frame_results = []
    fake_probs = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted_indices:
            face = detect_and_filter_face(frame)
            if face is None:
                frame_results.append({"frame_index": idx, "used_in_vote": False})
            else:
                feats = extract_features(face).reshape(1, -1)
                prob_fake = float(pipeline.predict_proba(feats)[0][LABEL_FAKE])
                fake_probs.append(prob_fake)
                frame_results.append({
                    "frame_index": idx,
                    "used_in_vote": True,
                    "probability_fake": prob_fake,
                })
        idx += 1
    cap.release()

    if not fake_probs:
        return None, None, frame_results

    # Confidence-weighted voting (section 0 fix #3): average per-frame P(fake)
    # rather than a flat majority count of hard REAL/FAKE votes.
    avg_prob_fake = sum(fake_probs) / len(fake_probs)
    label = LABEL_FAKE if avg_prob_fake >= config.VOTE_THRESHOLD else LABEL_REAL
    confidence = avg_prob_fake if label == LABEL_FAKE else 1 - avg_prob_fake
    return label, confidence, frame_results
