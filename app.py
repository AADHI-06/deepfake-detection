"""Flask API (documentation section 8; Build_Specifications_Checklist.md
sections 4-6): a single synchronous /analyze endpoint. Synchronous, not a job
queue — the build spec calls this out explicitly as the right call for this
project's scale, so there's no polling/job-status endpoint to build.
"""

import os
import uuid

import cv2
from flask import Flask, jsonify, render_template, request

import config
from explainability import build_explainability
from face_utils import detect_and_filter_face
from inference import LABEL_NAMES, load_pipeline, predict_video

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_BYTES

pipeline = load_pipeline()
os.makedirs(config.UPLOAD_DIR, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


def error_response(code, message, status=400):
    return jsonify({"error": code, "message": message}), status


@app.errorhandler(413)
def handle_too_large(_e):
    return error_response("file_too_large", "Video exceeds the 500MB upload limit.", 413)


def allowed_file(filename):
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in config.ALLOWED_VIDEO_EXTENSIONS


def build_frame_results(frame_results):
    """Reshape inference.py's internal frame dicts into the API contract shape.
    Discarded frames (no face survived detection/quality filtering) never went
    through the classifier, so their prediction/confidence are genuinely
    unknown — reported as null rather than invented."""
    out = []
    for fr in frame_results:
        entry = {"frame_index": fr["frame_index"], "used_in_vote": fr["used_in_vote"]}
        if fr["used_in_vote"]:
            prob_fake = fr["probability_fake"]
            pred_label = 1 if prob_fake >= config.VOTE_THRESHOLD else 0
            entry["prediction"] = LABEL_NAMES[pred_label]
            entry["confidence"] = prob_fake if pred_label == 1 else 1 - prob_fake
        else:
            entry["prediction"] = None
            entry["confidence"] = None
        out.append(entry)
    return out


def build_first_frame_explainability(video_path, frame_results):
    """Re-locates the first frame that made it into the vote and regenerates
    its face crop, purely for the explainability visuals — predict_video()
    doesn't keep pixel data around since evaluate.py calls it in bulk."""
    first_used_idx = next((fr["frame_index"] for fr in frame_results if fr["used_in_vote"]), None)
    if first_used_idx is None:
        return None

    cap = cv2.VideoCapture(video_path)
    idx = 0
    frame = None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if idx == first_used_idx:
            frame = f
            break
        idx += 1
    cap.release()

    face = detect_and_filter_face(frame)
    return build_explainability(frame, face)


@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files or request.files["video"].filename == "":
        return error_response("missing_file", "No 'video' file provided.")

    file = request.files["video"]
    if not allowed_file(file.filename):
        return error_response(
            "unsupported_format",
            f"Unsupported video format. Allowed: {', '.join(sorted(config.ALLOWED_VIDEO_EXTENSIONS))}.",
        )

    ext = file.filename.rsplit(".", 1)[-1].lower()
    temp_path = os.path.join(config.UPLOAD_DIR, f"{uuid.uuid4().hex}.{ext}")
    file.save(temp_path)

    try:
        cap = cv2.VideoCapture(temp_path)
        is_valid = cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
        cap.release()
        if not is_valid:
            return error_response("invalid_video", "Could not read this video — it may be corrupted or empty.")

        label, confidence, frame_results = predict_video(temp_path, pipeline, config.FRAMES_PER_VIDEO_TEST)

        if label is None:
            return error_response("no_face_detected", "No usable face was found in any sampled frame.")

        return jsonify({
            "prediction": LABEL_NAMES[label],
            "confidence": confidence,
            "frame_results": build_frame_results(frame_results),
            "frames_discarded": sum(1 for fr in frame_results if not fr["used_in_vote"]),
            "explainability": build_first_frame_explainability(temp_path, frame_results),
        })
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    app.run(debug=False, port=5000)
