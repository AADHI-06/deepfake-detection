# Build Specifications — Things to Nail Down Before Claude Code Starts

> The improved documentation says *what* the pipeline does. This file says *how it should be built*, so Claude Code doesn't fill gaps with guesses.

---

## 1. Environment & Dependencies

- **Python version:** 3.10 (pin it — MTCNN/facenet-pytorch and scikit-learn versions matter)
- **Pin exact versions** in `requirements.txt`, not just package names. Unpinned versions are how "works on my machine" bugs happen.
- **Hardware assumption:** CPU-only. State this explicitly, or Claude Code may write CUDA-dependent code by default (facenet-pytorch will try GPU if available — force CPU or make it configurable).
- Use a virtual environment (`venv`), documented in a `SETUP.md` or README section.

## 2. Dataset Handling

- FaceForensics++ requires a license application — **it will already be downloaded locally**, not fetched by code. Specify the exact local path convention, e.g.:
  ```
  dataset/
      original_sequences/c23/videos/
      manipulated_sequences/DeepFakes/c23/videos/
  ```
- State this directly to Claude Code: *"Do not write any dataset download/scraping logic. Assume videos already exist at the paths above."*
- Specify how the video-level train/val/test split is persisted — e.g., a `splits.json` listing which video filenames belong to which split, generated once and reused (so re-running training doesn't reshuffle and silently create leakage).

## 3. Configuration, Not Hardcoding

- All paths, thresholds, and hyperparameters (frame count, image size, face-confidence threshold, PCA variance target, RF grid) go in a single `config.py` or `config.yaml` — not scattered as magic numbers across files.
- Random seed (42) set once, centrally, and reused everywhere (numpy, sklearn, torch) for reproducibility.

## 4. API Contract (`/analyze` endpoint)

Specify this explicitly or Claude Code will improvise a schema:

**Request:** multipart/form-data, single video file field named `video`

**Response (JSON):**
```json
{
  "prediction": "FAKE",
  "confidence": 0.82,
  "frame_results": [
    {"frame_index": 0, "prediction": "FAKE", "confidence": 0.91, "used_in_vote": true},
    {"frame_index": 1, "prediction": "REAL", "confidence": 0.55, "used_in_vote": false}
  ],
  "frames_discarded": 2,
  "explainability": {
    "original_frame": "base64...",
    "face_crop": "base64...",
    "grayscale": "base64...",
    "dft_magnitude": "base64...",
    "radial_profile_plot": "base64..."
  }
}
```
- Decide: images returned as base64 inline, or saved to `/static/results/<job_id>/` and referenced by URL? (Base64 is simpler for a small project; URL is cleaner for larger ones. For your scale — base64 is fine.)

## 5. Error Handling — specify expected behavior for each:

- No face detected in **any** sampled frame → return a clear error response (`"error": "no_face_detected"`), not a crash or silent misprediction.
- Video shorter than expected frame count → sample fewer frames, don't fail.
- Corrupted/unreadable video file → return `"error": "invalid_video"`.
- Unsupported format or file >500MB → reject before processing, not after upload completes.
- All error responses should have consistent shape: `{"error": "<code>", "message": "<human-readable>"}`.

## 6. Frontend Behavior

- Is the analyze call synchronous (browser waits, spinner shown) or does it poll a job-status endpoint? **For this project size: synchronous is simpler and defensible — say so explicitly** so Claude Code doesn't build unnecessary job-queue infrastructure.
- Progress screen: since there's no real progress events from a synchronous call, either (a) show an indeterminate spinner with pipeline-stage labels updated via a fake timed sequence, or (b) keep it honest as just "Processing..." — decide which, don't leave it ambiguous.

## 7. Testing

- At least one test video (real, small, in `tests/sample_videos/`) so Claude Code can verify the inference path works end-to-end without needing the full dataset.
- Unit tests for feature extraction functions (DFT profile, LBP histogram, residual stats) with a fixed dummy image — checks shapes/ranges, not full integration.

## 8. Build Order (give this to Claude Code explicitly — don't let it one-shot everything)

Build and get working, in this order, confirming each before moving to the next:

1. `feature_extraction.py` — all three feature branches, unit-tested on a dummy image
2. `data_split.py` — video-level split logic, writes `splits.json`
3. `train.py` — training pipeline through to a saved joblib model (no API yet)
4. Model evaluation script — confusion matrix, ROC-AUC, CV scores, error analysis output
5. `app.py` — Flask backend, `/analyze` endpoint, using the saved model
6. Frontend (`templates/index.html`, static JS/CSS)
7. Wire frontend to backend, end-to-end manual test

## 9. Code Style Constraints (say this explicitly, it won't happen by default)

- No unnecessary abstraction layers, no premature class hierarchies for a project this size — plain functions and a couple of small classes where genuinely useful.
- Comments should explain *why* (e.g., "video-level split to prevent leakage"), not restate *what* the line does.
- No extra frameworks beyond what's in the stack (no adding Celery, Redis, Docker, etc. unless you explicitly ask later).
- Commit after each numbered build-order step, not one giant commit at the end.

## 10. Acceptance Criteria (definition of done)

- Training script runs end-to-end and prints CV mean ± std, not just a single accuracy number.
- Evaluation script outputs a confusion matrix image and a written error-analysis text file listing 5+ misclassified examples with a plausible reason each.
- `/analyze` endpoint returns correctly shaped JSON for: a real video, a fake video, a video with no detectable face, and a corrupted file.
- You (Aadhi) can explain every function in the codebase without needing Claude Code to explain it back to you first.
