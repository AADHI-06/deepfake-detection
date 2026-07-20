# CLAUDE.md

This file is read automatically by Claude Code at the start of every session in this project. Follow it without being reminded.

## Project

Deepfake video detector using frequency-domain (DFT radial profile) + texture (LBP) + noise-residual handcrafted features, fused and classified with classical ML (Random Forest, tuned via GridSearchCV + StratifiedKFold CV). Flask backend, vanilla HTML/CSS/JS frontend. Dataset: FaceForensics++ (C23), already present locally — never write dataset download/scraping code.

**Actual dataset layout on disk** (differs from the original doc's `original_sequences/manipulated_sequences` convention — confirmed with user, encoded in `config.py`):
```
dataset/training/real/       (200 videos)
dataset/training/deepfake/   (200 videos)
dataset/testing/real/        (50 videos)
dataset/testing/deepfake/    (50 videos)
```
**These folder names no longer indicate train/val/test membership.** They were the original physical layout, but that layout turned out to leak identity across splits (see step 4 in the Build Progress Log) — `data_split.py` now pools all 500 videos from both folders and re-splits by identity group. `splits.json` is the only source of truth for which split a video belongs to; a path under `dataset/training/...` may legitimately be assigned to `test` in `splits.json`.

Full spec: `Deepfake_Detection_Improved_Documentation.md`
Build constraints: `Build_Specifications_Checklist.md`
Read both before writing any code.

## Non-negotiable methodology rules

- **Video-level train/val/test split, always, before frame sampling.** Never let frames from the same source video appear in more than one split. This is the single most important correctness rule in this project — if you're ever unsure whether a change risks leakage, stop and ask rather than assume.
- **Video-level is not enough — this dataset requires identity-level splitting.** The same actor's face appears across multiple *different* video files (different scenario clips, different filenames), so a filename-disjoint split can still leak identity. Fake videos are grouped by source actor (parsed from the `{source}_{target}__scenario__hash.mp4` filename — the source actor's face is what's swapped into frame, not the target's). Real videos have no identity in the filename, so they're grouped by face-embedding cluster (`identity_grouping.py`, cosine similarity > 0.5). If a change touches `data_split.py` or adds a new dataset, re-check whether this applies — don't assume distinct filenames mean distinct people. (Found as a real bug in step 4, after a filename-only split produced a 99% test accuracy — see Build Progress Log.)
- **This also applies inside cross-validation.** A plain `StratifiedKFold` over frame-level samples can split one video's frames across a fold boundary — leakage within the CV loop, not just across train/val/test. Use `StratifiedGroupKFold` with `groups=<video path per frame>` for any CV over frame-level data. (Found as a real bug in step 3 — see Build Progress Log.) Note this is video-group-aware, not identity-group-aware — CV folds within a single split can still repeat the same actor across folds; only train/val/test boundaries are identity-disjoint.
- Fit `StandardScaler`, `SMOTE`, and `PCA` on training data only. Never fit on val/test.
- Random seed = 42, set centrally in `config.py`, reused everywhere (numpy, sklearn, torch).
- Report cross-validation mean ± std, not a single train/test split accuracy, as the headline number.
- If test accuracy comes back at 90%+ at any point, treat it as a probable bug (leakage) and check the split logic before treating it as a result.

## Build order — build and confirm each step before moving to the next

1. `feature_extraction.py` (DFT profile + LBP + noise residuals, unit-tested on a dummy image)
2. `data_split.py` (identity-level split → `splits.json`; originally video/filename-level, reworked in step 4 — see Build Progress Log)
3. `train.py` (through to saved joblib model)
4. Evaluation script (confusion matrix, ROC-AUC, CV scores, error analysis)
5. `app.py` (Flask, `/analyze` endpoint)
6. Frontend
7. Wire frontend ↔ backend, manual end-to-end test

Do not skip ahead or generate multiple stages in one giant pass. Stop after each step and let me review before continuing.

## Code style

- Plain functions, minimal classes, minimal abstraction. This is a portfolio/interview project — I need to be able to explain every line myself, so don't write anything I wouldn't write.
- Comments explain **why** (e.g., "split at video level to prevent leakage"), not what the syntax does.
- No new frameworks or infra (no Docker, Celery, Redis, job queues, etc.) unless I explicitly ask.
- All paths/thresholds/hyperparameters live in `config.py` — no magic numbers scattered in files.
- Consistent error response shape across the API: `{"error": "<code>", "message": "<human-readable>"}`.

## Commands

```bash
python -m venv venv && source venv/Scripts/activate   # Windows Git Bash; venv already created
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu  # CPU-only wheels — install before the line below or pip may pull a CUDA build
pip install -r requirements.txt
python data_split.py       # (re)generates splits.json — identity-aware, ~5-8 min (face-embedding clustering on all real videos, cached to feature_cache/real_identity_embeddings.npz after first run; first-ever run also downloads ~107MB of pretrained VGGFace2 weights)
python train.py            # runs training pipeline, saves joblib model (~30-40 min incl. MTCNN extraction; caches to feature_cache/)
python evaluate.py         # generates confusion matrix, CV report, error analysis
python app.py              # starts Flask server
pytest tests/              # runs unit tests for feature extraction
```

**Python version:** doc pinned 3.10, but only 3.9/3.13/3.14 are installed on this machine. User chose **3.13** — `venv/` is built on it. `requirements.txt` pins are bumped from the doc's originals to versions with prebuilt Windows wheels for 3.13 (no local C build toolchain available).

## When in doubt

Ask me rather than guessing on: dataset paths, API response shape, frontend progress-indicator behavior, or anything involving the train/val/test split. Getting these wrong silently is worse than asking.

## Build Progress Log

Update this section after every completed build-order step: what was built, what deviated from the docs and why, what's still open.

### Step 1 — `feature_extraction.py` (confirmed)
- Built `config.py` (seed, image size, LBP/DFT params) + `feature_extraction.py` (DFT radial profile, LBP histogram, noise-residual stats, fused via `extract_features`) + `tests/test_feature_extraction.py` (7 tests on dummy images, all passing).
- **Deviation:** LBP histogram uses **18 bins** (`P+2` for `P=16`, skimage's uniform-LBP convention), not the doc's stated "26 bins" — 26 doesn't correspond to any standard LBP formula for these parameters. Flagged to user, no objection raised.
- **Deviation:** noise residual computed as `original − GaussianBlur(original)`, not wavelet denoising — avoids adding `pywt` as a dependency for what the doc lists as an "e.g." option; same rationale (high-pass residual stats).
- Fused feature vector length: 64 (DFT) + 18 (LBP) + 5 (residual stats) = **87**.

### Step 2 — `data_split.py` (confirmed, later superseded — see step 4)
- Found `dataset/` already populated (not empty as originally assumed) with a pre-existing `training/`/`testing/` split — see updated layout in the Project section above. Confirmed with user rather than assuming.
- Original approach: scanned the four class/split folders, held out 40 real + 40 fake from `training/` (seed=42, filename/video-level) as `val`, left `testing/` untouched as `test`. `assert_no_leakage()` verified zero cross-split filename duplicates across 500 videos.
- **This approach turned out to be insufficient** — filename-disjoint is not identity-disjoint for this dataset. Fully reworked in step 4; see there for what replaced it and why. Kept this entry for history since the filename-level split logic (and its leakage check) is still a real, correct component of the current approach, just not sufficient on its own.

### Step 3 — `train.py` (confirmed on the original split; retrained in step 4 after the split rework — numbers below are historical)
- New deps: `torch`/`torchvision` (CPU-only wheels via PyTorch's own index — regular PyPI has no cp313 wheel), `facenet-pytorch` (MTCNN), `scikit-learn`, `imbalanced-learn` (SMOTE), `joblib`, `tqdm`. facenet-pytorch's declared pins (`numpy<2`, `torch<2.3`, `pillow<10.3`) are stale; installed with `--no-deps` and verified MTCNN still works correctly against the newer versions actually installed.
- **Bug found and fixed during this step:** first run used plain `StratifiedKFold` for `GridSearchCV`, which splits at the frame level — a single video's frames could land on both sides of a CV fold, leaking correlated same-video information across the fold boundary (distinct from, and in addition to, the outer video-level split, which was never violated). Switched to `StratifiedGroupKFold` with `groups=<source video path per frame>`. Confirmed the fix mattered: CV F1 dropped across all three candidates after the fix (SVM 0.9709→0.9456, RF 0.9403→0.9150, GB 0.9428→0.9242), in the expected direction once the leak was closed. Promoted this as a general rule to the Non-negotiable methodology rules section above.
- **Deviation:** `BLUR_THRESHOLD` (Laplacian variance cutoff for discarding blurry face crops) has no number in the doc. An initial default of 100 discarded ~half of all otherwise-good frames — measured empirically that median blur variance on this dataset's (C23-compressed) face crops is ~70, so 100 was silently gutting the dataset. Recalibrated to 20 (~5th percentile of an empirical sample) so only genuine outliers get filtered. Caught via a smoke test before committing to the full run, not after.
- `feature_extraction.extract_features` frame-level results, extracted from 5 frames/video (train, val) with MTCNN face detection (confidence ≥0.95) + blur filtering: train 320 videos → 1,414 surviving frames (669 fake, 745 real), val 80 videos → 362 surviving frames (170 fake, 192 real). Cached to `feature_cache/{train,val}.npz` (includes per-frame video-group id, needed for `StratifiedGroupKFold`) so reruns skip the expensive MTCNN pass.
- Pipeline per doc section 7: `StandardScaler → SMOTE → PCA(0.95 variance) → classifier`, built as an `imblearn.pipeline.Pipeline` (not plain sklearn) so `GridSearchCV` refits the whole thing per fold — SMOTE only ever runs on that fold's own training portion, and is automatically skipped at predict time (no manual pipeline surgery needed for inference later).
- GridSearchCV (5-fold, group-aware) results, frame-level CV F1: Random Forest 0.9150±0.0387, Gradient Boosting 0.9242±0.0302, **SVM 0.9456±0.0265 (selected)**. MLP baseline (not eligible for selection, per doc section 9): 0.9116 val accuracy.
- Selected SVM confirmed on held-out val: 0.9318 frame-level accuracy. Retrained on train+val combined, saved to `best_deepfake_detector.joblib`.
- **Open question for step 4:** all reported numbers above are frame-level, not the video-level accuracy the doc's 80-88% target refers to. Video-level accuracy (via confidence-weighted voting across a video's frames, against the untouched `test/` set) is what the evaluation script in step 4 needs to produce — if that also lands at 90%+, apply the same leakage scrutiny this step needed.

### Step 4 — evaluation script + major leakage fix (confirmed)

**Built:** `face_utils.py` (extracted MTCNN/frame-sampling helpers out of `train.py` so `inference.py` doesn't import from a script named "train"), `inference.py` (`predict_video()` — confidence-weighted voting shared by `evaluate.py` now and `app.py` later), `evaluate.py` (video-level accuracy/precision/recall/F1/ROC-AUC/confusion matrix + heuristic error analysis against `test/`), `identity_grouping.py` (see below), rewrote `data_split.py`.

**The leakage this step found (bigger than step 3's):** first evaluation run scored **99.0% test accuracy** — triggered the project's 90% self-check rule immediately. Investigation:
1. Deepfake filenames follow `{source_actor}_{target_actor}__scenario__hash.mp4` (face-swap convention: source actor's face is swapped into frame). Parsing all 250 deepfake filenames found only **6 unique source actors in the entire dataset**, and the old filename-level test split reused 2 of those 6 — accounting for half the fake test videos. Same person, different clip, different filename → not caught by the filename-uniqueness check.
2. Checked whether real videos (`000.mp4`-`249.mp4`, no identity in filename) had the same issue: computed face embeddings (facenet-pytorch `InceptionResnetV1`, vggface2 weights) for all 250, clustered by cosine similarity > 0.5. Found 31 multi-member clusters (same person, multiple clips); **19 of those 31 crossed the old train/val/test boundary** — real-side identity leakage too, independent of the fake-side issue.
3. Tried cross-matching the 6 fake source actors against real-video embeddings to see if the same physical people appear in both pools. **Inconclusive** — best-match similarities landed at 0.36-0.55, indistinguishable from noise (GAN face-swap alters the embedding too much for reliable matching with this approach). Documented as an accepted, unresolved residual limitation rather than silently ignored.

**The fix:** `identity_grouping.py` — `get_fake_source_actor()` (filename parse), `build_real_identity_clusters()` (embedding + union-find, cached to `feature_cache/real_identity_embeddings.npz`), `assign_groups_to_splits()` (one deterministic greedy rule for both domains: sort groups by size descending, assign each to whichever split has the largest remaining quota deficit against targets `config.SPLIT_TARGET_COUNTS = {train:160, val:40, test:50}`). `data_split.py` now pools all 500 videos from both old folders and re-splits by identity group instead of by folder/filename. `assert_no_identity_leakage()` added alongside the original `assert_no_leakage()`; independently re-verified after the run (separately from the script's own assertions) — zero source-actor or real-cluster overlap across all three splits.

**Resulting split** (real hit targets exactly; fake is constrained by only 6 groups so landed close, not exact):
- train: 160 real + 165 fake (source actors 02, 03, 06)
- val: 40 real + 43 fake (source actors 04, 05)
- test: 50 real + 42 fake (source actor 01)

**Retrained on the new split** (`train_log.txt`): 1,445 train / 370 val frame-level samples. GridSearchCV: RF 0.9382±0.0096, GB 0.9386±0.0154, **SVM 0.9541±0.0121 (selected)**, MLP baseline 0.9081 val acc. Selected model val accuracy 0.9264. Saved to `best_deepfake_detector.joblib`, CV summary to `training_report.json`.

**Final video-level test evaluation** (`documentation/evaluation_report.md`, `error_analysis.md`, `confusion_matrix.png`, `roc_curve.png`):
- **Accuracy: 92.3%** (91/92 scored, 1 excluded — no usable face in any of its 10 sampled frames), ROC-AUC 0.9684
- Precision/Recall/F1 (REAL/FAKE): 0.904/0.949, 0.959/0.881, 0.931/0.914
- 7 misclassified videos (>5, so no "listed all of them, still short" caveat needed) — heuristic error analysis notes vote confidence, discarded-frame ratio, and per-frame prediction spread per case; explicitly flagged as automated proxy signals, not a substitute for watching the clips.
- 92.3% is still above the doc's 80-88% target and the project's 90% self-check line — `evaluation_report.md` carries a "Leakage sanity check" section explaining what was fixed (both leakage mechanisms above) and naming the one that's still open (real-to-fake cross-domain identity matching). Treat this number as meaningfully more trustworthy than the original 99%, not as fully leakage-proof.
- **Deviation:** report/error-analysis files are written with explicit `encoding="utf-8"` — first run without it mangled em-dashes on this Windows machine's default locale (cp1252).

**Open for future work, not blocking:** the unresolved real-to-fake identity cross-match (point 3 above) is the main remaining integrity gap. `StratifiedGroupKFold` in `train.py` groups CV folds by video path, not by actor — so within-train CV folds can still repeat an actor across folds; only the outer train/val/test boundary is identity-disjoint. Neither is currently a blocker, both are documented.

### Step 5 — `app.py` (confirmed)

**Built:** `explainability.py` (builds the base64 image set for the API response — reuses `feature_extraction.compute_dft_magnitude`, which was split out of `compute_dft_radial_profile` for exactly this reuse, verified via `pytest tests/` that the refactor didn't change behavior), `app.py` (single `/analyze` POST endpoint).

- **Synchronous, no job queue** — per Build_Specifications_Checklist.md section 6's explicit call, not a judgment call made here.
- Request: `multipart/form-data`, field name `video`. Response shape matches Build_Specifications_Checklist.md section 4 exactly, with one deliberate interpretation: discarded frames (no face survived detection/quality filtering) get `"prediction": null, "confidence": null` rather than a fabricated value — the doc's example JSON shows a discarded frame with a real prediction, which only makes sense under a "discard by low classifier confidence" model; this project discards *before* classification (face-detection confidence + blur only, per section 0), so a discarded frame never reaches the classifier and genuinely has no prediction to report.
- Errors use the project's standard `{"error": "<code>", "message": "..."}` shape: `missing_file`, `unsupported_format`, `invalid_video`, `no_face_detected`, `file_too_large` (413, via Flask's `MAX_CONTENT_LENGTH` + a custom error handler so the 413 body matches the same shape instead of Flask's default HTML page).
- Uploads saved to `uploads/<uuid>.<ext>`, deleted in a `finally` block regardless of outcome — verified empty after a full test run.
- **Manually tested against the running server** (all via curl, not just unit-level): real test video → correct REAL prediction; fake test video → correct FAKE prediction; synthetic no-face video → `no_face_detected`; corrupted file → `invalid_video`; `.txt` file → `unsupported_format`; missing field → `missing_file`; a 4-frame video against `FRAMES_PER_VIDEO_TEST=10` → sampled 4 frames and succeeded, didn't fail. Satisfies Build_Specifications_Checklist.md section 10's acceptance criterion (real/fake/no-face/corrupted all correctly shaped) plus the extra edge cases.

### Step 6 — frontend (confirmed)

**Built:** `templates/index.html`, `static/css/style.css`, `static/js/main.js`. Vanilla HTML/CSS/JS, no frameworks/build step, per code style. `app.py` got one addition: a `GET /` route rendering `index.html` (Flask's default static/templates folder conventions, no extra config needed).

- **Progress UI, per explicit user decision:** honest indeterminate spinner (no fake per-stage timed progress), but with static descriptive text listing the real pipeline stages (sampling → detection/quality-filtering → feature extraction → classification → confidence-weighted voting) so the user isn't left with just a bare "Analyzing..." — this was a "when in doubt, ask" item per CLAUDE.md and was asked before building.
- Sections: upload (drag-drop + file picker, client-side extension/size validation mirroring `config.ALLOWED_VIDEO_EXTENSIONS`/`MAX_UPLOAD_BYTES` — duplicated intentionally since JS can't read `config.py`) → processing → results (verdict badge, 5 explainability images, frame-by-frame table) → error (renders the API's `message` field directly).
- **Bug found and fixed during manual testing:** the disabled-button style (`button:disabled`) never applied — `button#analyze-button`'s ID selector (specificity 1,0,1) beat the pseudo-class rule (specificity 0,1,1) regardless of source order, so the "Analyze" button looked enabled even while genuinely disabled. Fixed by targeting `#analyze-button:disabled` etc. directly. Caught via `getComputedStyle` inspection during browser testing, not by eyeballing alone.
- **Tested for real in a browser**, not just curl: no file-upload tool was available for this browser surface, so a real test video was temporarily copied into `static/`, fetched via the page's own `fetch()`, turned into a `File` via `DataTransfer`, and fed through the actual file input's `change` event — a genuine end-to-end run through the real `/analyze` endpoint, not mocked data. Verified: file selection enables the button and shows the filename; clicking Analyze shows the processing screen; a real REAL-labeled test video renders the green verdict, 99.9% confidence, all 5 explainability images, and a 10-row frame table with correct per-frame values; "Analyze another video" resets cleanly; an unsupported extension is rejected client-side before upload; a corrupted `.mp4` (valid extension, garbage bytes) correctly surfaces the backend's `invalid_video` error in the error card, and "Try another video" resets from there too. Temp video removed from `static/` afterward.

### Step 7 — wire frontend ↔ backend, manual end-to-end test (done, awaiting user confirmation)

Frontend/backend were already wired as of step 6 (same Flask app serves both `/` and `/analyze`); this step's job was closing out the remaining acceptance-criteria coverage and infra Build_Specifications_Checklist.md called for.

- **`tests/sample_videos/sample_real.mp4`** added (copied from `dataset/testing/real/218.mp4`, the smallest test-split real video, ~460KB) per Build_Specifications_Checklist.md section 7 — "at least one test video ... so Claude Code can verify the inference path works end-to-end without needing the full dataset." Verified it actually works: `inference.predict_video()` on it directly returns REAL at 99.1% confidence, 10/10 frames used.
- **Completed browser-driven coverage of all 4 acceptance-criteria scenarios** (step 6 covered real-video-success and corrupted-file-error; this step added fake-video-success and no-face-error through the same real `/analyze` flow): fake test video → red FAKE badge at 72.0% confidence (matches the earlier curl result exactly); synthetic no-face video → `no_face_detected` renders correctly in the error card.
- **Investigated an apparent anomaly, found it wasn't a bug:** mid-testing, one browser check showed a result for a file ("353.mp4") that didn't match anything I'd uploaded. Server logs showed more `POST /analyze` requests than my own test count. Rather than assume it away, did an isolated single-click test with an explicit click-counter and before/after server-log-line-count check — confirmed exactly one request per click, correct status code. Root cause: a `snapshots/` directory (`D_*.png`, `R_*.png`, `input_screen.png`, etc., timestamped in the same window) appeared during testing — the user was independently testing the same running dev server concurrently. Not touched; it's the user's own artifact, not build output.
- Full `pytest tests/` regression run: still 7/7 passing after all changes since step 1.
- `uploads/` confirmed empty after the full testing session (temp files cleaned up correctly under concurrent load too).

**All 7 build-order steps complete.** Remaining known limitations, all already documented rather than silently accepted: unresolved real-to-fake identity cross-match (step 4), CV folds within train not actor-disjoint (step 4), C23-only training data won't generalize claim-free to C40 or other manipulation types (documentation section 13).
