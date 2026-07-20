# Deepfake Detection Project — Improved Documentation

> Revised version of the original pipeline with specific changes aimed at (a) closing the accuracy gap from ~75% toward a defensible 80–90% range, and (b) making the project interview-defensible, not just functional.

---

## 0. What Changed and Why (read this first)

The original pipeline (DFT → radial frequency profile → StandardScaler → PCA → Random Forest → majority vote) is a reasonable *idea* but has four concrete weaknesses that cap it near 75%. Each fix below is tied to one of those weaknesses — this isn't a random list of "best practices," it's a direct response to what's likely holding accuracy down.

| # | Weakness in original pipeline | Fix | Why it should help |
|---|---|---|---|
| 1 | Single feature type (radial frequency profile only) — throws away a lot of signal | Add complementary handcrafted features: **Local Binary Patterns (LBP)** on the face crop, **noise residual statistics** (high-pass filter residual mean/std/kurtosis), and **JPEG/compression artifact features** (blockiness score) | Deepfake artifacts show up in texture (LBP), sensor-noise inconsistency (residuals), and compression mismatch — frequency profile alone misses these |
| 2 | Risk of frame-level train/test split → **video leakage** | Enforce **video-level split** before frame sampling: no video's frames appear in both train and test | If frames from the same video leak across splits, accuracy is inflated and meaningless — this is the single most common bug in deepfake detection projects |
| 3 | Simple majority voting (each frame counted equally regardless of confidence) | **Confidence-weighted voting**: sum predicted probabilities across frames instead of counting hard votes; also **discard low-confidence / low-face-quality frames** (blurry, occluded, off-angle) before voting | A frame where the model is 51% confident shouldn't count the same as one where it's 95% confident; filtering bad frames reduces noise |
| 4 | No hyperparameter search, no cross-validation, single train/test split | **Stratified 5-fold cross-validation** on the training set + **GridSearchCV** for Random Forest (`n_estimators`, `max_depth`, `min_samples_leaf`) and compare against Gradient Boosting / SVM properly | Gives you a real accuracy range (mean ± std) to quote in interviews instead of one lucky/unlucky split number |
| 5 | No calibration or error analysis | Add **confusion matrix, ROC-AUC, per-class precision/recall**, and a short written error analysis (which kinds of videos get misclassified) | This is what separates "I ran a model" from "I understand my model" in an interview |

**Realistic target after these changes: 80–88% video-level accuracy**, with an honest cross-validated range. If you hit 90%+, verify immediately for leakage before celebrating — it's more likely a bug than a breakthrough at this dataset size.

---

## 1. Project Overview

**Title:** Deepfake Detection Using Frequency-Domain and Texture-Based Feature Fusion with Classical Machine Learning

**Objective:** Detect manipulated (deepfake) videos using an explainable, lightweight pipeline that fuses frequency-domain and texture/noise-domain handcrafted features, avoiding the interpretability and compute cost of deep CNN-based detectors.

**One-line pitch for resume/interview:**
> "Built a feature-fusion deepfake detector combining frequency-domain and texture-residual analysis with classical ML, achieving [X]% video-level accuracy on FaceForensics++ with rigorous video-level train/test separation to prevent leakage."

---

## 2. Motivation (unchanged, still valid)

- Digital media forensics
- Social media content verification
- Educational/demonstration tooling
- Research baseline for explainable detection
- Preliminary screening layer before expensive deep models

---

## 3. Technology Stack

| Layer | Tools |
|---|---|
| Language | Python |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Computer Vision | OpenCV, facenet-pytorch (MTCNN) |
| ML | Scikit-learn (Random Forest, Gradient Boosting, SVM, MLP) |
| Feature Engineering | DFT, LBP (scikit-image), noise residuals (custom high-pass filter) |
| Preprocessing | StandardScaler, SMOTE, PCA |
| Model Selection | GridSearchCV, StratifiedKFold |
| Visualization | Matplotlib, Seaborn (for confusion matrix / ROC) |
| Serialization | Joblib |

---

## 4. Dataset

**Dataset:** FaceForensics++ (C23 compression)

**Classes:** Original (REAL) vs. DeepFakes (FAKE)

**Split strategy — CHANGED:**
- Split is performed **at the video level first**, then frames are sampled within each split.
- Training: 200 REAL + 200 FAKE videos → 5 frames/video
- Validation (new): 40 REAL + 40 FAKE videos, held out from training videos → used for hyperparameter selection
- Testing: 50 REAL + 50 FAKE videos (completely separate videos, never seen in train/val) → 10 frames/video

This three-way split is the single most important methodological fix — it's what makes your accuracy number defensible.

---

## 5. Complete Workflow (updated)

```text
Upload Video
      │
      ▼
Validate Video
      │
      ▼
Sample Frames (uniform temporal sampling)
      │
      ▼
MTCNN Face Detection
      │
      ▼
Frame Quality Filter (blur/occlusion/confidence check)  ← NEW
      │
      ▼
Face Crop
      │
      ▼
Resize (128×128)
      │
      ├──────────────┬─────────────────┐
      ▼              ▼                 ▼
  Grayscale      Grayscale         Grayscale
      │              │                 │
      ▼              ▼                 ▼
     DFT            LBP          Noise Residual        ← NEW: 3 feature branches
      │              │                 │
      ▼              ▼                 ▼
 Radial Profile  Histogram      Residual Stats
      │              │                 │
      └──────┬───────┴────────┬────────┘
             ▼                ▼
       Feature Concatenation (Feature Fusion)          ← NEW
                    │
                    ▼
              StandardScaler
                    │
                    ▼
                   PCA
                    │
                    ▼
       Random Forest (GridSearchCV-tuned)               ← CHANGED
                    │
                    ▼
          Frame Predictions (with confidence)
                    │
                    ▼
     Quality-Filtered, Confidence-Weighted Voting       ← CHANGED
                    │
                    ▼
          Final REAL / FAKE Prediction + Confidence Score
```

---

## 6. Feature Engineering (expanded)

### 6.1 Frequency-domain branch (original, kept)
Face → Grayscale → DFT → FFT Shift → Magnitude Spectrum → Radial Frequency Profile

### 6.2 Texture branch (new)
Face → Grayscale → Local Binary Pattern (uniform, radius=2, points=16) → LBP histogram (26 bins)

*Rationale:* GAN-generated faces often have subtly different micro-texture patterns than camera-captured skin, especially around blending boundaries.

### 6.3 Noise residual branch (new)
Face → Grayscale → High-pass filter (e.g., Laplacian or wavelet denoising residual) → Residual image → Statistical features (mean, std, skewness, kurtosis, energy)

*Rationale:* Real camera sensors leave a consistent noise fingerprint (PRNU-like); GAN outputs typically have a different, more uniform noise floor. This is a well-established forensic signal.

### 6.4 Fusion
Concatenate all three feature vectors → single feature vector per frame → proceed to scaling/PCA as before.

---

## 7. Training Pipeline (updated)

1. Load FaceForensics++ dataset.
2. **Split videos into train/val/test at the video level (before any frame sampling).**
3. Sample 5 frames/video (train), 5 frames/video (val).
4. Detect face using MTCNN; **discard frames below a face-detection confidence threshold (e.g., <0.95) or with excessive blur (Laplacian variance below threshold).**
5. Crop face, resize to 128×128.
6. Extract DFT radial profile, LBP histogram, and noise residual stats.
7. Concatenate into fused feature vector.
8. Standardize features (fit scaler on train only).
9. Apply SMOTE on training data only.
10. Reduce dimensionality using PCA (95% variance, fit on train only).
11. **Run StratifiedKFold (k=5) cross-validation with GridSearchCV** over Random Forest, Gradient Boosting, SVM hyperparameters.
12. Select best model by mean CV F1-score, confirm on held-out validation set.
13. Retrain best model+hyperparameters on full train+val data.
14. Save model pipeline (scaler + PCA + classifier) using Joblib.

---

## 8. Inference Pipeline (updated)

1. Upload video.
2. Sample 10 uniformly distributed frames.
3. Detect primary face per frame; **skip low-quality frames.**
4. Extract fused features (DFT + LBP + residuals).
5. Apply saved StandardScaler → PCA.
6. Predict probability per frame.
7. **Weight each frame's vote by its predicted probability (confidence), not a flat count.**
8. Aggregate into final REAL/FAKE label + overall confidence score.

---

## 9. Machine Learning — Model Selection (updated)

Candidates evaluated via 5-fold CV + GridSearchCV:
- Random Forest (`n_estimators`: [100, 200, 300], `max_depth`: [None, 10, 20], `min_samples_leaf`: [1, 2, 4])
- Gradient Boosting (`n_estimators`, `learning_rate`, `max_depth`)
- SVM (RBF kernel, `C`, `gamma` grid)
- MLP (kept as baseline comparison only — likely to overfit on this dataset size)

Report **mean ± std CV accuracy/F1** for each, not a single number. Select based on best mean with acceptable variance, not best single fold.

---

## 10. Evaluation (updated — this is the section that impresses interviewers)

Report all of the following, at the **video level**:
- Accuracy
- Precision, Recall, F1 (per class)
- ROC-AUC
- Confusion matrix
- Cross-validation mean ± std (not just test set number)
- **Error analysis**: manually inspect 5–10 misclassified videos — is there a pattern? (e.g., heavy compression, extreme poses, low light)

Old approximate accuracy: **~75%**
Target after improvements: **80–88%**, reported as a CV range, with test-set confirmation.

If you get 90%+: stop and check for video-level leakage before reporting it.

---

## 11. Project Structure (updated)

```text
Deepfake-Detection/

app.py
train.py
feature_extraction.py        ← NEW (DFT + LBP + residuals module)
data_split.py                 ← NEW (video-level split logic)
model_selection.py            ← NEW (GridSearchCV + CV)
best_deepfake_detector.joblib
requirements.txt

templates/
    index.html

static/
    css/
    js/
    images/

uploads/
dataset/
documentation/
    error_analysis.md          ← NEW
    evaluation_report.md        ← NEW
```

---

## 12. Advantages (unchanged, still true)

- Lightweight, explainable, low compute
- Fast inference
- Classical ML — easy to explain every step in an interview
- Feature fusion adds robustness without adding a deep model

## 13. Limitations (honest, expanded)

- Still depends on face detection quality — total failure if no face detected
- Single primary face processed per frame (no multi-face)
- No temporal/motion modeling (frame-independent)
- Sensitive to heavy compression (C40) — likely to underperform there; state this explicitly
- Limited to FaceForensics++ DeepFakes subset — doesn't generalize claim-free to Face2Face, FaceSwap, NeuralTextures, or diffusion-based deepfakes without retesting
- Dataset size (400 videos) is small for strong statistical claims — report CV variance to be honest about this

## 14. Future Scope

- Test generalization across other FaceForensics++ manipulation types (Face2Face, FaceSwap, NeuralTextures)
- Multi-face detection
- Live webcam inference
- Compare against a lightweight CNN baseline (e.g., EfficientNet-B0) to quantify the accuracy/interpretability tradeoff explicitly
- Cloud deployment, mobile app

---

## 15. Interview Talking Points (new section — use this)

Be ready for these questions, because they will come up:

1. **"Why classical ML instead of a CNN?"** → Explainability and compute cost tradeoff; state the accuracy gap honestly (CNNs likely score higher, ~90%+, but are less interpretable and heavier).
2. **"How do you know your test accuracy isn't inflated?"** → Explain the video-level split explicitly. This is your strongest answer if asked.
3. **"What happens on compressed/low-quality video?"** → Point to your C23 vs. C40 limitation and the frequency-degradation reasoning.
4. **"Would this work on a deepfake type it wasn't trained on?"** → Be honest: likely not well, since it's trained only on the DeepFakes subset. This is a stated limitation, not a hidden flaw.
5. **"What's your actual accuracy?"** → Give the CV mean ± std and the held-out test number, not just one figure.

---

## 16. Conclusion

This revised pipeline keeps the original project's core strength — an explainable, lightweight, frequency-domain approach — while fixing the methodological gaps (feature poverty, leakage risk, unweighted voting, lack of cross-validation) that capped the original at ~75%. The realistic, defensible target is 80–88% video-level accuracy, reported honestly with cross-validation and error analysis, which is a stronger interview asset than an unexplained 90%+ number.
