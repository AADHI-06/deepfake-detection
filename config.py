"""Central config: paths, thresholds, hyperparameters. No magic numbers elsewhere."""

import os
import random

import numpy as np
import torch

# Reproducibility — set once, reused everywhere (numpy, sklearn via random_state=, torch)
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Dataset layout — already present on disk, pre-split into training/testing at the
# video level (verified: no filename overlap between the two). See data_split.py.
DATASET_DIR = "dataset"
TRAINING_REAL_DIR = os.path.join(DATASET_DIR, "training", "real")
TRAINING_FAKE_DIR = os.path.join(DATASET_DIR, "training", "deepfake")
TESTING_REAL_DIR = os.path.join(DATASET_DIR, "testing", "real")
TESTING_FAKE_DIR = os.path.join(DATASET_DIR, "testing", "deepfake")

# Target video counts per split, per class (real and fake each targeted
# separately). Identity-group sizes are coarse (esp. the 6 fake source actors),
# so actual counts land close to these but rarely hit them exactly — see
# data_split.py's assign_groups_to_splits.
SPLIT_TARGET_COUNTS = {"train": 160, "val": 40, "test": 50}

SPLITS_JSON_PATH = "splits.json"

# Face crop size fed into all three feature branches (see documentation section 6)
IMAGE_SIZE = (128, 128)  # (width, height) for cv2.resize

# LBP (texture branch, section 6.2)
LBP_RADIUS = 2
LBP_POINTS = 16
LBP_METHOD = "uniform"
# Uniform LBP with P points has P+2 distinct pattern labels (skimage convention).
# Note: source doc says "26 bins" but that doesn't match P=16 under the standard
# uniform-LBP formula — using the correct P+2=18 here; flagged for confirmation.
LBP_N_BINS = LBP_POINTS + 2

# DFT (frequency branch, section 6.1) — number of radial bins for the profile
DFT_RADIAL_BINS = 64

# Frame sampling (documentation section 7 step 3 / section 4)
FRAMES_PER_VIDEO_TRAIN = 5
FRAMES_PER_VIDEO_VAL = 5
FRAMES_PER_VIDEO_TEST = 10  # used by inference/evaluation, not train.py

# Face detection + quality filtering (section 7 step 4). Doc specifies the 0.95
# confidence threshold explicitly. BLUR_THRESHOLD isn't given a number in the doc;
# a generic default of 100 turned out to discard ~half of all frames (measured
# median Laplacian variance on this dataset's face crops is ~70 — C23 compression
# softens detail, so "typical" frames here read as blurrier than an uncompressed
# photo would). Calibrated instead to ~5th percentile of an empirical sample
# (n=47 across 16 videos) so only genuine outliers get filtered.
FACE_CONFIDENCE_THRESHOLD = 0.95
BLUR_THRESHOLD = 20.0

# PCA (section 7 step 10)
PCA_VARIANCE_TARGET = 0.95

# Model selection (section 9). RF grid is exactly as specified in the doc.
# GB/SVM grids aren't given explicit values there — chosen as small, reasonable
# grids appropriate for a dataset this size (a few thousand frame-level samples).
CV_FOLDS = 5

RF_PARAM_GRID = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_leaf": [1, 2, 4],
}
GB_PARAM_GRID = {
    "classifier__n_estimators": [100, 200],
    "classifier__learning_rate": [0.05, 0.1],
    "classifier__max_depth": [2, 3, 4],
}
SVM_PARAM_GRID = {
    "classifier__C": [0.1, 1, 10],
    "classifier__gamma": ["scale", 0.01, 0.001],
}

FEATURE_CACHE_DIR = "feature_cache"
MODEL_OUTPUT_PATH = "best_deepfake_detector.joblib"
TRAINING_REPORT_PATH = "training_report.json"

# Inference / evaluation (documentation section 8 + 10)
# Video-level label = FAKE if the mean of per-frame P(fake) across surviving
# frames is >= this; confidence-weighted voting per section 0 fix #3.
VOTE_THRESHOLD = 0.5

DOCUMENTATION_DIR = "documentation"
MIN_ERROR_ANALYSIS_EXAMPLES = 5

# Flask API (Build_Specifications_Checklist.md section 4/5)
UPLOAD_DIR = "uploads"
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500MB, reject before processing
