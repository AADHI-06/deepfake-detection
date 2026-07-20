"""Training pipeline (documentation section 7):

extract fused features from train/val frames -> GridSearchCV + StratifiedKFold
over Random Forest / Gradient Boosting / SVM (scaler+SMOTE+PCA inside the CV
pipeline, so each fold only ever sees its own training portion) -> compare
against an MLP baseline -> select best by mean CV F1 -> confirm on held-out
val -> retrain the winning config on train+val -> save with joblib.
"""

import json
import os

import cv2
import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config
from data_split import load_splits
from face_utils import detect_and_filter_face, sample_frame_indices
from feature_extraction import extract_features

LABEL_REAL = 0
LABEL_FAKE = 1


def extract_video_features(video_path, n_frames):
    """Sample frames from one video, run face detection + quality filtering,
    return a fused feature vector for each frame that survives."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted_indices = sample_frame_indices(total, n_frames)

    features = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted_indices:
            face = detect_and_filter_face(frame)
            if face is not None:
                features.append(extract_features(face))
        idx += 1
    cap.release()
    return features


def build_feature_matrix(video_paths, label, n_frames):
    """Also returns a group id (the source video path) per frame. Frames from the
    same video are correlated (same face, same lighting) — CV folds must keep a
    video's frames together, or a fold effectively trains and tests on the same
    video, inflating scores the same way a train/test video leak would."""
    X, y, groups = [], [], []
    for i, path in enumerate(video_paths):
        for feats in extract_video_features(path, n_frames):
            X.append(feats)
            y.append(label)
            groups.append(path)
        if (i + 1) % 20 == 0:
            print(f"    ...{i + 1}/{len(video_paths)} videos")
    return X, y, groups


def load_or_build_split_features(split_name, splits, n_frames):
    """Face detection is the expensive step here, so cache the extracted feature
    matrix per split to avoid reprocessing all videos on every rerun."""
    os.makedirs(config.FEATURE_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(config.FEATURE_CACHE_DIR, f"{split_name}.npz")

    if os.path.exists(cache_path):
        print(f"  loading cached {split_name} features from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data["X"], data["y"], data["groups"]

    print(f"  extracting {split_name} real videos...")
    X_real, y_real, g_real = build_feature_matrix(splits[split_name]["real"], LABEL_REAL, n_frames)
    print(f"  extracting {split_name} fake videos...")
    X_fake, y_fake, g_fake = build_feature_matrix(splits[split_name]["fake"], LABEL_FAKE, n_frames)

    X = np.array(X_real + X_fake)
    y = np.array(y_real + y_fake)
    groups = np.array(g_real + g_fake)
    np.savez(cache_path, X=X, y=y, groups=groups)
    return X, y, groups


def build_pipeline(classifier):
    """scaler -> SMOTE -> PCA -> classifier, in the order documentation section 7
    specifies. Using imblearn's Pipeline (not sklearn's) means SMOTE only runs
    during .fit(), and GridSearchCV refits this whole pipeline per CV fold, so
    each fold's SMOTE/scaler/PCA are fit on that fold's training data only."""
    return ImbPipeline(steps=[
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=config.RANDOM_SEED)),
        ("pca", PCA(n_components=config.PCA_VARIANCE_TARGET, random_state=config.RANDOM_SEED)),
        ("classifier", classifier),
    ])


CANDIDATES = {
    "random_forest": (RandomForestClassifier(random_state=config.RANDOM_SEED), config.RF_PARAM_GRID),
    "gradient_boosting": (GradientBoostingClassifier(random_state=config.RANDOM_SEED), config.GB_PARAM_GRID),
    "svm": (SVC(kernel="rbf", probability=True, random_state=config.RANDOM_SEED), config.SVM_PARAM_GRID),
}


def run_grid_search(X_train, y_train, groups_train):
    # Group-aware: keeps every video's frames on the same side of each fold, so a
    # fold never trains and tests on frames from the same source video.
    cv = StratifiedGroupKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    results = {}
    for name, (classifier, grid) in CANDIDATES.items():
        print(f"  grid search: {name}")
        search = GridSearchCV(build_pipeline(classifier), grid, scoring="f1", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train, groups=groups_train)
        results[name] = search
        std = search.cv_results_["std_test_score"][search.best_index_]
        print(f"    best CV F1 = {search.best_score_:.4f} +/- {std:.4f}  params={search.best_params_}")
    return results


def evaluate_mlp_baseline(X_train, y_train, X_val, y_val):
    """Baseline comparison only (documentation section 9) — not grid-searched,
    not eligible for selection, likely to overfit at this dataset size."""
    pipeline = build_pipeline(MLPClassifier(random_state=config.RANDOM_SEED, max_iter=500))
    pipeline.fit(X_train, y_train)
    val_acc = pipeline.score(X_val, y_val)
    print(f"  mlp_baseline: val accuracy = {val_acc:.4f} (not part of model selection)")
    return val_acc


def select_best_model(results, X_val, y_val):
    best_name = max(results, key=lambda name: results[name].best_score_)
    best_search = results[best_name]
    val_acc = best_search.score(X_val, y_val)
    print(f"selected model: {best_name}  (CV F1={best_search.best_score_:.4f}, val accuracy={val_acc:.4f})")
    return best_name, best_search.best_estimator_, val_acc


def retrain_on_full_data(best_estimator, X_train, y_train, X_val, y_val):
    """Documentation section 7 step 13: retrain the winning hyperparameters on
    train+val combined for the final artifact; test/ stays untouched for
    evaluate.py."""
    X_full = np.concatenate([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    final_pipeline = clone(best_estimator)
    final_pipeline.fit(X_full, y_full)
    return final_pipeline


def main():
    splits = load_splits()

    print("Extracting train features...")
    X_train, y_train, groups_train = load_or_build_split_features("train", splits, config.FRAMES_PER_VIDEO_TRAIN)
    print(f"  train: {len(y_train)} frame-level samples "
          f"({int(np.sum(y_train))} fake, {int(len(y_train) - np.sum(y_train))} real)")

    print("Extracting val features...")
    X_val, y_val, _ = load_or_build_split_features("val", splits, config.FRAMES_PER_VIDEO_VAL)
    print(f"  val: {len(y_val)} frame-level samples "
          f"({int(np.sum(y_val))} fake, {int(len(y_val) - np.sum(y_val))} real)")

    print("Running GridSearchCV (random_forest / gradient_boosting / svm)...")
    results = run_grid_search(X_train, y_train, groups_train)

    print("Evaluating MLP baseline...")
    mlp_val_acc = evaluate_mlp_baseline(X_train, y_train, X_val, y_val)

    best_name, best_estimator, best_val_acc = select_best_model(results, X_val, y_val)

    print("Retraining best model on train+val...")
    final_pipeline = retrain_on_full_data(best_estimator, X_train, y_train, X_val, y_val)

    joblib.dump(final_pipeline, config.MODEL_OUTPUT_PATH)
    print(f"Saved final pipeline ({best_name}) to {config.MODEL_OUTPUT_PATH}")

    # Persisted so evaluate.py can report "CV mean +/- std" (documentation section
    # 10) without re-running the ~30min GridSearchCV.
    report = {
        "selected_model": best_name,
        "selected_model_val_accuracy": best_val_acc,
        "mlp_baseline_val_accuracy": mlp_val_acc,
        "candidates": {
            name: {
                "cv_f1_mean": search.best_score_,
                "cv_f1_std": float(search.cv_results_["std_test_score"][search.best_index_]),
                "best_params": search.best_params_,
            }
            for name, search in results.items()
        },
        "train_samples": len(y_train),
        "val_samples": len(y_val),
    }
    with open(config.TRAINING_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved training report to {config.TRAINING_REPORT_PATH}")


if __name__ == "__main__":
    main()
