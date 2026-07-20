"""Evaluation (documentation section 10): video-level accuracy/precision/recall/F1,
confusion matrix, ROC-AUC, and a written error analysis — all against the
untouched test/ split, which no earlier step has read.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")  # headless: saving PNGs only, no display needed
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve

import config
from data_split import load_splits
from inference import LABEL_FAKE, LABEL_NAMES, LABEL_REAL, load_pipeline, predict_video


def run_test_predictions(pipeline, splits):
    """Predict every test video. Returns (results, excluded) — excluded holds
    paths where no sampled frame yielded a usable face (documentation-required
    error case, not something to silently drop from view)."""
    results = []
    excluded = []

    test_videos = [(p, LABEL_REAL) for p in splits["test"]["real"]] + \
                  [(p, LABEL_FAKE) for p in splits["test"]["fake"]]

    for i, (path, true_label) in enumerate(test_videos):
        label, confidence, frame_results = predict_video(path, pipeline, config.FRAMES_PER_VIDEO_TEST)
        if label is None:
            excluded.append(path)
            continue

        used = [f for f in frame_results if f["used_in_vote"]]
        probs = [f["probability_fake"] for f in used]
        results.append({
            "path": path,
            "true_label": true_label,
            "predicted_label": label,
            "confidence": confidence,
            "frames_sampled": len(frame_results),
            "frames_used": len(used),
            "frames_discarded": len(frame_results) - len(used),
            "frame_prob_std": float(np.std(probs)) if len(probs) > 1 else 0.0,
        })
        if (i + 1) % 20 == 0:
            print(f"  ...{i + 1}/{len(test_videos)} test videos")

    return results, excluded


def compute_metrics(results):
    y_true = np.array([r["true_label"] for r in results])
    y_pred = np.array([r["predicted_label"] for r in results])
    # Reconstruct continuous P(fake) per video for ROC-AUC (confidence is
    # relative to the predicted label, not always "probability of FAKE")
    y_score = np.array([
        r["confidence"] if r["predicted_label"] == LABEL_FAKE else 1 - r["confidence"]
        for r in results
    ])

    accuracy = float(np.mean(y_true == y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[LABEL_REAL, LABEL_FAKE], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[LABEL_REAL, LABEL_FAKE])
    auc = roc_auc_score(y_true, y_score)

    metrics = {
        "accuracy": accuracy,
        "precision": {"REAL": float(precision[0]), "FAKE": float(precision[1])},
        "recall": {"REAL": float(recall[0]), "FAKE": float(recall[1])},
        "f1": {"REAL": float(f1[0]), "FAKE": float(f1[1])},
        "support": {"REAL": int(support[0]), "FAKE": int(support[1])},
        "confusion_matrix": cm.tolist(),
        "roc_auc": float(auc),
    }
    return metrics, y_true, y_score, cm


def save_confusion_matrix_plot(cm, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["REAL", "FAKE"], yticklabels=["REAL", "FAKE"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (video-level, test set)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_roc_curve_plot(y_true, y_score, auc, path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (video-level, test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build_error_analysis(results):
    """Heuristic error analysis over measurable pipeline signals (vote
    confidence, discarded-frame ratio, per-frame prediction spread). This is an
    automated first pass, not a substitute for watching the misclassified clips
    — flagged explicitly rather than inventing claims about video content the
    script never looked at."""
    misclassified = [r for r in results if r["true_label"] != r["predicted_label"]]

    lines = ["# Error Analysis", ""]
    lines.append(f"{len(misclassified)} of {len(results)} test videos misclassified "
                 f"({len(misclassified) / len(results):.1%}).")
    lines.append("")

    if len(misclassified) < config.MIN_ERROR_ANALYSIS_EXAMPLES:
        lines.append(f"**Note:** fewer than {config.MIN_ERROR_ANALYSIS_EXAMPLES} "
                      f"misclassifications occurred on this test set — all of them are "
                      f"listed below rather than a fixed-size sample.")
        lines.append("")

    lines.append(
        "Each entry lists automated proxy signals only (vote confidence, how many "
        "of the sampled frames were discarded by face-detection/blur filtering, and "
        "how much per-frame predictions disagreed with each other). These are "
        "measurable from the pipeline's own output, not a human review of the video "
        "content — a real error-analysis pass should also involve watching these clips."
    )
    lines.append("")

    for r in misclassified:
        true_name = LABEL_NAMES[r["true_label"]]
        pred_name = LABEL_NAMES[r["predicted_label"]]
        reasons = []
        if r["confidence"] < 0.6:
            reasons.append("near-threshold vote confidence — model was genuinely uncertain")
        if r["frames_discarded"] > 0:
            reasons.append(
                f"{r['frames_discarded']}/{r['frames_sampled']} sampled frames discarded "
                f"by face-detection/blur filtering — fewer frames contributed to the vote"
            )
        if r["frame_prob_std"] > 0.25:
            reasons.append("high disagreement between individual frame predictions "
                            "(possible inconsistent visibility/pose across the video)")
        if not reasons:
            reasons.append("confidently wrong — no obvious pipeline-level explanation from "
                            "these proxy signals; needs manual review of the clip")

        lines.append(f"## {r['path']}")
        lines.append(f"- True: {true_name}, Predicted: {pred_name}, confidence: {r['confidence']:.3f}")
        lines.append(f"- Frames used in vote: {r['frames_used']}/{r['frames_sampled']}")
        lines.append(f"- Candidate reasons: {'; '.join(reasons)}")
        lines.append("")

    return "\n".join(lines)


def build_evaluation_report(metrics, training_report, excluded, n_results):
    lines = ["# Evaluation Report", "", "## Test-set video-level metrics", ""]
    lines.append(f"- Accuracy: {metrics['accuracy']:.4f}")
    lines.append(f"- ROC-AUC: {metrics['roc_auc']:.4f}")
    lines.append(f"- Precision (REAL / FAKE): {metrics['precision']['REAL']:.4f} / {metrics['precision']['FAKE']:.4f}")
    lines.append(f"- Recall (REAL / FAKE): {metrics['recall']['REAL']:.4f} / {metrics['recall']['FAKE']:.4f}")
    lines.append(f"- F1 (REAL / FAKE): {metrics['f1']['REAL']:.4f} / {metrics['f1']['FAKE']:.4f}")
    lines.append(f"- Support (REAL / FAKE): {metrics['support']['REAL']} / {metrics['support']['FAKE']}")
    lines.append(f"- Videos scored: {n_results} (of {n_results + len(excluded)} in test/); "
                 f"{len(excluded)} excluded — no usable face in any sampled frame")
    if excluded:
        lines.append("  - Excluded: " + ", ".join(excluded))
    lines.append("")

    lines.append("## Cross-validation results (from training, train split only)")
    lines.append("")
    lines.append(f"Selected model: **{training_report['selected_model']}**")
    lines.append("")
    for name, c in training_report["candidates"].items():
        marker = " (selected)" if name == training_report["selected_model"] else ""
        lines.append(f"- {name}{marker}: CV F1 = {c['cv_f1_mean']:.4f} +/- {c['cv_f1_std']:.4f}, "
                     f"best params = {c['best_params']}")
    lines.append(f"- mlp_baseline (not eligible for selection): "
                 f"val accuracy = {training_report['mlp_baseline_val_accuracy']:.4f}")
    lines.append(f"- Selected model val accuracy (held out during CV): "
                 f"{training_report['selected_model_val_accuracy']:.4f}")
    lines.append("")

    if metrics["accuracy"] >= 0.90:
        lines.append("## Leakage sanity check")
        lines.append("")
        lines.append(
            f"Test accuracy is {metrics['accuracy']:.1%}, at or above the project's 90% "
            "self-check threshold (see CLAUDE.md). Two leakage mechanisms were found and "
            "fixed earlier in this project: CV-fold leakage in training (step 3) and "
            "actor/identity leakage in the original filename-level split, where the same "
            "actor's face appeared in both train and test under different clips (step 4 — "
            "this is what the 99%->92% jump reflects). The current split is identity-group-"
            "disjoint on both sides: fake videos by source actor (parsed from filenames), "
            "real videos by face-embedding cluster (cosine similarity > 0.5)."
        )
        lines.append("")
        lines.append(
            "**One leakage mechanism remains unresolved, not just unverified:** whether the "
            "same physical person appears as both a REAL video and as a FAKE video's source "
            "actor in different splits. Cross-matching the 6 known fake source actors against "
            "all 250 real-video embeddings was attempted and came back inconclusive (best-match "
            "cosine similarities of 0.36-0.55, indistinguishable from noise — GAN face-swap "
            "alters the embedding enough that this comparison isn't reliable with the tools "
            "used here). This test accuracy should be read with that gap in mind, not treated "
            "as a fully leakage-proof number."
        )
        lines.append("")

    return "\n".join(lines)


def main():
    os.makedirs(config.DOCUMENTATION_DIR, exist_ok=True)

    pipeline = load_pipeline()
    splits = load_splits()

    with open(config.TRAINING_REPORT_PATH) as f:
        training_report = json.load(f)

    print("Running predictions on test set...")
    results, excluded = run_test_predictions(pipeline, splits)
    print(f"  scored {len(results)} videos, excluded {len(excluded)} (no usable face)")

    metrics, y_true, y_score, cm = compute_metrics(results)
    print(f"Test accuracy: {metrics['accuracy']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")

    cm_path = os.path.join(config.DOCUMENTATION_DIR, "confusion_matrix.png")
    roc_path = os.path.join(config.DOCUMENTATION_DIR, "roc_curve.png")
    save_confusion_matrix_plot(cm, cm_path)
    save_roc_curve_plot(y_true, y_score, metrics["roc_auc"], roc_path)
    print(f"Saved {cm_path} and {roc_path}")

    report_md = build_evaluation_report(metrics, training_report, excluded, len(results))
    report_path = os.path.join(config.DOCUMENTATION_DIR, "evaluation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Saved {report_path}")

    error_md = build_error_analysis(results)
    error_path = os.path.join(config.DOCUMENTATION_DIR, "error_analysis.md")
    with open(error_path, "w", encoding="utf-8") as f:
        f.write(error_md)
    print(f"Saved {error_path}")


if __name__ == "__main__":
    main()
