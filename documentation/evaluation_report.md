# Evaluation Report

## Test-set video-level metrics

- Accuracy: 0.9231
- ROC-AUC: 0.9684
- Precision (REAL / FAKE): 0.9038 / 0.9487
- Recall (REAL / FAKE): 0.9592 / 0.8810
- F1 (REAL / FAKE): 0.9307 / 0.9136
- Support (REAL / FAKE): 49 / 42
- Videos scored: 91 (of 92 in test/); 1 excluded — no usable face in any sampled frame
  - Excluded: dataset/training/real/030.mp4

## Cross-validation results (from training, train split only)

Selected model: **svm**

- random_forest: CV F1 = 0.9382 +/- 0.0096, best params = {'classifier__max_depth': None, 'classifier__min_samples_leaf': 1, 'classifier__n_estimators': 300}
- gradient_boosting: CV F1 = 0.9386 +/- 0.0154, best params = {'classifier__learning_rate': 0.1, 'classifier__max_depth': 4, 'classifier__n_estimators': 200}
- svm (selected): CV F1 = 0.9541 +/- 0.0121, best params = {'classifier__C': 10, 'classifier__gamma': 0.01}
- mlp_baseline (not eligible for selection): val accuracy = 0.9081
- Selected model val accuracy (held out during CV): 0.9264

## Leakage sanity check

Test accuracy is 92.3%, at or above the project's 90% self-check threshold (see CLAUDE.md). Two leakage mechanisms were found and fixed earlier in this project: CV-fold leakage in training (step 3) and actor/identity leakage in the original filename-level split, where the same actor's face appeared in both train and test under different clips (step 4 — this is what the 99%->92% jump reflects). The current split is identity-group-disjoint on both sides: fake videos by source actor (parsed from filenames), real videos by face-embedding cluster (cosine similarity > 0.5).

**One leakage mechanism remains unresolved, not just unverified:** whether the same physical person appears as both a REAL video and as a FAKE video's source actor in different splits. Cross-matching the 6 known fake source actors against all 250 real-video embeddings was attempted and came back inconclusive (best-match cosine similarities of 0.36-0.55, indistinguishable from noise — GAN face-swap alters the embedding enough that this comparison isn't reliable with the tools used here). This test accuracy should be read with that gap in mind, not treated as a fully leakage-proof number.
