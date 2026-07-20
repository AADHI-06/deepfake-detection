# Error Analysis

7 of 91 test videos misclassified (7.7%).

Each entry lists automated proxy signals only (vote confidence, how many of the sampled frames were discarded by face-detection/blur filtering, and how much per-frame predictions disagreed with each other). These are measurable from the pipeline's own output, not a human review of the video content — a real error-analysis pass should also involve watching these clips.

## dataset/testing/real/234.mp4
- True: REAL, Predicted: FAKE, confidence: 0.986
- Frames used in vote: 10/10
- Candidate reasons: confidently wrong — no obvious pipeline-level explanation from these proxy signals; needs manual review of the clip

## dataset/training/real/193.mp4
- True: REAL, Predicted: FAKE, confidence: 0.645
- Frames used in vote: 5/10
- Candidate reasons: 5/10 sampled frames discarded by face-detection/blur filtering — fewer frames contributed to the vote

## dataset/training/deepfake/01_02__walk_down_hall_angry__YVGY8LOK.mp4
- True: FAKE, Predicted: REAL, confidence: 0.566
- Frames used in vote: 10/10
- Candidate reasons: near-threshold vote confidence — model was genuinely uncertain; high disagreement between individual frame predictions (possible inconsistent visibility/pose across the video)

## dataset/training/deepfake/01_12__outside_talking_pan_laughing__TNI7KUZ6.mp4
- True: FAKE, Predicted: REAL, confidence: 0.862
- Frames used in vote: 9/10
- Candidate reasons: 1/10 sampled frames discarded by face-detection/blur filtering — fewer frames contributed to the vote; high disagreement between individual frame predictions (possible inconsistent visibility/pose across the video)

## dataset/training/deepfake/01_12__outside_talking_still_laughing__TNI7KUZ6.mp4
- True: FAKE, Predicted: REAL, confidence: 0.869
- Frames used in vote: 10/10
- Candidate reasons: confidently wrong — no obvious pipeline-level explanation from these proxy signals; needs manual review of the clip

## dataset/training/deepfake/01_20__hugging_happy__FW94AIMJ.mp4
- True: FAKE, Predicted: REAL, confidence: 0.541
- Frames used in vote: 10/10
- Candidate reasons: near-threshold vote confidence — model was genuinely uncertain; high disagreement between individual frame predictions (possible inconsistent visibility/pose across the video)

## dataset/training/deepfake/01_20__walk_down_hall_angry__FW94AIMJ.mp4
- True: FAKE, Predicted: REAL, confidence: 0.545
- Frames used in vote: 10/10
- Candidate reasons: near-threshold vote confidence — model was genuinely uncertain; high disagreement between individual frame predictions (possible inconsistent visibility/pose across the video)
