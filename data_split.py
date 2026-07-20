"""Identity-aware train/val/test split, persisted to splits.json.

Run once; downstream steps (train.py, evaluation) read splits.json rather than
re-splitting, so re-running training doesn't reshuffle and silently create leakage.

This pools all real+fake videos from dataset/training/ and dataset/testing/
together and re-splits by IDENTITY GROUP, not by filename. A prior filename-level
split (no exact file appears twice) still let the same actor's face appear in
both train and test under a different clip — a filename-level split originally
scored a suspicious 99% test accuracy, which turned out to be inflated by
exactly this identity leakage. Groups come from identity_grouping.py: fake
videos by source actor ID (parsed from the filename), real videos by face-
embedding cluster.
"""

import json
import os
from collections import defaultdict

import config
from identity_grouping import assign_groups_to_splits, build_real_identity_clusters, get_fake_source_actor


def list_videos(directory):
    return sorted(f for f in os.listdir(directory) if f.lower().endswith(".mp4"))


def to_rel_path(directory, filename):
    return os.path.join(directory, filename).replace("\\", "/")


def all_videos(real_dirs, fake_dirs):
    real_paths = [to_rel_path(d, f) for d in real_dirs for f in list_videos(d)]
    fake_paths = [to_rel_path(d, f) for d in fake_dirs for f in list_videos(d)]
    return real_paths, fake_paths


def build_splits():
    real_dirs = [config.TRAINING_REAL_DIR, config.TESTING_REAL_DIR]
    fake_dirs = [config.TRAINING_FAKE_DIR, config.TESTING_FAKE_DIR]
    real_paths, fake_paths = all_videos(real_dirs, fake_dirs)

    # --- fake side: group by source actor (whose face is swapped into frame) ---
    fake_group_of = {p: f"fake_actor_{get_fake_source_actor(p)}" for p in fake_paths}
    fake_group_sizes = defaultdict(int)
    for g in fake_group_of.values():
        fake_group_sizes[g] += 1

    fake_targets = {"train": config.SPLIT_TARGET_COUNTS["train"],
                     "val": config.SPLIT_TARGET_COUNTS["val"],
                     "test": config.SPLIT_TARGET_COUNTS["test"]}
    fake_assignment = assign_groups_to_splits(dict(fake_group_sizes), fake_targets)

    # --- real side: group by face-embedding identity cluster ---
    print("  building real-video identity clusters (face embeddings)...")
    real_group_of = build_real_identity_clusters(real_paths)
    real_group_sizes = defaultdict(int)
    for g in real_group_of.values():
        real_group_sizes[g] += 1

    real_targets = dict(fake_targets)
    real_assignment = assign_groups_to_splits(dict(real_group_sizes), real_targets)

    splits = {name: {"real": [], "fake": []} for name in ("train", "val", "test")}
    for p in real_paths:
        split_name = real_assignment[real_group_of[p]]
        splits[split_name]["real"].append(p)
    for p in fake_paths:
        split_name = fake_assignment[fake_group_of[p]]
        splits[split_name]["fake"].append(p)

    for name in splits:
        splits[name]["real"].sort()
        splits[name]["fake"].sort()

    return splits, fake_group_of, real_group_of


def assert_no_leakage(splits):
    """No video filename may appear in more than one split — the baseline
    correctness check. Identity-level leakage (the actual concern that
    motivated this rewrite) is checked separately in assert_no_identity_leakage,
    since it requires the group-id information, not just filenames."""

    def basenames(split):
        return {os.path.basename(p) for p in split["real"] + split["fake"]}

    train_names = basenames(splits["train"])
    val_names = basenames(splits["val"])
    test_names = basenames(splits["test"])

    assert not (train_names & val_names), "leakage between train and val"
    assert not (train_names & test_names), "leakage between train and test"
    assert not (val_names & test_names), "leakage between val and test"


def assert_no_identity_leakage(splits, fake_group_of, real_group_of):
    """No identity group (source actor or real-face cluster) may span more than
    one split — this is the check that actually matters for this dataset."""

    def group_ids(split, group_map):
        return {group_map[p] for p in split if p in group_map}

    fake_groups_by_split = {name: group_ids(splits[name]["fake"], fake_group_of) for name in splits}
    real_groups_by_split = {name: group_ids(splits[name]["real"], real_group_of) for name in splits}

    names = list(splits.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            assert not (fake_groups_by_split[a] & fake_groups_by_split[b]), \
                f"fake source-actor leakage between {a} and {b}"
            assert not (real_groups_by_split[a] & real_groups_by_split[b]), \
                f"real identity-cluster leakage between {a} and {b}"


def save_splits(splits, path=config.SPLITS_JSON_PATH):
    with open(path, "w") as f:
        json.dump(splits, f, indent=2)


def load_splits(path=config.SPLITS_JSON_PATH):
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    splits, fake_group_of, real_group_of = build_splits()
    assert_no_leakage(splits)
    assert_no_identity_leakage(splits, fake_group_of, real_group_of)
    save_splits(splits)

    for name in ("train", "val", "test"):
        n_real = len(splits[name]["real"])
        n_fake = len(splits[name]["fake"])
        print(f"{name}: {n_real} real + {n_fake} fake = {n_real + n_fake} videos")
