"""Identity-level grouping for a leakage-safe split.

Filename-level video-level splitting (data_split.py's original approach) is not
sufficient for this dataset: the same actor appears in multiple distinct video
files. Two identity domains, grouped differently:

- FAKE videos are named "{source_actor}_{target_actor}__scenario__hash.mp4".
  The source actor's face is what's swapped into the frame (face-swap
  convention), so fake videos are grouped by source actor ID.
- REAL videos have no identity encoded in the filename, so identity is
  estimated via face-embedding clustering (facenet-pytorch InceptionResnetV1,
  vggface2 weights). Two real videos are treated as the same identity if their
  face embeddings' cosine similarity exceeds CLUSTER_THRESHOLD.

Cross-matching fake source actors against real-video clusters was tried and
found inconclusive (GAN face-swap alters the embedding enough that similarity
scores for known-same actors were in the same 0.3-0.5 range as unrelated
pairs) — real-to-fake identity leakage is a known, unresolved limitation, not
a solved problem.
"""

import os
import re

import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

from face_utils import detect_and_filter_face

CLUSTER_THRESHOLD = 0.5
EMBEDDING_CACHE_PATH = "feature_cache/real_identity_embeddings.npz"

_resnet = None


def _get_resnet():
    global _resnet
    if _resnet is None:
        _resnet = InceptionResnetV1(pretrained="vggface2").eval()
    return _resnet


def get_fake_source_actor(video_path):
    """Parses "{source}_{target}__..." from the filename; returns None if the
    filename doesn't match (defensive — every deepfake file in this dataset
    matches, verified during the leakage investigation)."""
    fname = os.path.basename(video_path)
    m = re.match(r"^(\d+)_(\d+)__", fname)
    return m.group(1) if m else None


def compute_face_embedding(video_path):
    """First usable face among the video's first few frames -> 512-d embedding.
    Returns None if no usable face is found at all."""
    resnet = _get_resnet()
    cap = cv2.VideoCapture(video_path)
    for _ in range(5):
        ok, frame = cap.read()
        if not ok:
            break
        face = detect_and_filter_face(frame)
        if face is not None:
            cap.release()
            face_rgb = cv2.cvtColor(cv2.resize(face, (160, 160)), cv2.COLOR_BGR2RGB)
            tensor = torch.tensor(face_rgb).permute(2, 0, 1).float()
            tensor = (tensor - 127.5) / 128.0
            with torch.no_grad():
                emb = resnet(tensor.unsqueeze(0))
            return emb.squeeze(0).numpy()
    cap.release()
    return None


def _cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def build_real_identity_clusters(real_video_paths, cache_path=EMBEDDING_CACHE_PATH):
    """Returns {video_path: cluster_id}. Videos with no usable face each get
    their own singleton cluster (no evidence they share identity with anything
    else, so treated conservatively as unique)."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        embeddings = {p: e for p, e in zip(data["paths"], data["embeddings"])}
    else:
        embeddings = {}
        for i, path in enumerate(real_video_paths):
            emb = compute_face_embedding(path)
            if emb is not None:
                embeddings[path] = emb
            if (i + 1) % 50 == 0:
                print(f"    ...{i + 1}/{len(real_video_paths)} real videos embedded")
        paths_arr = np.array(list(embeddings.keys()))
        emb_arr = np.array(list(embeddings.values()))
        np.savez(cache_path, paths=paths_arr, embeddings=emb_arr)

    parent = {p: p for p in real_video_paths}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    paths = [p for p in real_video_paths if p in embeddings]
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if _cosine_sim(embeddings[paths[i]], embeddings[paths[j]]) > CLUSTER_THRESHOLD:
                union(paths[i], paths[j])

    return {p: f"real_{find(p)}" for p in real_video_paths}


def assign_groups_to_splits(group_sizes, target_counts):
    """Deterministic greedy bin-packing: largest group first, assigned to
    whichever split has the largest remaining quota deficit. Same rule used
    for both the real (many small clusters) and fake (6 actor groups) domains,
    so there's one simple algorithm to explain rather than two ad hoc ones."""
    groups_sorted = sorted(group_sizes.items(), key=lambda kv: (-kv[1], kv[0]))
    remaining = dict(target_counts)
    assignment = {}
    for group_id, size in groups_sorted:
        best_split = max(remaining, key=lambda s: remaining[s])
        assignment[group_id] = best_split
        remaining[best_split] -= size
    return assignment
