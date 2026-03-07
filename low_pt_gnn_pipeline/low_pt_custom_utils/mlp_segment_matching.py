"""
MLP-based segment pair scoring for helix track building.

Replaces the hand-crafted Gaussian-weighted score in segment_matching.py with a
learnable MLP that takes helix parameter features from two segments and outputs a
match probability.

Feature vector per segment (8 features, normalized):
    [xc, yc, R, pitch, n_hits, inner_r, outer_r, residual_rms]
    each divided by its entry in feature_scales (configurable, sensible defaults).

Feature vector per pair (19 features = 8 + 8 + 3 derived):
    [...seg_a..., ...seg_b..., center_dist, R_ratio, |pitch_diff|]
    center_dist and pitch_diff are also scale-normalized.

Only segments with fit_quality == "good" (3+ hits, valid Kasa circle fit) are
included as matching candidates.
"""

from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

from low_pt_custom_utils.segment_matching import SegmentInfo


SEGMENT_FEATURE_DIM = 8   # xc, yc, R, pitch, n_hits, inner_r, outer_r, residual_rms
PAIR_FEATURE_DIM = 19     # 8 + 8 + 3 derived

# Default normalization scales — can be overridden via feature_scales in config/YAML.
# Keys match the feature names; derived pair features have their own entries.
DEFAULT_FEATURE_SCALES = {
    "xc":           500.0,    # circle center coordinates (mm)
    "yc":           500.0,
    "R":            1000.0,   # circle radius (mm)
    "pitch":        1.0,      # dz/ds (already ~[-0.5, 0.5])
    "nhits":        20.0,     # number of hits per segment
    "inner_r":      1000.0,   # radial extent (mm)
    "outer_r":      1000.0,
    "residual_rms": 10.0,     # Kasa circle fit residual RMS (mm)
    "center_dist":  1000.0,   # pairwise: distance between circle centers (mm)
    "pitch_diff":   1.0,      # pairwise: |pitch_a - pitch_b|
}


# ─── MLP Model ──────────────────────────────────────────────────────────────


class SegmentPairMLP(nn.Module):
    """
    MLP that scores a pair of segments as a match (output ≈ 1) or non-match (≈ 0).

    Architecture:
        BatchNorm1d(19) → [Linear → ReLU → Dropout] × n_layers → Linear(hidden, 1) → Sigmoid

    Args:
        hidden_dims: List of hidden layer widths (e.g. [64, 64]).
        dropout: Dropout probability applied after each hidden activation.
        input_dim: Feature dimension (default 19).
    """

    def __init__(
        self,
        hidden_dims: List[int] = None,
        dropout: float = 0.0,
        input_dim: int = PAIR_FEATURE_DIM,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]

        layers = [nn.BatchNorm1d(input_dim)]
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, 19) float tensor of pair features.
        Returns:
            (N, 1) match probabilities in [0, 1].
        """
        return self.net(x)


# ─── Feature Extraction ─────────────────────────────────────────────────────


def extract_segment_features(
    seg: SegmentInfo,
    scales: Optional[dict] = None,
) -> Optional[np.ndarray]:
    """
    Extract an 8-element normalized feature vector for a single segment.

    Returns None if the segment has no valid helix fit (fit_quality != "good").

    Features:
        xc            — circle center x (mm), divided by scales['xc']
        yc            — circle center y (mm), divided by scales['yc']
        R             — circle radius (mm), divided by scales['R']
        pitch         — dz/ds, divided by scales['pitch']
        n_hits        — number of hits, divided by scales['nhits']
        inner_r       — innermost hit radius (mm), divided by scales['inner_r']
        outer_r       — outermost hit radius (mm), divided by scales['outer_r']
        residual_rms  — Kasa circle fit RMS residual (mm), divided by scales['residual_rms']

    Args:
        seg:    SegmentInfo with a fitted helix.
        scales: Override dict for any normalization scale key.
                Missing keys fall back to DEFAULT_FEATURE_SCALES.
    """
    if seg.helix is None or seg.helix.fit_quality != "good":
        return None

    s = {**DEFAULT_FEATURE_SCALES, **(scales or {})}
    pitch = seg.helix.pitch if seg.helix.pitch is not None else 0.0

    return np.array([
        seg.helix.xc        / s["xc"],
        seg.helix.yc        / s["yc"],
        seg.helix.R         / s["R"],
        pitch               / s["pitch"],
        len(seg.hits)       / s["nhits"],
        seg.inner_r         / s["inner_r"],
        seg.outer_r         / s["outer_r"],
        seg.helix.residual_rms / s["residual_rms"],
    ], dtype=np.float32)


def extract_pair_features(
    seg_a: SegmentInfo,
    seg_b: SegmentInfo,
    scales: Optional[dict] = None,
) -> Optional[np.ndarray]:
    """
    Extract a 19-element feature vector for a segment pair.

    Returns None if either segment lacks a valid helix fit.

    Structure: [8 features for seg_a | 8 features for seg_b | 3 derived pairwise features]

    Derived features:
        center_dist  — Euclidean distance between circle centers, divided by scales['center_dist']
        R_ratio      — min(R_a, R_b) / max(R_a, R_b), in [0, 1]  (scale-free)
        |pitch_diff| — absolute pitch difference, divided by scales['pitch_diff']

    Args:
        seg_a, seg_b: Segments with fitted helices.
        scales:       Override dict for any normalization scale key.
    """
    fa = extract_segment_features(seg_a, scales)
    fb = extract_segment_features(seg_b, scales)
    if fa is None or fb is None:
        return None

    s = {**DEFAULT_FEATURE_SCALES, **(scales or {})}

    center_dist = np.sqrt(
        (seg_a.helix.xc - seg_b.helix.xc) ** 2 +
        (seg_a.helix.yc - seg_b.helix.yc) ** 2
    ) / s["center_dist"]

    R_max = max(seg_a.helix.R, seg_b.helix.R)
    R_min = min(seg_a.helix.R, seg_b.helix.R)
    R_ratio = float(R_min / R_max) if R_max > 1e-6 else 0.0

    pa = seg_a.helix.pitch if seg_a.helix.pitch is not None else 0.0
    pb = seg_b.helix.pitch if seg_b.helix.pitch is not None else 0.0
    pitch_diff = abs(pa - pb) / s["pitch_diff"]

    return np.concatenate([fa, fb, [center_dist, R_ratio, pitch_diff]]).astype(np.float32)


# ─── Training Data Mining ───────────────────────────────────────────────────


def build_event_training_pairs(
    graph,
    segments: List[SegmentInfo],
    neg_ratio: int = 5,
    rng: Optional[np.random.Generator] = None,
    feature_scales: Optional[dict] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Mine positive and negative segment pairs from one event using ground truth.

    A segment's "true particle" is the majority hit_particle_id among its hits,
    excluding noise (pid == 0). Segments with no signal hits or a poor/none fit
    are skipped.

    Positive pairs: both segments have the same majority particle id.
    Negative pairs: segments from different particles, subsampled to neg_ratio × n_pos.

    Args:
        graph: PyG Data object with hit_particle_id.
        segments: List of SegmentInfo (helix already fitted).
        neg_ratio: Max negatives per positive (to balance training).
        rng: Optional numpy random generator for reproducible subsampling.
        feature_scales: Override dict for normalization scales (see DEFAULT_FEATURE_SCALES).

    Returns:
        positive_features: List of (19,) arrays.
        negative_features: List of (19,) arrays.
    """
    if rng is None:
        rng = np.random.default_rng()

    particle_ids = graph.hit_particle_id.cpu().numpy()

    # Determine majority particle for each segment
    seg_particles = []
    seg_features_list = []
    for seg in segments:
        feats = extract_segment_features(seg, feature_scales)
        if feats is None:
            seg_particles.append(-1)
            seg_features_list.append(None)
            continue

        pids = particle_ids[np.array(seg.hits, dtype=int)]
        signal_pids = pids[pids > 0]
        if len(signal_pids) == 0:
            seg_particles.append(-1)
            seg_features_list.append(None)
            continue

        values, counts = np.unique(signal_pids, return_counts=True)
        majority_pid = int(values[np.argmax(counts)])
        seg_particles.append(majority_pid)
        seg_features_list.append(feats)

    positive_features = []
    negative_features = []

    n = len(segments)
    for i in range(n):
        if seg_features_list[i] is None or seg_particles[i] <= 0:
            continue
        for j in range(i + 1, n):
            if seg_features_list[j] is None or seg_particles[j] <= 0:
                continue

            pair_feats = extract_pair_features(segments[i], segments[j], feature_scales)
            if pair_feats is None:
                continue

            if seg_particles[i] == seg_particles[j]:
                positive_features.append(pair_feats)
            else:
                negative_features.append(pair_feats)

    # Subsample negatives to avoid extreme class imbalance
    n_pos = len(positive_features)
    max_neg = n_pos * neg_ratio
    if len(negative_features) > max_neg and max_neg > 0:
        indices = rng.choice(len(negative_features), size=max_neg, replace=False)
        negative_features = [negative_features[k] for k in indices]

    return positive_features, negative_features


# ─── MLP-Based Matching ──────────────────────────────────────────────────────


def match_segments_mlp(
    segments: List[SegmentInfo],
    model: SegmentPairMLP,
    config: dict,
    device: str = "cpu",
    feature_scales: Optional[dict] = None,
) -> Tuple[List[List[SegmentInfo]], List[SegmentInfo]]:
    """
    Greedy segment matching using MLP pair scores.

    Drop-in replacement for match_segments() from segment_matching.py.
    Uses the same greedy algorithm but scores pairs with the MLP instead of
    the Gaussian-weighted heuristic.

    Args:
        segments: SegmentInfo list with fitted helices.
        model: Trained SegmentPairMLP.
        config: Dict with:
            mlp_score_threshold (default 0.5): minimum MLP score for a candidate pair.
            outer_r_threshold (default 1000.0 mm): segments reaching outer detector
                are complete tracks and excluded from matching.
        device: Torch device string.
        feature_scales: Override dict for normalization scales (see DEFAULT_FEATURE_SCALES).
                        Must match the scales used during training.

    Returns:
        matched_tracks: List of [SegmentInfo, SegmentInfo] pairs that form one track.
        unmatched: All segments not consumed by a matched pair (including complete tracks).
    """
    outer_r_threshold = config.get("outer_r_threshold", 1000.0)
    score_threshold = config.get("mlp_score_threshold", 0.5)

    # Separate complete tracks (reach outer detector — no matching needed)
    complete = set()
    for i, seg in enumerate(segments):
        if seg.outer_r >= outer_r_threshold:
            complete.add(i)

    # Fittable, non-complete segments are matching candidates
    fittable = [
        (i, seg) for i, seg in enumerate(segments)
        if i not in complete and seg.helix and seg.helix.fit_quality == "good"
    ]

    # Build feature matrix for all candidate pairs
    pair_indices = []
    pair_features = []
    for ai in range(len(fittable)):
        for bi in range(ai + 1, len(fittable)):
            idx_a, seg_a = fittable[ai]
            idx_b, seg_b = fittable[bi]
            feats = extract_pair_features(seg_a, seg_b, feature_scales)
            if feats is not None:
                pair_indices.append((idx_a, idx_b))
                pair_features.append(feats)

    # Score all pairs with MLP in one batch
    candidates = []
    if pair_features:
        X = torch.tensor(np.array(pair_features), dtype=torch.float32).to(device)
        model.eval()
        with torch.no_grad():
            scores = model(X).squeeze(-1).cpu().numpy()

        for (idx_a, idx_b), score in zip(pair_indices, scores):
            if float(score) >= score_threshold:
                candidates.append((float(score), idx_a, idx_b))

    # Greedy matching: highest score first
    candidates.sort(key=lambda x: x[0], reverse=True)
    matched_set = set()
    matched_tracks = []

    for _score, idx_a, idx_b in candidates:
        if idx_a in matched_set or idx_b in matched_set:
            continue
        matched_tracks.append([segments[idx_a], segments[idx_b]])
        matched_set.add(idx_a)
        matched_set.add(idx_b)

    # Everything not in a matched pair becomes standalone
    unmatched = [seg for i, seg in enumerate(segments) if i not in matched_set]

    return matched_tracks, unmatched


# ─── Model I/O Helpers ──────────────────────────────────────────────────────


def load_mlp_model(model_path: str, mlp_config: dict = None, device: str = "cpu") -> SegmentPairMLP:
    """
    Load a trained SegmentPairMLP from a checkpoint file.

    Checkpoints saved by train_mlp_segment_matcher.py are dicts with keys:
        "state_dict", "hidden_dims", "dropout"
    Architecture is read from the checkpoint; mlp_config is ignored for new-style
    checkpoints and used only as a fallback for old plain state-dict files.

    Args:
        model_path: Path to the .pt checkpoint.
        mlp_config: Optional fallback dict with 'hidden_dims' and 'dropout'
                    (only needed for checkpoints saved before architecture embedding).
        device: Torch device string.

    Returns:
        Loaded model in eval mode on the requested device.
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        hidden_dims = checkpoint["hidden_dims"]
        dropout = checkpoint["dropout"]
        state = checkpoint["state_dict"]
        feature_scales = checkpoint.get("feature_scales")
    else:
        # Legacy plain state-dict — fall back to mlp_config
        cfg = mlp_config or {}
        hidden_dims = cfg.get("hidden_dims", [64, 64])
        dropout = cfg.get("dropout", 0.0)
        feature_scales = cfg.get("feature_scales")
        state = checkpoint

    model = SegmentPairMLP(hidden_dims=hidden_dims, dropout=dropout)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    model.feature_scales = feature_scales  # attached for downstream use
    return model
