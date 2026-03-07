"""
Utility functions for the custom loop track builder.

Contains helper functions for:
- Layer detection from hit_r values
- Edge/adjacency lookup utilities
- Connected Components clustering
- Problem cluster identification
- Outward segment building (tree exploration)
- Conflict resolution between segments
- Loop segment matching

Layer detection: Uses hit_r values with 10mm tolerance to group hits into layers.
    TODO: For exact layer IDs, add hit_layer_disk to PyG graphs by adding it to
    hit_features in acorn_configs/convert_csv_to_pyg_sets.yaml. The attribute
    already exists in ActsReader (acorn/stages/data_reading/models/acts_reader.py:151).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple
from time import perf_counter

import numpy as np
import torch
import scipy.sparse as sps
from torch_geometric.utils import remove_isolated_nodes, to_scipy_sparse_matrix

# Global flag for verbose timing (set by caller)
_VERBOSE_TIMING = False

def set_verbose_timing(verbose: bool):
    """Enable or disable verbose timing output."""
    global _VERBOSE_TIMING
    _VERBOSE_TIMING = verbose


# ─── Data Structures ────────────────────────────────────────────────────────

@dataclass
class Segment:
    """A candidate outward-going track segment."""
    hits: List[int]                          # Ordered node indices (increasing layer)
    start_hit: int                           # First-layer starting hit
    denied_edges: Set[Tuple[int, int]] = field(default_factory=set)  # (src, dst) pairs already rejected


# ─── Layer Detection ────────────────────────────────────────────────────────

def assign_layers(hit_r, tolerance=10.0):
    """
    Assign detector layer IDs based on hit_r values.
    Hits within `tolerance` mm of each other are considered same layer.

    Returns:
        layer_ids: np.ndarray of integer layer IDs per hit
        unique_layers: sorted list of unique layer IDs
    """
    r_values = hit_r.cpu().numpy() if isinstance(hit_r, torch.Tensor) else np.array(hit_r)

    # Sort unique radii and group into layers
    sorted_r = np.sort(np.unique(r_values))
    layer_boundaries = []
    current_layer_r = sorted_r[0]
    layer_boundaries.append(current_layer_r)

    for r in sorted_r[1:]:
        if r - current_layer_r > tolerance:
            layer_boundaries.append(r)
            current_layer_r = r
        else:
            # Update representative radius to be the latest in the cluster
            current_layer_r = r

    layer_boundaries = np.array(layer_boundaries)

    # Assign each hit to nearest layer boundary
    layer_ids = np.zeros(len(r_values), dtype=int)
    for i, r in enumerate(r_values):
        layer_ids[i] = np.argmin(np.abs(layer_boundaries - r))

    return layer_ids, np.sort(np.unique(layer_ids))


# ─── Edge Lookup Utilities ──────────────────────────────────────────────────

def build_adjacency(graph, score_cut):
    """
    Build adjacency dict from edge_index filtered by score_cut.

    Returns:
        adj: dict {src_node: [(dst_node, score), ...]}
        adj_reverse: dict {dst_node: [(src_node, score), ...]}
    """
    edge_index = graph.edge_index.cpu().numpy()
    scores = graph.edge_scores.cpu().numpy()

    mask = scores > score_cut
    src = edge_index[0, mask]
    dst = edge_index[1, mask]
    sc = scores[mask]

    adj = {}
    adj_reverse = {}
    for s, d, score in zip(src, dst, sc):
        adj.setdefault(int(s), []).append((int(d), float(score)))
        adj_reverse.setdefault(int(d), []).append((int(s), float(score)))
        # Undirected: add reverse edge too
        adj.setdefault(int(d), []).append((int(s), float(score)))
        adj_reverse.setdefault(int(s), []).append((int(d), float(score)))

    return adj, adj_reverse


def build_full_adjacency(graph):
    """
    Build adjacency dict from ALL edges (no score filtering).
    Used for weak edge matching in loop segment connection.

    Returns:
        adj_full: dict {src_node: [(dst_node, score), ...]}
    """
    edge_index = graph.edge_index.cpu().numpy()
    scores = graph.edge_scores.cpu().numpy()

    adj_full = {}
    for s, d, score in zip(edge_index[0], edge_index[1], scores):
        adj_full.setdefault(int(s), []).append((int(d), float(score)))
        # Undirected: add reverse edge too
        adj_full.setdefault(int(d), []).append((int(s), float(score)))

    return adj_full


def get_edge_score_from_adj(adj, src, dst):
    """Look up edge score between src and dst."""
    for neighbor, score in adj.get(src, []):
        if neighbor == dst:
            return score
    return 0.0


# ─── Step 0: Initial Connected Components ───────────────────────────────────

def initial_cc_clustering(graph, score_cut):
    """
    Run standard Connected Components clustering.
    Reuses logic from ACORN's ConnectedComponents._build_event().
    """
    edge_mask = graph.edge_scores > score_cut
    edges = graph.edge_index[:, edge_mask]

    num_nodes = graph.hit_x.size(0) if hasattr(graph, "hit_x") else graph.edge_index.max().item() + 1

    edges_clean, _, mask = remove_isolated_nodes(edges, num_nodes=num_nodes)
    sparse_edges = to_scipy_sparse_matrix(edges_clean, num_nodes=mask.sum().item())

    _, candidate_labels = sps.csgraph.connected_components(
        sparse_edges, directed=False, return_labels=True
    )

    labels = (torch.ones(num_nodes) * -1).long()
    labels[mask] = torch.tensor(candidate_labels, dtype=torch.long)

    return labels


# ─── Step 1: Identify Problem Clusters ──────────────────────────────────────

def identify_problem_clusters(labels, layer_ids, hit_r, outer_r_threshold=1000.0):
    """
    Find clusters that need disentangling.

    Simple clusters (kept as-is):
    1. Single hit on first layer AND reaches outer radius (straight track through detector)
    2. Two hits on first layer AND does NOT reach outer radius (perfect loop, curves back)

    Problem clusters (need disentangling):
    1. More than 2 hits on first layer (merged looping trajectories)
    2. Single hit on first layer AND doesn't reach outer radius (incomplete track)
    3. Two hits on first layer AND reaches outer radius (ambiguous, possibly two merged tracks)

    Args:
        labels: cluster labels per hit
        layer_ids: layer ID per hit
        hit_r: radial coordinate per hit
        outer_r_threshold: radius threshold for considering track as exited (default 1000 mm)

    Returns:
        problem_clusters: list of cluster IDs that need processing
        simple_clusters: list of cluster IDs that are fine as-is
    """
    unique_clusters = torch.unique(labels)
    problem_clusters = []
    simple_clusters = []

    for cluster_id in unique_clusters:
        if cluster_id.item() == -1:
            continue

        cluster_mask = labels == cluster_id.item()
        cluster_indices = torch.where(cluster_mask)[0].numpy()
        cluster_layers = layer_ids[cluster_indices]
        cluster_r = hit_r[cluster_indices]

        # Find innermost layer and outermost radius
        min_layer = cluster_layers.min()
        first_layer_count = (cluster_layers == min_layer).sum()
        max_r = cluster_r.max()

        # Simple cluster conditions
        if first_layer_count == 1 and max_r >= outer_r_threshold:
            # Simple case 1: Straight track (1 first-layer hit, reaches outer radius)
            simple_clusters.append(cluster_id.item())
        elif first_layer_count == 2 and max_r < outer_r_threshold:
            # Simple case 2: Perfect loop (2 first-layer hits, doesn't reach outer radius)
            simple_clusters.append(cluster_id.item())
        else:
            # Problem clusters: everything else
            # - first_layer_count > 2: merged trajectories
            # - first_layer_count == 1 and max_r < outer_r_threshold: incomplete track
            # - first_layer_count == 2 and max_r >= outer_r_threshold: ambiguous/merged tracks
            problem_clusters.append(cluster_id.item())

    return problem_clusters, simple_clusters


# ─── Step 2: Build Outward Segments ─────────────────────────────────────────

def build_outward_segments(adj, cluster_indices, first_layer_hits, layer_ids, max_branching_factor=3,
                          hit_x=None, hit_y=None, hit_z=None):
    """
    Build outward-going track segments from each first-layer hit using tree
    exploration. All paths are built simultaneously, then the longest/best
    path is selected per starting hit.

    Edge selection uses spatial-aware sorting:
    - Primary criterion: edge score (higher is better)
    - Secondary criterion: spatial proximity (physically closer hits preferred as tie-breaker)
    This helps avoid shortcuts (e.g., prefers 1→2→3 over 1→3 when scores are similar)

    Args:
        adj: adjacency dict {node: [(neighbor, score), ...]}
        cluster_indices: set of node indices in this cluster
        first_layer_hits: list of node indices on the first layer
        layer_ids: array of layer IDs per node
        max_branching_factor: maximum number of neighbors to explore per node
            (None or inf = explore all neighbors)
        hit_x, hit_y, hit_z: optional hit coordinates for spatial distance calculation
            If not provided, falls back to layer-based distance

    Returns:
        segments: list of Segment objects (one per first-layer hit)
    """
    t_func_start = perf_counter()
    segments = []

    for start_idx, start_hit in enumerate(first_layer_hits):
        # Tree exploration: BFS building all candidate paths
        # Each path is a tuple of node indices
        active_paths = [(start_hit,)]
        completed_paths = []

        while active_paths:
            new_paths = []
            for path in active_paths:
                current_node = path[-1]
                current_layer = layer_ids[current_node]

                # Find neighbors at higher layer within cluster
                candidates = []
                for neighbor, score in adj.get(current_node, []):
                    if (neighbor in cluster_indices
                            and layer_ids[neighbor] > current_layer
                            and neighbor not in path):
                        candidates.append((neighbor, score))

                if not candidates:
                    # Dead end - path is complete
                    completed_paths.append(path)
                else:
                    # Prune: keep only top-N highest-scoring neighbors
                    # Spatial-aware sorting: prefer physically closer hits when scores are similar
                    if max_branching_factor is not None and max_branching_factor > 0:
                        # Use spatial distance if coordinates available, else fallback to layer distance
                        if hit_x is not None and hit_y is not None and hit_z is not None:
                            # Spatial distance tie-breaker (3D Euclidean distance)
                            def calc_spatial_distance(neighbor_idx):
                                dx = hit_x[neighbor_idx] - hit_x[current_node]
                                dy = hit_y[neighbor_idx] - hit_y[current_node]
                                dz = hit_z[neighbor_idx] - hit_z[current_node]
                                return (dx*dx + dy*dy + dz*dz) ** 0.5

                            candidates.sort(
                                key=lambda x: (
                                    x[1],  # Primary: edge score (higher is better)
                                    -calc_spatial_distance(x[0])  # Secondary: spatial proximity (closer is better)
                                ),
                                reverse=True
                            )
                        else:
                            # Fallback to layer distance
                            candidates.sort(
                                key=lambda x: (
                                    x[1],  # Primary: edge score (higher is better)
                                    -(layer_ids[x[0]] - current_layer)  # Secondary: layer proximity (closer is better)
                                ),
                                reverse=True
                            )
                        candidates = candidates[:max_branching_factor]

                    # Extend path with selected candidates
                    for neighbor, score in candidates:
                        new_paths.append(path + (neighbor,))

            active_paths = new_paths

        # Pick the longest path (break ties by cumulative score)
        if completed_paths:
            best_path = max(
                completed_paths,
                key=lambda p: (len(p), sum(
                    get_edge_score_from_adj(adj, p[i], p[i + 1])
                    for i in range(len(p) - 1)
                ))
            )
            segments.append(Segment(
                hits=list(best_path),
                start_hit=start_hit,
            ))
        else:
            # Only the starting hit itself
            segments.append(Segment(
                hits=[start_hit],
                start_hit=start_hit,
            ))

    return segments


# ─── Step 3: Resolve Conflicts ──────────────────────────────────────────────

def find_conflicts(segments):
    """
    Find spacepoints shared between segments.

    Returns:
        conflicts: list of (shared_hit, seg_idx_a, seg_idx_b)
    """
    hit_to_segments = {}
    for seg_idx, seg in enumerate(segments):
        for hit in seg.hits:
            hit_to_segments.setdefault(hit, []).append(seg_idx)

    conflicts = []
    for hit, seg_indices in hit_to_segments.items():
        if len(seg_indices) > 1:
            for i in range(len(seg_indices)):
                for j in range(i + 1, len(seg_indices)):
                    conflicts.append((hit, seg_indices[i], seg_indices[j]))

    return conflicts


def resolve_conflicts(segments, adj, cluster_indices, layer_ids, max_iterations=100):
    """
    Iteratively resolve shared spacepoints between segments.

    When two segments share a hit:
    - Compare edge scores leading TO that hit
    - Winner keeps the hit
    - Loser backtracks and takes second-best edge
    - Loser cannot use any node from winner's segment
    """
    for iteration in range(max_iterations):
        conflicts = find_conflicts(segments)
        if not conflicts:
            break

        # Process first conflict (lowest layer shared hit)
        conflicts.sort(key=lambda c: layer_ids[c[0]])
        shared_hit, idx_a, idx_b = conflicts[0]

        seg_a = segments[idx_a]
        seg_b = segments[idx_b]

        # Find position of shared hit in each segment
        pos_a = seg_a.hits.index(shared_hit)
        pos_b = seg_b.hits.index(shared_hit)

        # Determine winner by edge score leading TO the shared hit
        score_a = get_edge_score_from_adj(adj, seg_a.hits[pos_a - 1], shared_hit) if pos_a > 0 else 0.0
        score_b = get_edge_score_from_adj(adj, seg_b.hits[pos_b - 1], shared_hit) if pos_b > 0 else 0.0

        if score_a >= score_b:
            winner_seg, loser_seg = seg_a, seg_b
            loser_pos = pos_b
        else:
            winner_seg, loser_seg = seg_b, seg_a
            loser_pos = pos_a

        # Loser backtracks to node before shared hit
        if loser_pos <= 0:
            # Loser starts at the shared hit - just remove it
            loser_seg.hits = []
            continue

        backtrack_node = loser_seg.hits[loser_pos - 1]
        # Record denied edge
        loser_seg.denied_edges.add((backtrack_node, shared_hit))

        # Truncate loser to before the conflict
        loser_seg.hits = loser_seg.hits[:loser_pos]

        # Winner's nodes are forbidden
        winner_nodes = set(winner_seg.hits)

        # Try to rebuild loser from backtrack point
        _rebuild_segment_from(
            loser_seg, backtrack_node, adj, cluster_indices, layer_ids, winner_nodes
        )

    return [seg for seg in segments if len(seg.hits) > 0]


def _rebuild_segment_from(segment, from_node, adj, cluster_indices, layer_ids, forbidden_nodes):
    """
    Rebuild a segment from a given node, avoiding forbidden nodes and denied edges.
    Greedy: always picks highest-scoring valid neighbor.
    """
    current_node = from_node

    while True:
        current_layer = layer_ids[current_node]

        # Find valid candidates
        candidates = []
        for neighbor, score in adj.get(current_node, []):
            if (neighbor in cluster_indices
                    and layer_ids[neighbor] > current_layer
                    and neighbor not in forbidden_nodes
                    and neighbor not in segment.hits
                    and (current_node, neighbor) not in segment.denied_edges):
                candidates.append((neighbor, score))

        if not candidates:
            break

        # Pick highest scoring
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_neighbor, _ = candidates[0]
        segment.hits.append(best_neighbor)
        current_node = best_neighbor


# ─── Step 4: Match Loop Segments ────────────────────────────────────────────

def match_loop_segments(segments, adj, adj_full, layer_ids, score_cut, outer_r_threshold=1000.0,
                       hit_r=None, hit_phi=None, check_verbose=False):
    """
    Match inward and outward segments of looping particles using global edge-based matching.

    This function receives segments from ALL problem clusters, allowing cross-cluster matching.

    Args:
        segments: List of Segment objects from all problem clusters
        adj: Adjacency dict with edges filtered by score_cut (for Priority 2)
        adj_full: Adjacency dict with ALL edges (for Priority 3 weak matching)
        layer_ids: Array of layer IDs per node
        score_cut: Score threshold for strong edge matching
        outer_r_threshold: Radius threshold for straight tracks
        hit_r, hit_phi: Hit coordinates
        check_verbose: Print matching statistics

    Priority 1: Straight tracks reaching R > outer_r_threshold are standalone
    Priority 2: Match via direct edge (score > score_cut) at same layer
    Priority 3: Match by edge score (even < score_cut), highest scores first
    Priority 4: Leftovers without any edge connection remain as individual tracks

    Returns:
        final_tracks: list of lists of node indices
        track_priorities: list of int (priority for each track: 1=P1, 2=P2, 3=P3, 4=P4)
        stats: dict with keys n_priority1, n_priority2, n_priority3, n_priority4, total_segments
    """
    if len(segments) == 0:
        return [], [], {'n_priority1': 0, 'n_priority2': 0, 'n_priority3': 0, 'n_priority4': 0, 'total_segments': 0}

    matched = set()
    final_tracks = []
    track_priorities = []  # Track which priority each track came from

    # Get endpoint info for each segment
    endpoint_info = []
    for i, seg in enumerate(segments):
        if len(seg.hits) == 0:
            continue
        last_hit = seg.hits[-1]
        last_layer = layer_ids[last_hit]
        last_r = hit_r[last_hit].item() if hit_r is not None else 0.0
        last_phi = hit_phi[last_hit].item() if hit_phi is not None else 0.0
        endpoint_info.append({
            'seg_idx': i,
            'last_hit': last_hit,
            'last_layer': last_layer,
            'last_r': last_r,
            'last_phi': last_phi,
        })

    # Priority 1: Straight tracks (R > threshold)
    n_priority1 = 0
    for info in endpoint_info:
        if info['seg_idx'] in matched:
            continue
        if info['last_r'] > outer_r_threshold:
            final_tracks.append(segments[info['seg_idx']].hits)
            track_priorities.append(1)  # Priority 1
            matched.add(info['seg_idx'])
            n_priority1 += 1

    # Priority 2: Match via direct edge with high score at same layer
    match_candidates = []
    for i, info_a in enumerate(endpoint_info):
        if info_a['seg_idx'] in matched:
            continue
        for j, info_b in enumerate(endpoint_info):
            if j <= i or info_b['seg_idx'] in matched:
                continue
            if info_a['last_layer'] != info_b['last_layer']:
                continue
            # Check if direct edge exists
            score = get_edge_score_from_adj(adj, info_a['last_hit'], info_b['last_hit'])
            if score > score_cut:
                match_candidates.append((score, info_a['seg_idx'], info_b['seg_idx']))

    # Match highest scoring pairs first
    match_candidates.sort(reverse=True)
    n_priority2 = 0
    for score, idx_a, idx_b in match_candidates:
        if idx_a in matched or idx_b in matched:
            continue
        # Combine segments into one track
        combined = segments[idx_a].hits + list(reversed(segments[idx_b].hits))
        final_tracks.append(combined)
        track_priorities.append(2)  # Priority 2
        matched.add(idx_a)
        matched.add(idx_b)
        n_priority2 += 1

    # Priority 3: Match by edge score (even below score_cut)
    # Optimization: group endpoints by layer, only check same-layer pairs
    endpoints_by_layer = {}
    for info in endpoint_info:
        if info['seg_idx'] not in matched:
            endpoints_by_layer.setdefault(info['last_layer'], []).append(info)

    match_candidates_weak = []
    for layer, infos in endpoints_by_layer.items():
        # Only check pairs within the same layer
        for i in range(len(infos)):
            for j in range(i + 1, len(infos)):
                info_a, info_b = infos[i], infos[j]
                if info_a['seg_idx'] in matched or info_b['seg_idx'] in matched:
                    continue
                # Check if any edge exists between endpoints (using full adjacency, no filtering)
                score = get_edge_score_from_adj(adj_full, info_a['last_hit'], info_b['last_hit'])
                if score > 0:  # Any edge exists (even weak ones < score_cut)
                    match_candidates_weak.append((score, info_a['seg_idx'], info_b['seg_idx']))

    # Match highest scoring pairs first (greedy matching)
    match_candidates_weak.sort(reverse=True)
    n_priority3 = 0
    for score, idx_a, idx_b in match_candidates_weak:
        if idx_a in matched or idx_b in matched:
            continue
        # Combine segments into one track
        combined = segments[idx_a].hits + list(reversed(segments[idx_b].hits))
        final_tracks.append(combined)
        track_priorities.append(3)  # Priority 3
        matched.add(idx_a)
        matched.add(idx_b)
        n_priority3 += 1

    # Priority 4: Leftovers without any edge connection remain as individual tracks
    n_priority4 = 0
    for info in endpoint_info:
        if info['seg_idx'] not in matched:
            seg = segments[info['seg_idx']]
            if len(seg.hits) > 0:
                final_tracks.append(seg.hits)
                track_priorities.append(4)  # Priority 4
                n_priority4 += 1

    if check_verbose:
        total_segments = len([s for s in segments if len(s.hits) > 0])
        parts = []
        if n_priority1 > 0:
            parts.append(f"P1:{n_priority1}")
        if n_priority2 > 0:
            parts.append(f"P2:{n_priority2}")
        if n_priority3 > 0:
            parts.append(f"P3:{n_priority3}")
        if n_priority4 > 0:
            parts.append(f"P4:{n_priority4}")
        print(f"  Matched segments: {' '.join(parts)} (total {total_segments} segments)")

    # Return tracks, priorities, and statistics
    stats = {
        'n_priority1': n_priority1,
        'n_priority2': n_priority2,
        'n_priority3': n_priority3,
        'n_priority4': n_priority4,
        'total_segments': len([s for s in segments if len(s.hits) > 0]),
    }
    return final_tracks, track_priorities, stats
