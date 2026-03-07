"""
Replacement for acorn's build_edges() that correctly returns the K CLOSEST
neighbors within r_max, rather than arbitrary neighbors within the ball.

The PyG radius() fallback used by acorn when FRNN is unavailable does NOT
guarantee returning the closest neighbors — its CPU implementation is biased
towards certain quadrants (documented PyG warning).

Embeddings from the metric learning model are L2-normalized to a unit sphere
via F.normalize(). For unit vectors, cosine similarity = dot product = a·b,
and Euclidean distance = sqrt(2 - 2*(a·b)). So finding the K closest neighbors
is equivalent to finding K largest dot products — computed via matmul, which
avoids materialising a full N×N distance matrix.

The r_max threshold converts as: dist <= r_max  ↔  sim >= 1 - r_max²/2
"""
import torch
from typing import Optional


def build_edges(
    query: torch.Tensor,
    database: torch.Tensor,
    indices: Optional[torch.Tensor] = None,
    r_max: float = 1.0,
    k_max: int = 10,
    return_indices: bool = False,
    backend: str = "FRNN",  # kept for drop-in compatibility, ignored here
) -> torch.Tensor:
    """
    Find up to k_max closest neighbors in `database` for each point in `query`,
    within Euclidean radius r_max. Returns edge_list[0]=query idx, edge_list[1]=db idx.

    Uses matmul on L2-normalized embeddings (unit sphere), so only the top-k
    slice [N_q × k_max] is stored rather than the full [N_q × N_d] distance matrix.

    Drop-in replacement for acorn's build_edges() without requiring FRNN.
    """
    N_q = query.shape[0]
    N_d = database.shape[0]
    k_actual = min(k_max, N_d)

    # For L2-normalized vectors: dist² = 2 - 2*(a·b)
    # So K closest ↔ K largest dot products. Matmul avoids full N×N dist matrix.
    sim = query @ database.T  # [N_q, N_d] cosine similarities

    # Top-k most similar (= closest) neighbors per query point
    topk_sim, topk_idx = torch.topk(sim, k=k_actual, dim=1, largest=True)

    # Convert similarity threshold: dist <= r_max  ↔  sim >= 1 - r_max²/2
    sim_min = 1.0 - (r_max ** 2) / 2.0
    mask = topk_sim >= sim_min  # [N_q, k_actual]

    # Build edge list: [0] = query indices, [1] = database indices
    src = torch.arange(N_q, device=query.device).unsqueeze(1).expand(-1, k_actual)
    edge_list = torch.stack([src[mask], topk_idx[mask]])

    # Reset query indices to global index if a subset index mapping is provided
    if indices is not None:
        edge_list[0] = indices[edge_list[0]]

    # Remove self-loops
    edge_list = edge_list[:, edge_list[0] != edge_list[1]]

    return edge_list


def patch_acorn_build_edges():
    """
    Monkey-patch acorn's metric_learning module to use this build_edges.

    Call this AFTER importing acorn but BEFORE training starts.
    Needed because metric_learning.py calls build_edges internally during
    training and validation steps (lines 154 and 369 in metric_learning.py).

    Usage in training script:
        from low_pt_custom_utils.graph_utils import patch_acorn_build_edges
        patch_acorn_build_edges()
    """
    import acorn.stages.graph_construction.models.metric_learning as _ml
    _ml.build_edges = build_edges
