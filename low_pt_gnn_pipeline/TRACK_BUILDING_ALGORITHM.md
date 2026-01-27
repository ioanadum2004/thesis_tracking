# Track Building Algorithm: Connected Components

## Overview

The `build_tracks.py` script uses the **Connected Components** algorithm to build candidate tracks from graphs with classified edges. This is a simple but effective graph-based track reconstruction method.

## Algorithm Steps

### Input
- **Graphs with edge scores**: Each graph contains:
  - Nodes: Spacepoints (hits) with coordinates (r, φ, z)
  - Edges: Connections between spacepoints
  - Edge scores: Probability that each edge belongs to a true track (from GNN classifier)

### Step 1: Apply Score Cut
```python
edge_mask = graph.edge_scores > score_cut  # Default: 0.5
edges = graph.edge_index[:, edge_mask]
```

**What happens:**
- Filters edges based on the `score_cut` threshold (default: 0.5)
- Only keeps edges with `edge_scores > score_cut`
- This removes low-confidence edges that are likely noise/fake connections

**Example:**
- If an edge has `edge_scores = 0.7`, it's kept (0.7 > 0.5)
- If an edge has `edge_scores = 0.3`, it's removed (0.3 < 0.5)

### Step 2: Remove Isolated Nodes
```python
edges, _, mask = remove_isolated_nodes(edges, num_nodes=num_nodes)
```

**What happens:**
- Removes nodes (hits) that have no connections after filtering
- These isolated hits cannot form tracks, so they're excluded

### Step 3: Convert to Sparse Graph
```python
sparse_edges = to_scipy_sparse_matrix(edges, num_nodes=mask.sum().item())
```

**What happens:**
- Converts the edge list to a sparse matrix representation
- Efficient for graph algorithms (saves memory for large graphs)

### Step 4: Find Connected Components
```python
_, candidate_labels = sps.csgraph.connected_components(
    sparse_edges, directed=False, return_labels=True
)
```

**What happens:**
- Runs the **connected components** algorithm on the filtered graph
- Finds all groups of nodes that are connected through edges
- Each connected component becomes a **track candidate**

**How it works:**
- Starts from any node and follows edges to find all reachable nodes
- All nodes reachable from each other belong to the same component
- Each component gets a unique label (0, 1, 2, ...)

**Example:**
```
Graph after filtering:
  Hit1 -- Hit2 -- Hit3
  Hit4 -- Hit5
  Hit6 (isolated, already removed)

Connected Components:
  Component 0: [Hit1, Hit2, Hit3]  → Track candidate 0
  Component 1: [Hit4, Hit5]        → Track candidate 1
```

### Step 5: Assign Track Labels
```python
labels = (torch.ones(num_nodes) * -1).long()
labels[mask] = torch.from_numpy(candidate_labels).long()
graph.hit_track_labels = labels
```

**What happens:**
- Assigns each hit a track ID based on its connected component
- Hits in the same component get the same track ID
- Isolated hits (removed earlier) get label -1 (no track)

### Step 6: Sort and Save Tracks
```python
# Sort hits by track_id and distance from origin (r²)
d = d[d.track_id >= 0].sort_values(["track_id", "r2"])
tracks = d.groupby("track_id")["hit_id"].apply(list)
```

**What happens:**
- Groups hits by track ID
- Sorts hits within each track by distance from origin (r² = r² + z²)
- Creates final track list: each track is a list of hit IDs in order

## Visual Example

### Before Track Building:
```
Graph with edge scores:
  Hit1 --0.8-- Hit2 --0.9-- Hit3
  Hit4 --0.6-- Hit5
  Hit6 --0.3-- Hit7  (low score, will be filtered)
```

### After Score Cut (score_cut=0.5):
```
Filtered graph:
  Hit1 ----- Hit2 ----- Hit3  (all scores > 0.5)
  Hit4 ----- Hit5      (score 0.6 > 0.5)
  Hit6, Hit7 removed   (score 0.3 < 0.5)
```

### After Connected Components:
```
Track 0: [Hit1, Hit2, Hit3]
Track 1: [Hit4, Hit5]
```

## Key Parameters

### `score_cut` (default: 0.5)
- **Lower values** (e.g., 0.3):
  - Keep more edges → More tracks, but potentially more fake tracks
  - Higher recall, lower precision
  
- **Higher values** (e.g., 0.7):
  - Keep fewer edges → Fewer tracks, but higher quality
  - Lower recall, higher precision

### `max_workers` (default: 1)
- Number of parallel workers for processing events
- Set to > 1 for faster processing on multi-core systems

## Advantages

1. **Simple and fast**: O(n + m) complexity where n = nodes, m = edges
2. **Deterministic**: Same input always gives same output
3. **No assumptions**: Works with any graph structure
4. **Handles branching**: Can find multiple paths through the graph

## Limitations

1. **No directionality**: Treats graph as undirected (ignores track direction)
2. **No path selection**: All connected nodes become one track (can merge separate tracks if they connect)
3. **No quality filtering**: Doesn't filter tracks by length, curvature, etc.
4. **Can create false connections**: If two tracks accidentally connect through noise, they merge

## Alternative Algorithms

The config supports other algorithms:

- **Walkthrough**: Finds paths from source to target nodes (better for directed graphs)
- **CCandWalk**: Combines connected components + walkthrough (better performance)
- **FastWalkthrough**: Optimized version of walkthrough

To use a different algorithm, change `model:` in `track_building.yaml`:
```yaml
model: CCandWalk  # or Walkthrough, FastWalkthrough, etc.
```

## Output Format

Tracks are saved as:
- **CSV/TXT files**: One file per event
- **Each row**: A track (list of hit IDs)
- **Graphs with labels**: Saved with `hit_track_labels` attribute

Example track file:
```
[1234, 1235, 1236, 1237]  # Track 0: 4 hits
[2345, 2346, 2347]         # Track 1: 3 hits
[3456, 3457, 3458, 3459, 3460]  # Track 2: 5 hits
```
