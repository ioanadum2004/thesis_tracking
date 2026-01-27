# Connected Components: Edge Case Analysis

## Your Scenario

**Setup:**
- 10 hits total
- All hits are mutually connected EXCEPT:
  - Hit5 and Hit6 are NOT directly connected
- But Hit5 is connected to: Hit1, Hit2, Hit3, Hit4, Hit7, Hit8, Hit9, Hit10
- And Hit6 is connected to: Hit1, Hit2, Hit3, Hit4, Hit7, Hit8, Hit9, Hit10

**Visual representation:**
```
Hit1 ── Hit2 ── Hit3 ── Hit4
 │       │       │       │
 │       │       │       │
Hit5     │       │     Hit6
 │       │       │       │
 │       │       │       │
Hit7 ── Hit8 ── Hit9 ── Hit10

Missing edge: Hit5 ── Hit6 (not directly connected)
```

## What Happens?

### Answer: They STILL form ONE track!

Even though Hit5 and Hit6 are not directly connected, they are still in the **same connected component** because there exists a **path** between them through other hits.

**Path example:**
- Hit5 → Hit1 → Hit6 (or)
- Hit5 → Hit4 → Hit6 (or)
- Hit5 → Hit7 → Hit8 → Hit9 → Hit10 → Hit6

The Connected Components algorithm finds **all nodes reachable from each other**, not just directly connected nodes.

### Algorithm Behavior:

```python
# Connected Components finds ALL reachable nodes
_, candidate_labels = sps.csgraph.connected_components(
    sparse_edges, directed=False, return_labels=True
)
```

**Result:** All 10 hits get the same track label (e.g., Track 0)

## Is This a Problem?

### Potential Issues:

1. **False Merging**: If Hit5 and Hit6 actually belong to **two separate tracks** that accidentally connect through other hits, they will be incorrectly merged into one track.

2. **Track Quality**: The resulting track might have unusual geometry if it's actually two tracks merged together.

### When This Happens:

- **Noise hits** create spurious connections between separate tracks
- **Overlapping tracks** share some hits, creating bridges
- **Edge classifier errors** keep fake edges that connect separate tracks

## Solutions

### Option 1: Increase `score_cut`
```yaml
score_cut: 0.7  # Higher threshold removes more edges
```
- Removes more low-confidence edges
- Reduces chance of false connections
- But might break real tracks if threshold is too high

### Option 2: Use a Different Algorithm

**CCandWalk** (Connected Components + Walkthrough):
```yaml
model: CCandWalk
```
- Uses connected components for simple paths
- Uses walkthrough for complex branching cases
- Better at handling ambiguous connections

**Walkthrough**:
```yaml
model: Walkthrough
```
- Finds paths from source to target nodes
- More selective about which paths form tracks
- Can handle branching better

### Option 3: Post-Processing

After track building, you could:
- Filter tracks by length (remove very long tracks)
- Filter tracks by curvature (remove tracks with unusual geometry)
- Split tracks at branching points

## Example with Numbers

### Scenario:
```
10 hits, all connected except Hit5-Hit6 missing edge

Edge scores:
  Hit5-Hit1: 0.9 ✓
  Hit5-Hit2: 0.8 ✓
  Hit5-Hit3: 0.7 ✓
  Hit5-Hit4: 0.6 ✓
  Hit5-Hit6: MISSING (no edge)
  Hit5-Hit7: 0.8 ✓
  ...
  Hit6-Hit1: 0.9 ✓
  Hit6-Hit2: 0.8 ✓
  ...
```

### With `score_cut = 0.5`:
- All edges kept (all scores > 0.5)
- **Result**: ONE track with all 10 hits
- Path: Hit5 → Hit1 → Hit6 (or any other path)

### With `score_cut = 0.7`:
- Hit5-Hit4 removed (0.6 < 0.7)
- Hit6-Hit4 removed (0.6 < 0.7)
- **Result**: Depends on remaining connections
  - If Hit5 and Hit6 still connect through other paths → ONE track
  - If all paths broken → TWO separate tracks

## Key Insight

The Connected Components algorithm is **transitive**:
- If A connects to B, and B connects to C, then A and C are in the same component
- Even if A and C are not directly connected

This is both a **strength** (finds complete tracks) and a **weakness** (can merge separate tracks).

## Recommendation

For your use case with potentially overlapping or closely-spaced tracks:

1. **Try CCandWalk** first - better handling of complex cases
2. **Tune score_cut** - find balance between keeping real tracks and removing fakes
3. **Evaluate results** - check if tracks are being incorrectly merged
4. **Consider post-processing** - filter or split tracks based on physics criteria
