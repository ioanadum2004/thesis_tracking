"""
Custom ACTS Reader for Low-pT Data with Time-Based Trajectory Ordering

This reader extends ActsReader to create trajectory-ordered sequential edges
using time information, which is essential for looping particles where
layer-based ordering creates incorrect zigzag patterns.

Usage:
    In convert_csv_to_pyg_sets.yaml, set:
        model: ActsCustomLowPTReader
"""

import sys
from pathlib import Path

# Add acorn to path
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(WORKSPACE_ROOT / 'acorn'))

from acorn.stages.data_reading.models.acts_reader import ActsReader
import numpy as np


class ActsCustomLowPTReader(ActsReader):
    """
    Custom reader for low-pT data that uses time-based trajectory ordering.
    
    For looping particles, we need to order hits by their actual trajectory order
    (using time), not by layer/module, to avoid zigzag patterns where particles
    loop back through the same layers.
    
    Creates sequential edges: hit[i] → hit[i+1] following trajectory order.
    """
    
    def _build_true_tracks(self, hits):
        """
        Override to create TRAJECTORY-ORDERED SEQUENTIAL EDGES.
        Uses time component to order hits along the actual particle trajectory,
        then creates sequential chain edges following that order.
        """
        assert all(
            col in hits.columns
            for col in [
                "hit_particle_id",
                "hit_id",
                "hit_x",
                "hit_y",
                "hit_z",
                "particle_vx",
                "particle_vy",
                "particle_vz",
            ]
        ), (
            "Need to add (particle_id, hit_id), (x,y,z) and (vx,vy,vz) features to hits"
            " dataframe in custom EventReader class"
        )

        signal = hits[(hits.hit_particle_id != 0)].copy()
        
        # Order hits by trajectory: use time if available, otherwise distance from vertex
        # Time increases monotonically along trajectory, even for looping particles
        # Columns get prefixed with "hit_" when converting to PyG, so check for "hit_t"
        time_col = None
        for col_name in ["hit_t", "hit_time", "t", "tt"]:
            if col_name in signal.columns:
                time_col = col_name
                break
        
        if time_col:
            # Sort by particle_id first, then by time (trajectory order)
            signal = signal.sort_values(["hit_particle_id", time_col]).reset_index(drop=False)
        else:
            # Fallback: sort by distance from production vertex
            # Note: This may not work perfectly for looping particles that return to same distance!
            signal = signal.assign(
                R=np.sqrt(
                    (signal.hit_x - signal.particle_vx) ** 2
                    + (signal.hit_y - signal.particle_vy) ** 2
                    + (signal.hit_z - signal.particle_vz) ** 2
                )
            )
            signal = signal.sort_values(["hit_particle_id", "R"]).reset_index(drop=False)

        # Group by particle ID and create SEQUENTIAL edges following trajectory order
        signal_index_list = (
            signal.groupby("hit_particle_id", sort=False)["index"]
            .agg(lambda x: list(x))
        )

        # Create SEQUENTIAL edges: hit[i] → hit[i+1] in trajectory order
        track_index_edges = []
        for particle_hits in signal_index_list.values:
            # Create sequential chain: hit[0]→hit[1], hit[1]→hit[2], ..., hit[n-2]→hit[n-1]
            if len(particle_hits) >= 2:
                for i in range(len(particle_hits) - 1):
                    track_index_edges.append((particle_hits[i], particle_hits[i + 1]))

        if len(track_index_edges) == 0:
            return np.array([]), np.array([]), np.array([])

        # Convert to numpy array format [2, num_edges]
        track_index_edges = np.array(track_index_edges).T

        track_edges = hits.hit_id.values[track_index_edges]

        assert (
            hits[hits.hit_id.isin(track_edges.flatten())].hit_particle_id == 0
        ).sum() == 0, "There are hits in the track edges that are noise"

        track_features = self._get_track_features(hits, track_index_edges, track_edges)

        # Remap
        track_edges, track_features, hits = self.remap_edges(
            track_edges, track_features, hits
        )

        return track_edges, track_features, hits
