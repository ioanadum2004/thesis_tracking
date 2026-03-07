#!/usr/bin/env python3
"""
visual_utils.py - Shared utility functions for visualization scripts

This module provides common functionality used across multiple visualization
scripts in the create_visuals/ directory, reducing code duplication and
ensuring consistency.

Common use cases:
- Coordinate conversion (cylindrical to Cartesian)
- PyG graph loading with error handling
- Node coordinate extraction from various attribute naming conventions
- Standard Plotly layout configuration
- Dataset validation
"""

from pathlib import Path
import numpy as np
import torch
import plotly.graph_objects as go


# ==============================================================================
# COORDINATE CONVERSION
# ==============================================================================

def cylindrical_to_cartesian(r, phi, z):
    """
    Convert cylindrical coordinates (r, phi, z) to Cartesian (x, y, z).

    Args:
        r: Radial distance (mm)
        phi: Azimuthal angle (radians)
        z: Z-coordinate (mm)

    Returns:
        tuple: (x, y, z) in Cartesian coordinates (mm)
    """
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y, z


# ==============================================================================
# GRAPH LOADING AND VALIDATION
# ==============================================================================

def validate_dataset_name(dataset_name):
    """
    Validate that dataset name is one of the allowed values.

    Args:
        dataset_name: Name to validate

    Raises:
        ValueError: If dataset_name is not 'trainset', 'valset', or 'testset'
    """
    if dataset_name not in ['trainset', 'valset', 'testset']:
        raise ValueError(
            f"Dataset must be 'trainset', 'valset', or 'testset', got '{dataset_name}'"
        )


def load_pyg_graph(graph_path, check_attributes=None):
    """
    Load a PyTorch Geometric graph with error handling.

    Args:
        graph_path: Path to .pyg file
        check_attributes: Optional list of attribute names to validate

    Returns:
        PyTorch Geometric Data object

    Raises:
        RuntimeError: If loading fails
        ValueError: If required attributes are missing
    """
    try:
        graph = torch.load(graph_path, map_location='cpu', weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load graph file {graph_path}: {e}")

    # Check for required attributes if specified
    if check_attributes:
        missing_attrs = [attr for attr in check_attributes if not hasattr(graph, attr)]
        if missing_attrs:
            available_attrs = [
                attr for attr in dir(graph)
                if not attr.startswith('_') and not callable(getattr(graph, attr, None))
            ]
            attrs_str = ', '.join(available_attrs[:15])
            if len(available_attrs) > 15:
                attrs_str += f", ... ({len(available_attrs)} total)"

            raise ValueError(
                f"\nGraph is missing required attributes: {missing_attrs}\n"
                f"  Graph file: {graph_path}\n"
                f"  Available attributes: {attrs_str}\n"
            )

    return graph


def load_graph_by_index(dataset_dir, index):
    """
    Load the Nth graph from a dataset directory.

    Args:
        dataset_dir: Path to directory containing .pyg files
        index: Graph index (1-based)

    Returns:
        tuple: (graph, graph_path)

    Raises:
        ValueError: If no graphs found or index out of range
    """
    dataset_dir = Path(dataset_dir)

    # Get all .pyg files sorted
    graph_files = sorted([f for f in dataset_dir.glob('*.pyg')])
    if len(graph_files) == 0:
        raise ValueError(f"No graph files found in {dataset_dir}")

    if index < 1 or index > len(graph_files):
        raise ValueError(f"Index {index} out of range. Available: 1-{len(graph_files)}")

    # Select the graph file at the specified index (convert to 0-indexed)
    graph_path = graph_files[index - 1]
    graph = load_pyg_graph(graph_path)

    return graph, graph_path


# ==============================================================================
# COORDINATE EXTRACTION
# ==============================================================================

def extract_node_coordinates(graph, return_cylindrical=True, return_cartesian=True):
    """
    Extract node coordinates from graph, handling various attribute naming conventions.

    Supports both prefixed (hit_r, hit_phi, hit_z) and non-prefixed (r, phi, z) names.
    Can also fall back to Cartesian coordinates (x, y, z) if cylindrical not available.

    Args:
        graph: PyTorch Geometric Data object
        return_cylindrical: If True, include (r, phi, z) in return value
        return_cartesian: If True, include (x, y, z_cart) in return value

    Returns:
        dict: Dictionary with keys:
            - 'r', 'phi', 'z': Cylindrical coordinates (if return_cylindrical=True)
            - 'x', 'y', 'z_cart': Cartesian coordinates (if return_cartesian=True)
            All values are numpy arrays

    Raises:
        ValueError: If no recognized coordinate attributes found
    """
    coords = {}

    # Try to get cylindrical coordinates
    r, phi, z = None, None, None

    if hasattr(graph, 'hit_r'):
        r = np.array(graph.hit_r.tolist())
        phi = np.array(graph.hit_phi.tolist())
        z = np.array(graph.hit_z.tolist())
    elif hasattr(graph, 'r'):
        r = np.array(graph.r.tolist())
        phi = np.array(graph.phi.tolist())
        z = np.array(graph.z.tolist())
    elif hasattr(graph, 'x') and hasattr(graph, 'z'):
        # Fall back to Cartesian if available
        x = np.array(graph.x.tolist())
        z_cart = np.array(graph.z.tolist())

        # Check for y coordinate
        if hasattr(graph, 'hit_y'):
            y = np.array(graph.hit_y.tolist())
        elif hasattr(graph, 'y'):
            y = np.array(graph.y.tolist())
        else:
            raise ValueError(
                "Graph has x and z but no y coordinate (checked 'y' and 'hit_y')"
            )

        # Calculate cylindrical from Cartesian
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        # z is already loaded

        # Store Cartesian directly
        if return_cartesian:
            coords['x'] = x
            coords['y'] = y
            coords['z_cart'] = z_cart
    else:
        raise ValueError(
            "Graph does not have recognizable coordinate attributes. "
            "Checked: hit_r/hit_phi/hit_z, r/phi/z, x/y/z"
        )

    # Convert to Cartesian if needed and not already done
    if return_cartesian and 'x' not in coords:
        x, y, z_cart = cylindrical_to_cartesian(r, phi, z)
        coords['x'] = x
        coords['y'] = y
        coords['z_cart'] = z_cart

    # Store cylindrical if requested
    if return_cylindrical:
        coords['r'] = r
        coords['phi'] = phi
        coords['z'] = z

    return coords


# ==============================================================================
# PLOTLY LAYOUT CONFIGURATION
# ==============================================================================

def get_standard_scene_layout(
    title_text,
    margin_t=40,
    margin_b=0,
    margin_l=0,
    margin_r=0,
    legend_opacity=0.9
):
    """
    Create standard Plotly layout configuration for 3D scatter plots.

    Args:
        title_text: Title for the plot (can include <br> for multiline)
        margin_t: Top margin (default: 40)
        margin_b: Bottom margin (default: 0)
        margin_l: Left margin (default: 0)
        margin_r: Right margin (default: 0)
        legend_opacity: Legend background opacity (default: 0.9)

    Returns:
        dict: Layout configuration for fig.update_layout()
    """
    return {
        'title': dict(
            text=title_text,
            x=0.5,
            xanchor='center'
        ),
        'scene': dict(
            xaxis=dict(
                title='x (mm)',
                backgroundcolor="white",
                gridcolor="lightgray"
            ),
            yaxis=dict(
                title='y (mm)',
                backgroundcolor="white",
                gridcolor="lightgray"
            ),
            zaxis=dict(
                title='z (mm)',
                backgroundcolor="white",
                gridcolor="lightgray"
            ),
            aspectmode='data'  # Preserves physical aspect ratio
        ),
        'showlegend': True,
        'legend': dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor=f"rgba(255, 255, 255, {legend_opacity})"
        ),
        'margin': dict(l=margin_l, r=margin_r, b=margin_b, t=margin_t),
        'hovermode': 'closest'
    }


# ==============================================================================
# EDGE BUILDING
# ==============================================================================

def build_edge_coordinates(edge_index, x, y, z, masks=None):
    """
    Build edge coordinate lists for Plotly from edge index.

    Creates lists of x, y, z coordinates with None separators between edges
    (Plotly's required format for drawing disconnected line segments).

    Args:
        edge_index: Edge index tensor/array of shape (2, num_edges)
        x, y, z: Node coordinate arrays
        masks: Optional dict of {name: boolean_mask} to separate edges into categories
               If None, returns all edges as single category

    Returns:
        dict: If masks provided, returns {category: (edge_x, edge_y, edge_z, count)}
              If no masks, returns (edge_x, edge_y, edge_z, count)
    """
    edge_index = np.array(edge_index.tolist()) if hasattr(edge_index, 'tolist') else edge_index

    if masks is None:
        # Build all edges as single list
        edge_x, edge_y, edge_z = [], [], []
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            edge_x.extend([x[src], x[dst], None])
            edge_y.extend([y[src], y[dst], None])
            edge_z.extend([z[src], z[dst], None])

        count = (len(edge_x) - edge_x.count(None)) // 2 if edge_x else 0
        return (edge_x, edge_y, edge_z, count)

    else:
        # Build edges by category
        edge_data = {}

        for category, mask in masks.items():
            edge_x, edge_y, edge_z = [], [], []

            for i in np.where(mask)[0]:
                src, dst = edge_index[0, i], edge_index[1, i]
                edge_x.extend([x[src], x[dst], None])
                edge_y.extend([y[src], y[dst], None])
                edge_z.extend([z[src], z[dst], None])

            count = (len(edge_x) - edge_x.count(None)) // 2 if edge_x else 0
            edge_data[category] = (edge_x, edge_y, edge_z, count)

        return edge_data


# ==============================================================================
# PARTICLE TYPE UTILITIES
# ==============================================================================

def pdg_to_particle_name(pdg_code):
    """
    Convert PDG code to human-readable particle name.

    Args:
        pdg_code: Integer PDG code (e.g., 13, -13, 211)

    Returns:
        str: Particle name with charge (e.g., "μ-", "π+", "e+")
    """
    pdg_map = {
        # Muons
        13: "μ-",
        -13: "μ+",
        # Pions
        211: "π+",
        -211: "π-",
        111: "π0",
        # Electrons
        11: "e-",
        -11: "e+",
        # Protons
        2212: "p",
        -2212: "p̄",
        # Kaons
        321: "K+",
        -321: "K-",
        311: "K0",
        # Neutrons
        2112: "n",
        -2112: "n̄",
        # Photons
        22: "γ",
    }

    return pdg_map.get(pdg_code, f"PDG{pdg_code}")


def build_hit_particle_type_map(graph):
    """
    Build mapping from hit_particle_id to particle_type (PDG code).

    Uses track_particle_id and track_particle_type from the graph to create
    a lookup table for hit-level particle type information.

    Args:
        graph: PyTorch Geometric Data object with track_particle_id and track_particle_type

    Returns:
        dict: Mapping {particle_id: pdg_code}
              Returns empty dict if required attributes not found
    """
    if not hasattr(graph, 'track_particle_id') or not hasattr(graph, 'track_particle_type'):
        return {}

    particle_id_to_type = {}
    track_particle_ids = np.array(graph.track_particle_id.tolist())
    track_particle_types = np.array(graph.track_particle_type.tolist())

    for pid, ptype in zip(track_particle_ids, track_particle_types):
        particle_id_to_type[int(pid)] = int(ptype)

    return particle_id_to_type


def get_particle_label(particle_id, particle_type_map):
    """
    Generate a display label for a particle combining ID and type.

    Args:
        particle_id: Particle ID
        particle_type_map: Dict from build_hit_particle_type_map()

    Returns:
        str: Formatted label like "Particle 12345 (μ-)" or "Particle 12345"
    """
    if particle_id == 0:
        return "Noise"

    if particle_id in particle_type_map:
        pdg_code = particle_type_map[particle_id]
        particle_name = pdg_to_particle_name(pdg_code)
        return f"Particle {particle_id} ({particle_name})"
    else:
        return f"Particle {particle_id}"


# ==============================================================================
# COLOR PALETTES
# ==============================================================================

def get_color_palette(palette='Set3'):
    """
    Get a standard color palette for particle/track coloring.

    Args:
        palette: Name of Plotly Express color palette (default: 'Set3')

    Returns:
        list: List of color strings
    """
    import plotly.express as px

    if palette == 'Set3':
        return px.colors.qualitative.Set3
    elif palette == 'Plotly':
        return px.colors.qualitative.Plotly
    elif palette == 'D3':
        return px.colors.qualitative.D3
    else:
        # Default to Set3 if unknown
        return px.colors.qualitative.Set3


# ==============================================================================
# HOVER TEXT HELPERS
# ==============================================================================

def create_node_hover_text(
    node_idx,
    coords,
    additional_fields=None,
    coord_precision=1,
    angle_precision=3
):
    """
    Create standardized hover text for a node.

    Args:
        node_idx: Node index
        coords: Dict with keys 'x', 'y', 'z_cart', 'r', 'phi'
                (from extract_node_coordinates)
        additional_fields: Optional dict of {field_name: value} to append
        coord_precision: Decimal places for coordinates (default: 1)
        angle_precision: Decimal places for angles (default: 3)

    Returns:
        str: Formatted hover text with <br> separators
    """
    hover_info = f"Node {node_idx}<br>"

    # Cartesian coordinates
    if 'x' in coords:
        hover_info += f"x: {coords['x']:.{coord_precision}f} mm<br>"
    if 'y' in coords:
        hover_info += f"y: {coords['y']:.{coord_precision}f} mm<br>"
    if 'z_cart' in coords:
        hover_info += f"z: {coords['z_cart']:.{coord_precision}f} mm<br>"

    # Cylindrical coordinates
    if 'r' in coords:
        hover_info += f"r: {coords['r']:.{coord_precision}f} mm<br>"
    if 'phi' in coords:
        hover_info += f"phi: {coords['phi']:.{angle_precision}f} rad"

    # Additional fields
    if additional_fields:
        for field_name, value in additional_fields.items():
            if isinstance(value, float):
                hover_info += f"<br>{field_name}: {value:.3f}"
            else:
                hover_info += f"<br>{field_name}: {value}"

    return hover_info


# ==============================================================================
# TRACK EVALUATION AND CLASSIFICATION
# ==============================================================================

def load_matching_df_for_event(graph, dataset_name, evaluation_dir=None):
    """
    Load matching DataFrame for a specific event from evaluation CSVs.

    Args:
        graph: PyTorch Geometric Data object with event_id attribute
        dataset_name: Name of dataset ('trainset', 'valset', or 'testset')
        evaluation_dir: Directory containing evaluation CSVs (default: ../data/track_evaluation)

    Returns:
        pandas.DataFrame: Matching DataFrame filtered for this event

    Raises:
        FileNotFoundError: If evaluation directory or matching CSV not found
    """
    import pandas as pd

    # Get event_id from graph
    event_id = graph.event_id
    if isinstance(event_id, list):
        event_id = event_id[0]
    event_id = int(event_id)

    # Set default evaluation directory
    if evaluation_dir is None:
        script_dir = Path(__file__).resolve().parent
        evaluation_dir = script_dir.parent / 'data' / 'track_evaluation'
    else:
        evaluation_dir = Path(evaluation_dir)

    eval_dataset_dir = evaluation_dir / dataset_name
    if not eval_dataset_dir.exists():
        raise FileNotFoundError(
            f"\nEvaluation directory not found: {eval_dataset_dir}\n"
            f"  Solution: Run evaluation first:\n"
            f"    python evaluate_tracks.py {dataset_name}\n"
        )

    # Load matching DataFrame
    matching_df_path = eval_dataset_dir / f"matching_df_{dataset_name}.csv"
    if not matching_df_path.exists():
        raise FileNotFoundError(
            f"\nMatching DataFrame not found: {matching_df_path}\n"
            f"  Solution: Run evaluation first:\n"
            f"    python evaluate_tracks.py {dataset_name}\n"
        )

    matching_df = pd.read_csv(matching_df_path)
    matching_df_event = matching_df[matching_df.event_id == event_id].copy()

    return matching_df_event


def classify_tracks(graph, matching_df_event, purity_threshold=0.75):
    """
    Categorize each track by quality based on matching DataFrame.

    This function classifies reconstructed tracks into categories:
    - high_purity: Matched tracks with purity >= purity_threshold
    - medium_purity: Matched tracks with 0.5 <= purity < purity_threshold
    - clone: Duplicate reconstructions of the same particle (2nd, 3rd, ... tracks)
    - fake: Unmatched tracks or tracks with purity < 0.5

    Args:
        graph: PyTorch Geometric Data object with hit_track_labels
        matching_df_event: Matching DataFrame filtered for this event
        purity_threshold: Threshold for high/medium purity split (default: 0.75)

    Returns:
        dict: Mapping track_id -> classification_info dict with keys:
            - category: str ('high_purity', 'medium_purity', 'clone', 'fake')
            - purity: float (max purity_reco)
            - particle_id: int (best match particle_id, or 0 for fake)
            - is_matched: bool
    """
    # Get all unique track IDs from hit_track_labels
    hit_track_labels = np.array(graph.hit_track_labels.tolist())
    all_track_ids = set(hit_track_labels.tolist())
    all_track_ids.discard(-1)  # Remove noise

    # Group matching_df by particle to find clones
    # particle_id -> list of (track_id, purity_reco) tuples
    particle_track_map = {}

    for _, row in matching_df_event.iterrows():
        if row.is_matched:
            pid = row.particle_id
            tid = row.track_id
            purity = row.purity_reco

            if pid not in particle_track_map:
                particle_track_map[pid] = []
            particle_track_map[pid].append((tid, purity))

    # Sort by purity for each particle (highest first)
    for pid in particle_track_map:
        particle_track_map[pid].sort(key=lambda x: x[1], reverse=True)

    # Identify clone tracks (2nd, 3rd, ... tracks for same particle)
    clone_track_ids = set()
    for pid, tracks in particle_track_map.items():
        if len(tracks) > 1:
            # First track is primary, rest are clones
            clone_track_ids.update(t[0] for t in tracks[1:])

    # Classify each track
    track_classifications = {}

    for track_id in all_track_ids:
        # Find best match for this track in matching_df
        track_matches = matching_df_event[matching_df_event.track_id == track_id]

        if len(track_matches) == 0:
            # Track has no particle overlap at all
            track_classifications[track_id] = {
                'category': 'fake',
                'purity': 0.0,
                'particle_id': 0,
                'is_matched': False
            }
        else:
            # Get best match (highest purity)
            best_match_idx = track_matches.purity_reco.idxmax()
            best_match = track_matches.loc[best_match_idx]

            if track_id in clone_track_ids:
                # This is a clone track
                track_classifications[track_id] = {
                    'category': 'clone',
                    'purity': best_match.purity_reco,
                    'particle_id': best_match.particle_id,
                    'is_matched': best_match.is_matched
                }
            elif best_match.is_matched:
                # Track is matched to a particle
                if best_match.purity_reco >= purity_threshold:
                    category = 'high_purity'
                else:
                    category = 'medium_purity'

                track_classifications[track_id] = {
                    'category': category,
                    'purity': best_match.purity_reco,
                    'particle_id': best_match.particle_id,
                    'is_matched': True
                }
            else:
                # Track failed matching criteria (low purity)
                track_classifications[track_id] = {
                    'category': 'fake',
                    'purity': best_match.purity_reco,
                    'particle_id': best_match.particle_id,
                    'is_matched': False
                }

    return track_classifications


def build_track_edges_by_category(graph, track_classifications, x, y, z):
    """
    Group track_edges by category for separate Plotly traces.

    Args:
        graph: PyTorch Geometric Data object with track_edges and hit_track_labels
        track_classifications: Dict from classify_tracks()
        x, y, z: Cartesian coordinates for all nodes

    Returns:
        dict: Mapping category -> (edge_x, edge_y, edge_z, count)
    """
    track_edges = np.array(graph.track_edges.tolist())
    hit_track_labels = np.array(graph.hit_track_labels.tolist())

    # Initialize edge lists for each category
    categories = {
        'high_purity': ([], [], []),
        'medium_purity': ([], [], []),
        'clone': ([], [], []),
        'fake': ([], [], [])
    }

    # Iterate through track_edges and categorize
    for i in range(track_edges.shape[1]):
        src, dst = track_edges[0, i], track_edges[1, i]

        # Determine which track this edge belongs to
        # Both endpoints should have the same track_id
        src_track = hit_track_labels[src]
        dst_track = hit_track_labels[dst]

        # Use the track ID (should be same for both, but check src first)
        track_id = src_track if src_track != -1 else dst_track

        if track_id == -1:
            # Isolated edge (shouldn't happen in track_edges, but handle it)
            continue

        # Get category
        if track_id in track_classifications:
            category = track_classifications[track_id]['category']
        else:
            # Shouldn't happen, but default to fake
            category = 'fake'

        # Add edge coordinates
        edge_x, edge_y, edge_z = categories[category]
        edge_x.extend([x[src], x[dst], None])
        edge_y.extend([y[src], y[dst], None])
        edge_z.extend([z[src], z[dst], None])

    # Convert to dict with counts
    edge_data = {}
    for category, (edge_x, edge_y, edge_z) in categories.items():
        count = (len(edge_x) - edge_x.count(None)) // 2 if edge_x else 0
        edge_data[category] = (edge_x, edge_y, edge_z, count)

    return edge_data


def build_ground_truth_trajectory_edges(graph):
    """
    Build ground truth trajectory edges using time-based ordering.

    For each particle, sorts hits by time and creates sequential edges
    representing the true trajectory through the detector.

    Args:
        graph: PyTorch Geometric Data object with hit_particle_id and hit_t

    Returns:
        list: List of (src, dst) tuples representing ground truth trajectory edges
    """
    import numpy as np

    hit_particle_ids = np.array(graph.hit_particle_id.tolist())
    hit_times = np.array(graph.hit_t.tolist())

    ground_truth_edges = []

    # Build time-ordered trajectory edges for each particle
    for particle_id in np.unique(hit_particle_ids):
        if particle_id <= 0:  # Skip noise hits
            continue

        # Get hits for this particle
        particle_mask = hit_particle_ids == particle_id
        particle_nodes = np.where(particle_mask)[0]
        particle_times = hit_times[particle_mask]

        # Sort by time to get trajectory order
        time_order = np.argsort(particle_times)
        sorted_nodes = particle_nodes[time_order]

        # Create sequential edges: hit[i] -> hit[i+1]
        for i in range(len(sorted_nodes) - 1):
            src, dst = sorted_nodes[i], sorted_nodes[i+1]
            ground_truth_edges.append((src, dst))

    return ground_truth_edges
