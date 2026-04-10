"""
create_spacepoints_looping <out_prefix> [--hits HITS_ROOT] [--max-pt FLOAT]

Produces a side-by-side 2D figure to visualise low-pT looping tracks:

  Left panel  : transverse plane  x vs y
  Right panel : longitudinal plane  z vs r  (r = sqrt(x^2+y^2))

Overlaid on both panels:
  - Detector layer positions (inferred from hit density in r and z)
  - Beam axis  (z-axis on the left panel as a + at origin;
                horizontal line at r=0 on the right panel)

Particle rendering:
  - pT >= --max-pt  : scatter dots only, coloured by particle, semi-transparent
  - pT <  --max-pt  : hits connected by lines (ordered by r) so the curl is
                       visible, drawn on top with full opacity
  - Noise (pid==0)  : small gray dots

Legend shows each particle individually with its pT value.

Requires particles.root in the same directory as hits.root for pT information.
If particles.root is absent all tracks are rendered as dots.

Usage:
  python create_spacepoints_looping.py out
  python create_spacepoints_looping.py out --hits /path/to/hits.root
  python create_spacepoints_looping.py out --max-pt 1.0
  python create_spacepoints_looping.py out --dpi 200

Dependencies: uproot, numpy, matplotlib
"""

import argparse
import sys
from pathlib import Path

import uproot
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

try:
    from create_track_table import VOLUME_MAP
except Exception:
    VOLUME_MAP = {}

DEFAULT_MAX_PT = 0.5   # GeV


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def read_hits(hits_path):
    """Return tx, ty, tz, sensitive_id, volume_id, particle_id (or None)."""
    try:
        f = uproot.open(str(hits_path))
    except Exception as e:
        raise RuntimeError(f"Cannot open ROOT file: {e}")

    keys = list(f.keys())
    if not keys:
        raise RuntimeError("No objects found in ROOT file.")

    chosen = next((k for k in keys if k.lower().startswith('hits')), keys[0])
    t = f[chosen]

    required = ['tx', 'ty', 'tz', 'sensitive_id', 'volume_id']
    missing = [b for b in required if b not in t.keys()]
    if missing:
        raise RuntimeError(f"Missing branches in '{chosen}': {missing}")

    tx  = np.asarray(t['tx'].array())
    ty  = np.asarray(t['ty'].array())
    tz  = np.asarray(t['tz'].array())
    mod = np.asarray(t['sensitive_id'].array())
    vol = np.asarray(t['volume_id'].array())

    pid = None
    if 'particle_id' in t.keys():
        try:
            pid = np.asarray(t['particle_id'].array())
        except Exception:
            pass

    return tx, ty, tz, mod, vol, pid


def _to_flat_numpy(arr):
    """Convert any uproot/awkward array to a flat 1-D numpy array.

    uproot can return awkward arrays that look array-like but crash on
    element-wise int()/float() conversion. This function tries multiple
    strategies in order to reliably flatten them.
    """
    # Strategy 1: awkward array with .to_numpy()
    if hasattr(arr, 'to_numpy'):
        try:
            return np.asarray(arr.to_numpy()).ravel()
        except Exception:
            pass
    # Strategy 2: has a .flatten() method (some uproot versions)
    if hasattr(arr, 'flatten'):
        try:
            return np.asarray(arr.flatten()).ravel()
        except Exception:
            pass
    # Strategy 3: cast to numpy then flatten if needed
    a = np.asarray(arr)
    if a.dtype == object or a.ndim > 1:
        return np.concatenate(
            [np.atleast_1d(np.asarray(item)) for item in a]
        ).ravel()
    return a.ravel()


def read_particle_pt(hits_path):
    """Return dict {particle_id: pt_GeV} from particles.root next to hits.root.

    Returns empty dict if file is absent or unreadable.
    """
    particles_path = Path(hits_path).parent / 'particles.root'
    if not particles_path.exists():
        return {}

    try:
        f = uproot.open(str(particles_path))
        keys = list(f.keys())
        chosen = next(
            (k for k in keys if k.lower().startswith('particle')),
            keys[0] if keys else None
        )
        if chosen is None:
            return {}

        t = f[chosen]
        tkeys = set(t.keys())

        pid_branch = next((b for b in ('particle_id', 'id', 'pid') if b in tkeys), None)
        pt_branch  = next((b for b in ('pt', 'p_t', 'pT') if b in tkeys), None)
        if pid_branch is None or pt_branch is None:
            print(f"Warning: could not find pid/pt branches in particles.root. "
                  f"Available: {sorted(tkeys)}")
            return {}

        pid_flat = _to_flat_numpy(t[pid_branch].array()).astype(np.int64)
        pt_flat  = _to_flat_numpy(t[pt_branch].array()).astype(np.float64)

        return {int(p): float(pt) for p, pt in zip(pid_flat, pt_flat)}

    except Exception as e:
        print(f"Warning: could not read particles.root: {e}")
        return {}


# ---------------------------------------------------------------------------
# Detector layer inference
# ---------------------------------------------------------------------------

def infer_detector_layers(r, z, n_bins=800, prominence_fraction=0.02):
    """Infer detector layer radii and z-positions from hit density peaks."""
    r_counts, r_edges = np.histogram(r, bins=n_bins)
    r_centres = 0.5 * (r_edges[:-1] + r_edges[1:])
    r_threshold = prominence_fraction * r_counts.max()
    layer_radii = _find_peaks(r_centres, r_counts, r_threshold, min_sep=5.0)

    az = np.abs(z)
    az = az[az > 50.0]
    if len(az) > 0:
        z_counts, z_edges = np.histogram(az, bins=n_bins)
        z_centres = 0.5 * (z_edges[:-1] + z_edges[1:])
        z_threshold = prominence_fraction * z_counts.max()
        layer_z = _find_peaks(z_centres, z_counts, z_threshold, min_sep=10.0)
    else:
        layer_z = np.array([])

    return layer_radii, layer_z


def _find_peaks(centres, counts, threshold, min_sep=5.0):
    peaks = []
    for i in range(1, len(counts) - 1):
        if counts[i] > threshold and counts[i] >= counts[i-1] and counts[i] >= counts[i+1]:
            if not peaks or (centres[i] - peaks[-1]) > min_sep:
                peaks.append(centres[i])
    return np.array(peaks)


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def make_particle_colormap(unique_pids):
    """Return {pid: rgba} using tab20; gray for noise (pid==0)."""
    cmap = plt.get_cmap('tab20')
    out = {0: (0.6, 0.6, 0.6, 0.35)}
    non_noise = [p for p in unique_pids if p != 0]
    for i, p in enumerate(non_noise):
        r, g, b, _ = cmap(i % 20)
        out[p] = (r, g, b, 1.0)
    return out


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_detector_overlay_xy(ax, layer_radii, r_max):
    for rad in layer_radii:
        if rad > r_max * 1.05:
            continue
        circle = plt.Circle((0, 0), rad, color='#aaaaaa', fill=False,
                             linewidth=0.6, linestyle='--', zorder=1)
        ax.add_patch(circle)
    ax.plot(0, 0, 'k+', markersize=10, markeredgewidth=1.2, zorder=5)


def draw_detector_overlay_rz(ax, layer_radii, layer_z, r_max, z_max):
    for rad in layer_radii:
        if rad > r_max * 1.05:
            continue
        ax.axhline(rad, color='#aaaaaa', linewidth=0.6, linestyle='--', zorder=1)
    for zd in layer_z:
        if zd > z_max * 1.05:
            continue
        ax.axvline( zd, color='#bbbbbb', linewidth=0.5, linestyle=':', zorder=1)
        ax.axvline(-zd, color='#bbbbbb', linewidth=0.5, linestyle=':', zorder=1)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='-', zorder=2)


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def make_figure(x, y, z, r, pids, pt_lookup, max_pt, layer_radii, layer_z,
                title, out_path, dpi):

    unique_pids = np.unique(pids) if pids is not None else np.array([0])
    color_map   = make_particle_colormap(unique_pids)

    fig, (ax_xy, ax_rz) = plt.subplots(1, 2, figsize=(16, 7))

    r_max = np.percentile(r, 99)
    z_max = np.percentile(np.abs(z), 99)

    draw_detector_overlay_xy(ax_xy, layer_radii, r_max)
    draw_detector_overlay_rz(ax_rz, layer_radii, layer_z, r_max, z_max)

    # Group hit indices by particle id
    pid_to_idx = {}
    if pids is not None:
        for i, p in enumerate(pids):
            pid_to_idx.setdefault(int(p), []).append(i)
    else:
        pid_to_idx = {0: list(range(len(x)))}

    # Per-particle legend entries (built during the loop below)
    particle_legend_handles = []
    noise_legend_added = False

    for pid_val, idxs in sorted(pid_to_idx.items()):
        idxs = np.array(idxs)
        col  = color_map.get(pid_val, (0.5, 0.5, 0.5, 0.5))

        hx = x[idxs]; hy = y[idxs]
        hr = r[idxs]; hz = z[idxs]

        # --- Noise ---
        if pid_val == 0:
            kw = dict(c=[col]*len(idxs), s=4, alpha=0.3, linewidths=0, zorder=2)
            ax_xy.scatter(hx, hy, **kw)
            ax_rz.scatter(hz, hr, **kw)
            if not noise_legend_added:
                noise_legend_added = True
            continue

        pt = pt_lookup.get(pid_val, None)
        is_low_pt = (pt is not None) and (pt < max_pt)

        if is_low_pt:
            # Order by r so lines follow the helix outward (and back if looping)
            order = np.argsort(hr)
            ox = hx[order]; oy = hy[order]
            or_ = hr[order]; oz = hz[order]

            ax_xy.plot(ox, oy, '-', color=col, linewidth=1.2, alpha=0.9, zorder=4)
            ax_xy.scatter(ox, oy, c=[col]*len(ox), s=20, linewidths=0, zorder=5)
            ax_rz.plot(oz, or_, '-', color=col, linewidth=1.2, alpha=0.9, zorder=4)
            ax_rz.scatter(oz, or_, c=[col]*len(oz), s=20, linewidths=0, zorder=5)

            pt_str = f"{pt:.3f} GeV" if pt is not None else "unknown"
            pt_label = f"Particle {pid_val}  pT={pt_str}  [low — lines]"
            linestyle = '-'
        else:
            kw = dict(c=[col]*len(idxs), s=12, alpha=0.6, linewidths=0, zorder=3)
            ax_xy.scatter(hx, hy, **kw)
            ax_rz.scatter(hz, hr, **kw)

            pt_str = f"{pt:.3f} GeV" if pt is not None else "unknown"
            pt_label = f"Particle {pid_val}  pT={pt_str}"
            linestyle = 'None'

        particle_legend_handles.append(
            mlines.Line2D([], [], marker='o', linestyle=linestyle,
                          color=col, markersize=5, label=pt_label)
        )

    # ------------------------------------------------------------------
    # Axes formatting
    # ------------------------------------------------------------------
    lim = r_max * 1.08
    ax_xy.set_xlim(-lim, lim)
    ax_xy.set_ylim(-lim, lim)
    ax_xy.set_aspect('equal')
    ax_xy.set_xlabel('x (mm)', fontsize=11)
    ax_xy.set_ylabel('y (mm)', fontsize=11)
    ax_xy.set_title('Transverse plane  (x vs y)', fontsize=12)
    ax_xy.grid(True, linestyle='--', linewidth=0.3, alpha=0.5)

    ax_rz.set_xlim(-z_max * 1.08, z_max * 1.08)
    ax_rz.set_ylim(-r_max * 0.05, r_max * 1.08)
    ax_rz.set_xlabel('z (mm)', fontsize=11)
    ax_rz.set_ylabel('r (mm)', fontsize=11)
    ax_rz.set_title('Longitudinal plane  (z vs r)', fontsize=12)
    ax_rz.grid(True, linestyle='--', linewidth=0.3, alpha=0.5)

    # ------------------------------------------------------------------
    # Legend: geometry symbols first, then one entry per particle with pT
    # ------------------------------------------------------------------
    geometry_handles = []

    if noise_legend_added:
        geometry_handles.append(
            mlines.Line2D([], [], marker='o', linestyle='None',
                          color=(0.6, 0.6, 0.6, 0.5), markersize=4,
                          label='Noise hits  (pid=0)'))

    geometry_handles.append(
        mlines.Line2D([], [], color='#aaaaaa', linewidth=0.8,
                      linestyle='--', label='Detector layer'))

    geometry_handles.append(
        mlines.Line2D([], [], color='black', linewidth=0.8,
                      linestyle='-', label='Beam axis'))

    if not pt_lookup:
        geometry_handles.append(
            mpatches.Patch(color='none',
                           label='pT unknown  (particles.root not found)'))

    all_handles = geometry_handles + particle_legend_handles

    fig.legend(
        handles=all_handles,
        loc='center left',
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8.5,
        framealpha=0.92,
        title=f'Legend  (low-pT threshold: {max_pt} GeV)',
        title_fontsize=9,
    )

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description='Visualise looping low-pT tracks with detector geometry overlay'
    )
    p.add_argument('out_prefix',
                   help='Output prefix -- image saved as <prefix>.png')
    p.add_argument('--hits', '-i', default='hits.root',
                   help='Path to hits.root (default: ./hits.root)')
    p.add_argument('--max-pt', type=float, default=DEFAULT_MAX_PT,
                   help=f'pT threshold in GeV for connected-line rendering '
                        f'(default: {DEFAULT_MAX_PT})')
    p.add_argument('--dpi', type=int, default=150,
                   help='Image DPI (default: 150)')
    args = p.parse_args()

    hits_path = Path(args.hits)
    if not hits_path.exists():
        print(f"ERROR: hits file not found: {hits_path}")
        sys.exit(1)

    try:
        tx, ty, tz, mod, vol, pid = read_hits(hits_path)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    mask = mod != 0
    if not np.any(mask):
        print("No measurement hits (sensitive_id != 0) found. Exiting.")
        sys.exit(1)

    x    = tx[mask]
    y    = ty[mask]
    z    = tz[mask]
    r    = np.sqrt(x**2 + y**2)
    pids = pid[mask] if pid is not None else None

    n_hits = len(x)
    print(f"Loaded {n_hits} measurement hits.")

    pt_lookup = read_particle_pt(hits_path)
    if pt_lookup:
        print(f"Loaded pT for {len(pt_lookup)} particles from particles.root.")
    else:
        print("Warning: particles.root not found or unreadable -- "
              "all tracks will be rendered as dots.")

    layer_radii, layer_z = infer_detector_layers(r, z)
    print(f"Inferred {len(layer_radii)} barrel layers, "
          f"{len(layer_z)} endcap disc positions.")

    title = f"Spacepoints  ({n_hits} hits)"
    if pids is not None:
        n_particles = len(np.unique(pids[pids > 0]))
        title += f"  --  {n_particles} particles"
    if pt_lookup:
        n_low = sum(1 for pt in pt_lookup.values() if pt < args.max_pt)
        title += f"  --  {n_low} low-pT (< {args.max_pt} GeV)"

    out_path = Path(f"{args.out_prefix}.png")
    make_figure(x, y, z, r, pids, pt_lookup, args.max_pt,
                layer_radii, layer_z, title, out_path, args.dpi)


if __name__ == '__main__':
    main()