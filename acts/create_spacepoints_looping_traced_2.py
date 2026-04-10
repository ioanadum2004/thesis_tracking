"""
create_spacepoints_looping <out_prefix> [--hits HITS_ROOT] [--max-pt FLOAT]

Produces a side-by-side 2D figure to visualise low-pT looping tracks:

  Left panel  : transverse plane  x vs y
  Right panel : longitudinal plane  z vs r  (r = sqrt(x^2+y^2))

Overlaid on both panels:
  - Detector layer positions (inferred from hit density in r and z)
  - Beam axis

For low-pT tracks (pT < --max-pt):
  - Hits are ordered by unwrapped azimuthal angle phi = atan2(y, x) so the
    connecting lines smoothly trace the helix arc rather than jumping around
  - A circle is fitted to the hits in x-y (algebraic least-squares fit) and
    drawn as a dashed circle, with its centre marked. This makes the looping
    structure explicit and shows that the rotation centre is offset from the
    beam axis.

Legend shows each particle with its pT, fitted radius, and circle centre.

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
    """Robustly convert any uproot/awkward array to a flat 1-D numpy array."""
    if hasattr(arr, 'to_numpy'):
        try:
            return np.asarray(arr.to_numpy()).ravel()
        except Exception:
            pass
    if hasattr(arr, 'flatten'):
        try:
            return np.asarray(arr.flatten()).ravel()
        except Exception:
            pass
    a = np.asarray(arr)
    if a.dtype == object or a.ndim > 1:
        return np.concatenate(
            [np.atleast_1d(np.asarray(item)) for item in a]
        ).ravel()
    return a.ravel()


def read_particle_pt(hits_path):
    """Return {particle_id: pt_GeV} from particles.root next to hits.root."""
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
            print(f"Warning: could not find pid/pt branches. Available: {sorted(tkeys)}")
            return {}
        pid_flat = _to_flat_numpy(t[pid_branch].array()).astype(np.int64)
        pt_flat  = _to_flat_numpy(t[pt_branch].array()).astype(np.float64)
        return {int(p): float(pt) for p, pt in zip(pid_flat, pt_flat)}
    except Exception as e:
        print(f"Warning: could not read particles.root: {e}")
        return {}


# ---------------------------------------------------------------------------
# Circle fitting  (algebraic least-squares, Coope method)
# ---------------------------------------------------------------------------

def fit_circle(x, y):
    """Fit a circle to points (x, y) using algebraic least squares.

    Returns (cx, cy, radius) or None if fit fails (e.g. fewer than 3 points).

    The method solves:   (x-cx)^2 + (y-cy)^2 = R^2
    Rearranged to a linear system:
        2*cx*x + 2*cy*y + (R^2 - cx^2 - cy^2) = x^2 + y^2
    i.e.   A * [cx, cy, c]^T = b   where c = R^2 - cx^2 - cy^2
    """
    if len(x) < 3:
        return None
    A = np.column_stack([2*x, 2*y, np.ones(len(x))])
    b = x**2 + y**2
    try:
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return None
    cx, cy = result[0], result[1]
    c = result[2]
    r2 = c + cx**2 + cy**2
    if r2 <= 0:
        return None
    return cx, cy, np.sqrt(r2)


# ---------------------------------------------------------------------------
# Hit ordering for smooth arc tracing
# ---------------------------------------------------------------------------

def order_by_phi(x, y):
    """Return indices that order hits by unwrapped azimuthal angle.

    Standard atan2 wraps at +/-pi, causing jumps for tracks that cross that
    boundary. We unwrap by fitting a circle first to find the centre, then
    computing angles relative to that centre and unwrapping them.
    """
    if len(x) < 2:
        return np.arange(len(x))

    fit = fit_circle(x, y)
    if fit is not None:
        cx, cy = fit[0], fit[1]
    else:
        cx, cy = 0.0, 0.0

    phi = np.arctan2(y - cy, x - cx)
    phi_unwrapped = np.unwrap(phi[np.argsort(phi)])[np.argsort(np.argsort(phi))]
    return np.argsort(phi_unwrapped)


# ---------------------------------------------------------------------------
# Detector layer inference
# ---------------------------------------------------------------------------

def infer_detector_layers(r, z, n_bins=800, prominence_fraction=0.02):
    r_counts, r_edges = np.histogram(r, bins=n_bins)
    r_centres = 0.5 * (r_edges[:-1] + r_edges[1:])
    layer_radii = _find_peaks(r_centres, r_counts,
                               prominence_fraction * r_counts.max(), min_sep=5.0)

    az = np.abs(z)
    az = az[az > 50.0]
    if len(az) > 0:
        z_counts, z_edges = np.histogram(az, bins=n_bins)
        z_centres = 0.5 * (z_edges[:-1] + z_edges[1:])
        layer_z = _find_peaks(z_centres, z_counts,
                               prominence_fraction * z_counts.max(), min_sep=10.0)
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
    cmap = plt.get_cmap('tab20')
    out = {0: (0.6, 0.6, 0.6, 0.25)}   # noise: very faint gray
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
        circle = plt.Circle((0, 0), rad, color='#cccccc', fill=False,
                             linewidth=0.7, linestyle='--', zorder=1)
        ax.add_patch(circle)
    # Beam axis: cross at origin
    ax.plot(0, 0, 'k+', markersize=12, markeredgewidth=1.5, zorder=6)


def draw_detector_overlay_rz(ax, layer_radii, layer_z, r_max, z_max):
    for rad in layer_radii:
        if rad > r_max * 1.05:
            continue
        ax.axhline(rad, color='#cccccc', linewidth=0.7, linestyle='--', zorder=1)
    for zd in layer_z:
        if zd > z_max * 1.05:
            continue
        ax.axvline( zd, color='#dddddd', linewidth=0.5, linestyle=':', zorder=1)
        ax.axvline(-zd, color='#dddddd', linewidth=0.5, linestyle=':', zorder=1)
    ax.axhline(0, color='black', linewidth=0.9, linestyle='-', zorder=2)


def draw_fitted_circle(ax, cx, cy, rad, col, r_max):
    """Draw the fitted circle and mark its centre on the x-y panel."""
    # Only draw if circle centre + radius is not absurdly large
    if rad > r_max * 3:
        return
    fitted = plt.Circle((cx, cy), rad, color=col, fill=False,
                         linewidth=1.0, linestyle=':', alpha=0.5, zorder=3)
    ax.add_patch(fitted)
    # Centre marker: small x
    ax.plot(cx, cy, marker='x', color=col, markersize=6,
            markeredgewidth=1.2, alpha=0.5, zorder=6)


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

    particle_legend_handles = []
    noise_legend_added = False

    for pid_val, idxs in sorted(pid_to_idx.items()):
        idxs = np.array(idxs)
        col  = color_map.get(pid_val, (0.5, 0.5, 0.5, 0.5))

        hx = x[idxs]; hy = y[idxs]
        hr = r[idxs]; hz = z[idxs]

        # --- Noise ---
        if pid_val == 0:
            ax_xy.scatter(hx, hy, c=[col]*len(idxs), s=3, alpha=0.2,
                          linewidths=0, zorder=2)
            ax_rz.scatter(hz, hr, c=[col]*len(idxs), s=3, alpha=0.2,
                          linewidths=0, zorder=2)
            if not noise_legend_added:
                noise_legend_added = True
            continue

        pt = pt_lookup.get(pid_val, None)
        is_low_pt = (pt is not None) and (pt < max_pt)

        if is_low_pt:
            # --- Order by unwrapped phi around fitted circle centre ---
            order = order_by_phi(hx, hy)
            ox = hx[order]; oy = hy[order]
            or_ = hr[order]; oz = hz[order]

            # Draw connecting lines (smooth arc)
            ax_xy.plot(ox, oy, '-', color=col, linewidth=1.3, alpha=0.5, zorder=4)
            ax_xy.scatter(ox, oy, c=[col]*len(ox), s=22, linewidths=0, alpha=0.5, zorder=5)

            # For r-z: order by z for a clean longitudinal view
            rz_order = np.argsort(hz)
            ax_rz.plot(hz[rz_order], hr[rz_order], '-', color=col,
                       linewidth=1.3, alpha=0.5, zorder=4)
            ax_rz.scatter(hz, hr, c=[col]*len(hz), s=22, linewidths=0, alpha=0.5, zorder=5)

            # --- Fit and draw circle in x-y ---
            fit = fit_circle(hx, hy)
            if fit is not None:
                cx, cy, fit_r = fit
                draw_fitted_circle(ax_xy, cx, cy, fit_r, col, r_max)
                fit_str = f"  R={fit_r:.0f}mm  ctr=({cx:.0f},{cy:.0f})"
            else:
                fit_str = ""

            pt_str = f"{pt:.3f} GeV"
            pt_label = f"Particle {pid_val}  pT={pt_str}{fit_str}  [low]"
            linestyle = '-'

        else:
            ax_xy.scatter(hx, hy, c=[col]*len(idxs), s=12, alpha=0.5,
                          linewidths=0, zorder=3)
            ax_rz.scatter(hz, hr, c=[col]*len(idxs), s=12, alpha=0.5,
                          linewidths=0, zorder=3)

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
    ax_xy.grid(True, linestyle='--', linewidth=0.3, alpha=0.4)

    ax_rz.set_xlim(-z_max * 1.08, z_max * 1.08)
    ax_rz.set_ylim(-r_max * 0.05, r_max * 1.08)
    ax_rz.set_xlabel('z (mm)', fontsize=11)
    ax_rz.set_ylabel('r (mm)', fontsize=11)
    ax_rz.set_title('Longitudinal plane  (z vs r)', fontsize=12)
    ax_rz.grid(True, linestyle='--', linewidth=0.3, alpha=0.4)

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------
    geometry_handles = []

    if noise_legend_added:
        geometry_handles.append(
            mlines.Line2D([], [], marker='o', linestyle='None',
                          color=(0.6, 0.6, 0.6, 0.5), markersize=4,
                          label='Noise hits  (pid=0)'))

    geometry_handles.append(
        mlines.Line2D([], [], color='#cccccc', linewidth=1.0,
                      linestyle='--', label='Detector layer'))

    geometry_handles.append(
        mlines.Line2D([], [], color='black', linewidth=0.9,
                      linestyle='-', label='Beam axis'))

    geometry_handles.append(
        mlines.Line2D([], [], color='gray', linewidth=1.0,
                      linestyle=':', label='Fitted circle (low-pT)'))

    geometry_handles.append(
        mlines.Line2D([], [], marker='x', linestyle='None', color='gray',
                      markersize=6, label='Fitted circle centre'))

    if not pt_lookup:
        geometry_handles.append(
            mpatches.Patch(color='none',
                           label='pT unknown  (particles.root not found)'))

    all_handles = geometry_handles + particle_legend_handles

    fig.legend(
        handles=all_handles,
        loc='center left',
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8.0,
        framealpha=0.93,
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
        description='Visualise looping low-pT tracks with fitted circles and detector overlay'
    )
    p.add_argument('out_prefix',
                   help='Output prefix -- image saved as <prefix>.png')
    p.add_argument('--hits', '-i', default='hits.root',
                   help='Path to hits.root (default: ./hits.root)')
    p.add_argument('--max-pt', type=float, default=DEFAULT_MAX_PT,
                   help=f'pT threshold in GeV (default: {DEFAULT_MAX_PT})')
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