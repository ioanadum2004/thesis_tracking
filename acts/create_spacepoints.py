"""
create_spacepoints <out_prefix> [--hits HITS_ROOT]

Reads `hits.root` (or the provided file) in the current directory and
produces a 3D scatter image of the spacepoints (x,y,z) saved as
`<out_prefix>.png` in the current working directory. Hits with
`sensitive_id == 0` are ignored (same convention as
`create_track_table.py`).

Usage examples:
  python create_spacepoints first_spacepoints
  python create_spacepoints first_spacepoints --hits /path/to/hits.root

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
try:
    from create_track_table import VOLUME_MAP
except Exception:
    VOLUME_MAP = {}


def read_hits(hits_path):
    """Read required branches from the hits ROOT file and return arrays.

    Returns (x, y, z, module_id)
    """
    try:
        treefile = uproot.open(str(hits_path))
    except Exception as e:
        raise RuntimeError(f"Error opening ROOT file: {e}")

    # pick tree that starts with 'hits' if available
    tree_names = list(treefile.keys())
    if len(tree_names) == 0:
        raise RuntimeError('No objects found in file')

    chosen = None
    for k in tree_names:
        if k.lower().startswith('hits'):
            chosen = k
            break
    if chosen is None:
        chosen = tree_names[0]

    t = treefile[chosen]

    required = ['tx', 'ty', 'tz', 'sensitive_id', 'volume_id']
    missing = [r for r in required if r not in t.keys()]
    if missing:
        raise RuntimeError(f"Missing required branches in {chosen}, missing: {missing}")

    tx = np.asarray(t['tx'].array())
    ty = np.asarray(t['ty'].array())
    tz = np.asarray(t['tz'].array())
    mod = np.asarray(t['sensitive_id'].array())
    vol = np.asarray(t['volume_id'].array())

    # optional particle id per hit
    pid = None
    if 'particle_id' in t.keys():
        try:
            pid = np.asarray(t['particle_id'].array())
        except Exception:
            pid = None

    return tx, ty, tz, mod, vol, pid


def equalize_axes(ax, x, y, z):
    # Try to use set_box_aspect (matplotlib >= 3.3). If unavailable, fall back
    # to manual scaling using limits so the axes look cubic.
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        # fallback: compute limits and set them to be equal ranges
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        zmin, zmax = np.min(z), np.max(z)
        xmid = 0.5 * (xmin + xmax)
        ymid = 0.5 * (ymin + ymax)
        zmid = 0.5 * (zmin + zmax)
        maxrange = max(xmax - xmin, ymax - ymin, zmax - zmin)
        if maxrange == 0:
            maxrange = 1.0
        half = 0.5 * maxrange
        ax.set_xlim(xmid - half, xmid + half)
        ax.set_ylim(ymid - half, ymid + half)
        ax.set_zlim(zmid - half, zmid + half)


def plot_spacepoints(x, y, z, labels=None, color_map=None, out_path=None, title=None, dpi=150):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    if labels is None:
        ax.scatter(x, y, z, c='red', s=8, alpha=0.8)
    else:
        # plot per-label so legend can be shown
        unique_labels = list(dict.fromkeys(labels))
        for lab in unique_labels:
            mask = [l == lab for l in labels]
            xs = x[mask]
            ys = y[mask]
            zs = z[mask]
            col = color_map.get(lab, 'gray') if color_map is not None else 'gray'
            ax.scatter(xs, ys, zs, c=col, s=8, alpha=0.8, label=lab)
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')
    if title is not None:
        ax.set_title(title)

    equalize_axes(ax, x, y, z)

    plt.tight_layout()
    if out_path is not None:
        fig.savefig(str(out_path), dpi=dpi)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description='Create 3D image of spacepoints from hits.root')
    p.add_argument('out_prefix', help='output prefix (image written as <prefix>.png)')
    p.add_argument('--hits', '-i', help='path to hits.root (default: ./hits.root)', default='hits.root')
    p.add_argument('--dpi', type=int, default=150, help='image DPI (default: 150)')
    p.add_argument('--interactive', dest='interactive', action='store_true', help='write an interactive HTML view using plotly (<out_prefix>.html)')
    p.add_argument('--no-interactive', dest='interactive', action='store_false', help='do not write interactive HTML')
    p.add_argument('--static', dest='static', action='store_true', help='also write a static PNG image (<out_prefix>.png). By default the script is interactive-only')
    p.set_defaults(interactive=True, static=False)
    args = p.parse_args()

    out_prefix = Path(args.out_prefix)
    hits_path = Path(args.hits)
    if not hits_path.exists():
        print(f"Could not find hits file: {hits_path}")
        sys.exit(1)

    try:
        tx, ty, tz, mod, vol, pid = read_hits(hits_path)
    except Exception as e:
        print(e)
        sys.exit(1)

    # Filter to measurement hits (sensitive_id != 0)
    mask = (mod != 0)
    if not np.any(mask):
        print('No measurement hits found (sensitive_id != 0). Exiting.')
        sys.exit(1)

    x = tx[mask]
    y = ty[mask]
    z = tz[mask]
    r = np.sqrt(x**2 + y**2)  # Calculate radial distance
    v = vol[mask]
    pids = pid[mask] if pid is not None else None  # Filter particle IDs

    # Map volume ids to human-readable labels using VOLUME_MAP
    def vol_label(vid):
        try:
            vm = VOLUME_MAP.get(int(vid), None)
        except Exception:
            vm = None
        if vm is None:
            return 'Unknown'
        region = vm.get('region')
        vtype = vm.get('type', 'Unknown')
        if region == 'barrel':
            return f"{vtype} Barrel"
        elif region == 'endcap':
            return f"{vtype} Endcap"
        else:
            return vtype

    labels = [vol_label(vid) for vid in v]

    # Define a color map for common detector types (fall back to gray)
    color_map = {
        'Pixel Barrel': 'blue',
        'ShortStrip Barrel': 'red',
        'LongStrip Barrel': 'green',
        'Pixel Endcap': 'magenta',
        'ShortStrip Endcap': 'cyan',
        'LongStrip Endcap': 'purple',
        'Unknown': 'gray',
    }

    # Load all particles to get Pt/eta for all particle IDs
    particle_info = {}  # particle_id -> (pt, eta)
    primary_pid = None
    particle_pt = None
    particle_eta = None
    try:
        if pids is not None:
            unique_pids, counts = np.unique(pids, return_counts=True)
            if unique_pids.size > 0:
                primary_pid = int(unique_pids[np.argmax(counts)])
        else:
            unique_pids = np.array([])

        particles_path = hits_path.parent / 'particles.root'
        if particles_path.exists():
            pfile = uproot.open(str(particles_path))
            p_names = list(pfile.keys())
            chosen_p = None
            for name in ('particles;1', 'particles'):
                if name in p_names:
                    chosen_p = name
                    break
            if chosen_p is not None:
                ptree = pfile[chosen_p]
                pkeys = set(ptree.keys())
                pid_branch = 'particle_id' if 'particle_id' in pkeys else None
                pt_branch = 'pt' if 'pt' in pkeys else None
                eta_branch = 'eta' if 'eta' in pkeys else None
                if pid_branch is None:
                    pid_branch = next((b for b in ('id', 'pid') if b in pkeys), None)
                if pt_branch is None:
                    pt_branch = next((b for b in ('p_t', 'pT') if b in pkeys), None)
                if eta_branch is None:
                    eta_branch = next((b for b in ('pseudorapidity',) if b in pkeys), None)

                if pid_branch is not None and pt_branch is not None:
                    part_pid_arr = np.asarray(ptree[pid_branch].array())
                    part_pt_arr = np.asarray(ptree[pt_branch].array())
                    part_eta_arr = None
                    if eta_branch is not None:
                        part_eta_arr = np.asarray(ptree[eta_branch].array())
                    
                    # Handle jagged arrays (arrays of arrays) from uproot
                    # Flatten if needed - uproot may return jagged arrays for multi-event files
                    if part_pid_arr.dtype == object or len(part_pid_arr.shape) > 1:
                        # Jagged array - flatten it
                        part_pid_flat = np.concatenate([np.atleast_1d(x) for x in part_pid_arr])
                        part_pt_flat = np.concatenate([np.atleast_1d(x) for x in part_pt_arr])
                        if part_eta_arr is not None:
                            part_eta_flat = np.concatenate([np.atleast_1d(x) for x in part_eta_arr])
                        else:
                            part_eta_flat = None
                    else:
                        # Regular flat array
                        part_pid_flat = part_pid_arr
                        part_pt_flat = part_pt_arr
                        part_eta_flat = part_eta_arr
                    
                    # Load all particles (not just primary)
                    for i in range(len(part_pid_flat)):
                        pid_val = int(part_pid_flat[i])
                        pt_val = float(part_pt_flat[i])
                        eta_val = float(part_eta_flat[i]) if part_eta_flat is not None else None
                        particle_info[pid_val] = (pt_val, eta_val)
                    
                    # Also set primary particle info for title
                    if primary_pid is not None:
                        matches = np.where(part_pid_flat == primary_pid)[0]
                        if matches.size > 0:
                            particle_pt = float(part_pt_flat[matches[0]])
                            if part_eta_flat is not None:
                                particle_eta = float(part_eta_flat[matches[0]])
    except Exception as e:
        import traceback
        print(f"Warning: Could not load particle info: {e}")
        traceback.print_exc()
        particle_info = {}
        particle_pt = None
        particle_eta = None
    
    # Debug: print particle_info status
    if particle_info:
        print(f"Loaded particle info for {len(particle_info)} particles")
        sample_pids = list(particle_info.keys())[:5]
        print(f"Sample particle IDs in particle_info: {sample_pids}")
        if pids is not None:
            unique_pids = np.unique(pids[pids != 0])
            matched = sum(1 for pid in unique_pids if pid in particle_info)
            print(f"Found {matched}/{len(unique_pids)} particle IDs from hits in particle_info")
            if matched < len(unique_pids):
                missing = [pid for pid in unique_pids if pid not in particle_info][:5]
                print(f"Sample missing particle IDs: {missing}")
    else:
        print("Warning: particle_info is empty - pt/eta values will not be shown")
        if 'particles_path' in locals():
            print(f"  particles.root path checked: {particles_path}")
            print(f"  particles.root exists: {particles_path.exists()}")
        else:
            print("  particles.root path not found in scope")

    out_png = Path(f"{out_prefix}.png")
    title = f"Spacepoints ({len(x)} hits)"
    # Add particle count if particle IDs are available
    if pids is not None:
        unique_particles = len(np.unique(pids[pids > 0]))  # Exclude noise (pid=0)
        title += f" — {unique_particles} unique particles"
    # Note: PT/eta info is shown in legend, not title (like create_visual_raw_csv_spacepoints.py)

    # Static PNG (only written if --static requested)
    if args.static:
        try:
            plot_spacepoints(x, y, z, labels=labels, color_map=color_map, out_path=out_png, title=title, dpi=args.dpi)
        except Exception as e:
            print(f"Error creating plot: {e}")
            sys.exit(1)

        print(f"Wrote image: {out_png}")

    # Interactive HTML using plotly (default unless --no-interactive)
    if args.interactive:
        try:
            import warnings
            import plotly.graph_objects as go
            import plotly.express as px
            import pandas as pd
            # Suppress FutureWarning from plotly's internal pandas grouping
            warnings.filterwarnings('ignore', category=FutureWarning, module='plotly')
        except Exception:
            print("Plotly or pandas not installed. Install with: pip install plotly pandas")
            sys.exit(1)

        # Downsample if extremely large to keep the interactive plot responsive
        max_points = 200000
        npoints = x.shape[0]
        if npoints > max_points:
            idx = np.random.default_rng(seed=0).choice(npoints, size=max_points, replace=False)
            xs = x[idx]
            ys = y[idx]
            zs = z[idx]
            rs = r[idx]
            labs = [labels[i] for i in idx]
            pids_sampled = pids[idx] if pids is not None else None
            info_msg = f"(downsampled from {npoints} to {max_points} points for interactive view)"
        else:
            xs = x
            ys = y
            zs = z
            rs = r
            labs = labels
            pids_sampled = pids
            info_msg = ""

        # Create hover text with x, y, z, r
        hover_text = []
        for i in range(len(xs)):
            hover_info = f"x: {xs[i]:.2f} mm<br>y: {ys[i]:.2f} mm<br>z: {zs[i]:.2f} mm<br>r: {rs[i]:.2f} mm<br>"
            hover_info += f"Volume: {labs[i]}<br>"
            if pids_sampled is not None:
                pid_val = int(pids_sampled[i])
                if pid_val != 0:
                    hover_info += f"particle_id: {pid_val}"
                    if pid_val in particle_info:
                        pt, eta = particle_info[pid_val]
                        if eta is not None:
                            hover_info += f" [Pt={pt:.3f} GeV, η={eta:.3f}]"
                        else:
                            hover_info += f" [Pt={pt:.3f} GeV]"
                    hover_info += "<br>"
                else:
                    hover_info += "particle_id: noise<br>"
            hover_text.append(hover_info)

        # Create figure with go.Scatter3d grouped by particle_id (like create_visual_raw_csv_spacepoints.py)
        fig = go.Figure()
        
        if pids_sampled is not None:
            # Group by particle ID for interactive legend
            unique_particles = sorted(np.unique(pids_sampled))
            colors = px.colors.qualitative.Set3
            
            for i, pid_val in enumerate(unique_particles):
                mask_pid = (pids_sampled == pid_val)
                pid_x = xs[mask_pid]
                pid_y = ys[mask_pid]
                pid_z = zs[mask_pid]
                pid_hover = [hover_text[j] for j in range(len(xs)) if mask_pid[j]]
                
                if pid_val == 0:
                    name = 'Noise'
                    color = 'gray'
                else:
                    if pid_val in particle_info:
                        pt, eta = particle_info[int(pid_val)]
                        if eta is not None:
                            name = f"Particle {int(pid_val)} [Pt={pt:.3f} GeV, η={eta:.3f}]"
                        else:
                            name = f"Particle {int(pid_val)} [Pt={pt:.3f} GeV]"
                    else:
                        name = f"Particle {int(pid_val)}"
                    color = colors[i % len(colors)]
                
                fig.add_trace(go.Scatter3d(
                    x=pid_x,
                    y=pid_y,
                    z=pid_z,
                    mode='markers',
                    marker=dict(size=3, color=color, opacity=0.8),
                    text=pid_hover,
                    hoverinfo='text',
                    name=name,
                    showlegend=True
                ))
        else:
            # No particle IDs available - color by volume label
            unique_labels = list(dict.fromkeys(labs))
            plotly_color_map = {lab: color_map.get(lab, 'gray') for lab in unique_labels}
            
            for lab in unique_labels:
                mask_lab = [l == lab for l in labs]
                lab_x = xs[mask_lab]
                lab_y = ys[mask_lab]
                lab_z = zs[mask_lab]
                lab_hover = [hover_text[j] for j in range(len(xs)) if mask_lab[j]]
                
                fig.add_trace(go.Scatter3d(
                    x=lab_x,
                    y=lab_y,
                    z=lab_z,
                    mode='markers',
                    marker=dict(size=3, color=plotly_color_map[lab], opacity=0.8),
                    text=lab_hover,
                    hoverinfo='text',
                    name=lab,
                    showlegend=True
                ))

        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{title} {info_msg}",
                x=0.5,
                xanchor='center'
            ),
            scene=dict(
                xaxis=dict(title='x (mm)', backgroundcolor="white", gridcolor="lightgray"),
                yaxis=dict(title='y (mm)', backgroundcolor="white", gridcolor="lightgray"),
                zaxis=dict(title='z (mm)', backgroundcolor="white", gridcolor="lightgray"),
                aspectmode='data'
            ),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)"
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            hovermode='closest'
        )

        out_html = Path(f"{out_prefix}.html")
        try:
            fig.write_html(str(out_html), include_plotlyjs='cdn')
        except Exception as e:
            print(f"Error writing interactive HTML: {e}")
            sys.exit(1)

        print(f"Wrote interactive HTML: {out_html}")


if __name__ == '__main__':
    main()
