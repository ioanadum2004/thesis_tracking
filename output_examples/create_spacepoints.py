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
    v = vol[mask]

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
        'ShortStrip Barrel': 'green',
        'LongStrip Barrel': 'orange',
        'Pixel Endcap': 'magenta',
        'ShortStrip Endcap': 'cyan',
        'LongStrip Endcap': 'purple',
        'Unknown': 'gray',
    }

    # Try to infer primary particle Pt/eta from a nearby particles.root (same dir as hits)
    particle_pt = None
    particle_eta = None
    try:
        if pid is not None:
            meas_indices = np.where(mask)[0]
            sel_pids = pid[meas_indices]
            unique_pids, counts = np.unique(sel_pids, return_counts=True)
            if unique_pids.size > 0:
                primary_pid = int(unique_pids[np.argmax(counts)])
            else:
                primary_pid = None
        else:
            primary_pid = None

        if primary_pid is not None:
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
                        matches = np.where(part_pid_arr == primary_pid)[0]
                        if matches.size > 0:
                            particle_pt = float(part_pt_arr[matches[0]])
                            if eta_branch is not None:
                                particle_eta = float(np.asarray(ptree[eta_branch].array())[matches[0]])
    except Exception:
        particle_pt = None
        particle_eta = None

    out_png = Path(f"{out_prefix}.png")
    title = f"Spacepoints ({len(x)} hits)"
    if particle_pt is not None:
        # append Pt/eta to title
        try:
            title += f" — Pt={particle_pt:.3f} , eta={particle_eta:.3f}"
        except Exception:
            title += f" — Pt={particle_pt}"

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
            import plotly.express as px
            import pandas as pd
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
            labs = [labels[i] for i in idx]
            info_msg = f"(downsampled from {npoints} to {max_points} points for interactive view)"
        else:
            xs = x
            ys = y
            zs = z
            labs = labels
            info_msg = ""

        df = pd.DataFrame({'x': xs, 'y': ys, 'z': zs, 'label': labs})
        # Map labels to colors for Plotly
        unique_labels = list(dict.fromkeys(df['label'].tolist()))
        plotly_color_map = {lab: color_map.get(lab, 'gray') for lab in unique_labels}

        fig = px.scatter_3d(df, x='x', y='y', z='z', color='label', title=f"{title} {info_msg}",
                color_discrete_map=plotly_color_map)
        # set marker size and opacity
        fig.update_traces(marker=dict(size=3, opacity=0.8))

        out_html = Path(f"{out_prefix}.html")
        try:
            fig.write_html(str(out_html), include_plotlyjs='cdn')
        except Exception as e:
            print(f"Error writing interactive HTML: {e}")
            sys.exit(1)

        print(f"Wrote interactive HTML: {out_html}")


if __name__ == '__main__':
    main()
