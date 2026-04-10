"""
create_spacepoints_2d <out_prefix> [--hits HITS_ROOT] [--color-by particle|volume]

Produces a side-by-side 2D scatter plot replicating the style of Figure 8 in
the TrainML paper:
  - Left panel:  raw hit coordinates    — x (mm) vs y (mm)
  - Right panel: mapped coordinates     — x/r1 vs y/r1

where the mapping is:
  r1 = sqrt(x^2 + y^2 + z^2)   (3D radius)
  x2 = x / r1
  y2 = y / r1

Hits with sensitive_id == 0 are ignored (passive material, not measurements).

Color modes (--color-by):
  particle  — each unique particle ID gets its own color (default, like the paper)
  volume    — color by detector volume/layer type

Usage examples:
  python create_spacepoints_2d.py my_plot
  python create_spacepoints_2d.py my_plot --hits /path/to/hits.root
  python create_spacepoints_2d.py my_plot --color-by volume
  python create_spacepoints_2d.py my_plot --dpi 200

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
import matplotlib.cm as cm

try:
    from create_track_table import VOLUME_MAP
except Exception:
    VOLUME_MAP = {}


# ------------------------------------------------------------------ #
# Reading hits                                                        #
# ------------------------------------------------------------------ #

def read_hits(hits_path):
    """Open hits ROOT file and return raw arrays.

    Returns: (x, y, z, sensitive_id, volume_id, particle_id or None)
    """
    try:
        f = uproot.open(str(hits_path))
    except Exception as e:
        raise RuntimeError(f"Cannot open ROOT file: {e}")

    keys = list(f.keys())
    if not keys:
        raise RuntimeError("No objects found in file.")

    # Prefer a tree whose name starts with 'hits'
    chosen = next((k for k in keys if k.lower().startswith('hits')), keys[0])
    t = f[chosen]

    required = ['tx', 'ty', 'tz', 'sensitive_id', 'volume_id']
    missing = [b for b in required if b not in t.keys()]
    if missing:
        raise RuntimeError(f"Missing branches in '{chosen}': {missing}")

    x   = np.asarray(t['tx'].array())
    y   = np.asarray(t['ty'].array())
    z   = np.asarray(t['tz'].array())
    mod = np.asarray(t['sensitive_id'].array())
    vol = np.asarray(t['volume_id'].array())

    pid = None
    if 'particle_id' in t.keys():
        try:
            pid = np.asarray(t['particle_id'].array())
        except Exception:
            pass

    return x, y, z, mod, vol, pid


# ------------------------------------------------------------------ #
# Optional: load Pt / eta from particles.root                        #
# ------------------------------------------------------------------ #

def load_particle_info(hits_path):
    """Try to read pt and eta for each particle from particles.root.

    Returns a dict: {particle_id (int): (pt (float), eta (float or None))}
    Returns an empty dict if the file is missing or unreadable.
    """
    particles_path = Path(hits_path).parent / 'particles.root'
    if not particles_path.exists():
        return {}

    try:
        pf = uproot.open(str(particles_path))
        p_keys = list(pf.keys())
        chosen = next((k for k in ('particles;1', 'particles') if k in p_keys), None)
        if chosen is None:
            chosen = p_keys[0]
        pt = pf[chosen]
        pk = set(pt.keys())

        pid_b = next((b for b in ('particle_id', 'id', 'pid') if b in pk), None)
        pt_b  = next((b for b in ('pt', 'p_t', 'pT')          if b in pk), None)
        eta_b = next((b for b in ('eta', 'pseudorapidity')     if b in pk), None)

        if pid_b is None or pt_b is None:
            return {}

        def _flat(arr):
            arr = np.asarray(arr)
            if arr.dtype == object or arr.ndim > 1:
                return np.concatenate([np.atleast_1d(a) for a in arr])
            return arr

        pids = _flat(pt[pid_b].array())
        pts  = _flat(pt[pt_b].array())
        etas = _flat(pt[eta_b].array()) if eta_b else None

        info = {}
        for i in range(len(pids)):
            info[int(pids[i])] = (float(pts[i]),
                                  float(etas[i]) if etas is not None else None)
        return info

    except Exception as e:
        print(f"Warning: could not load particle info: {e}")
        return {}


# ------------------------------------------------------------------ #
# Volume label helper                                                 #
# ------------------------------------------------------------------ #

def vol_label(vid):
    try:
        vm = VOLUME_MAP.get(int(vid))
    except Exception:
        vm = None
    if vm is None:
        return 'Unknown'
    region = vm.get('region', '')
    vtype  = vm.get('type', 'Unknown')
    if region == 'barrel':
        return f"{vtype} Barrel"
    elif region == 'endcap':
        return f"{vtype} Endcap"
    return vtype


VOLUME_COLOR_MAP = {
    'Pixel Barrel':       '#1f77b4',
    'ShortStrip Barrel':  '#d62728',
    'LongStrip Barrel':   '#2ca02c',
    'Pixel Endcap':       '#9467bd',
    'ShortStrip Endcap':  '#17becf',
    'LongStrip Endcap':   '#8c564b',
    'Unknown':            '#7f7f7f',
}


# ------------------------------------------------------------------ #
# Core plot function                                                  #
# ------------------------------------------------------------------ #

def make_side_by_side(
    x, y, x2, y2,
    color_by,
    pids,
    vol_labels,
    particle_info,
    title,
    out_path,
    dpi=150,
):
    fig, (ax_raw, ax_mapped) = plt.subplots(
        1, 2,
        figsize=(12, 5),
        facecolor='white',
    )

    if color_by == 'particle' and pids is not None:
        unique_ids = sorted(np.unique(pids))
        cmap = cm.get_cmap('tab20', max(len(unique_ids), 1))

        for idx, pid_val in enumerate(unique_ids):
            mask = (pids == pid_val)
            color  = 'lightgray' if pid_val == 0 else cmap(idx % 20)
            zorder = 1           if pid_val == 0 else 2

            if pid_val == 0:
                label = 'Noise'
            elif pid_val in particle_info:
                pt, eta = particle_info[pid_val]
                label = (f"Particle {pid_val}  Pt={pt:.2f} GeV"
                         + (f"  η={eta:.2f}" if eta is not None else ""))
            else:
                label = f"Particle {pid_val}"

            kw = dict(s=6, alpha=0.7, color=color,
                      zorder=zorder, label=label, linewidths=0)
            ax_raw.scatter(x[mask],  y[mask],  **kw)
            ax_mapped.scatter(x2[mask], y2[mask], **kw)

    else:
        # Color by detector volume
        vol_arr     = np.array(vol_labels)
        unique_vols = list(dict.fromkeys(vol_labels))

        for vol in unique_vols:
            mask  = (vol_arr == vol)
            color = VOLUME_COLOR_MAP.get(vol, '#7f7f7f')
            kw = dict(s=6, alpha=0.7, color=color,
                      zorder=2, label=vol, linewidths=0)
            ax_raw.scatter(x[mask],  y[mask],  **kw)
            ax_mapped.scatter(x2[mask], y2[mask], **kw)

    # ---- axes styling ------------------------------------------------ #
    for ax in (ax_raw, ax_mapped):
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', linewidth=0.5,
                color='lightgray', zorder=0)
        ax.set_facecolor('white')
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    ax_raw.set_xlabel('x (mm)', fontsize=11)
    ax_raw.set_ylabel('y (mm)', fontsize=11)
    ax_raw.set_title('Track hits', fontsize=12, fontweight='bold')

    ax_mapped.set_xlabel('x / r₁', fontsize=11)
    ax_mapped.set_ylabel('y / r₁', fontsize=11)
    ax_mapped.set_title('Preprocessed track hits', fontsize=12, fontweight='bold')

    # Mapped coordinates are bounded in [-1, 1] by construction
    ax_mapped.set_xlim(-1.05, 1.05)
    ax_mapped.set_ylim(-1.05, 1.05)

    # ---- legend ------------------------------------------------------ #
    handles, labels_leg = ax_raw.get_legend_handles_labels()
    max_legend = 20
    if len(handles) > max_legend:
        handles    = handles[:max_legend]
        labels_leg = labels_leg[:max_legend]
        n_total    = len(np.unique(pids)) if pids is not None else '?'
        labels_leg[-1] = f"… ({n_total} total)"

    fig.legend(
        handles, labels_leg,
        loc='center right',
        bbox_to_anchor=(1.18, 0.5),
        fontsize=7,
        framealpha=0.9,
        markerscale=1.8,
        title='Legend',
        title_fontsize=8,
    )

    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote: {out_path}")


# ------------------------------------------------------------------ #
# Entry point                                                         #
# ------------------------------------------------------------------ #

def main():
    p = argparse.ArgumentParser(
        description='2D side-by-side spacepoint plot (paper style)'
    )
    p.add_argument('out_prefix',
                   help='Output prefix — image saved as <prefix>_2d.png')
    p.add_argument('--hits', '-i', default='hits.root',
                   help='Path to hits.root (default: ./hits.root)')
    p.add_argument('--color-by', choices=['particle', 'volume'],
                   default='particle',
                   help='Color by particle ID (default) or detector volume')
    p.add_argument('--dpi', type=int, default=150,
                   help='Image DPI (default: 150)')
    args = p.parse_args()

    hits_path = Path(args.hits)
    if not hits_path.exists():
        print(f"Error: hits file not found: {hits_path}")
        sys.exit(1)

    # ---- load hits --------------------------------------------------- #
    print("Reading hits …")
    try:
        rx, ry, rz, mod, vol, pid = read_hits(hits_path)
    except Exception as e:
        print(e)
        sys.exit(1)

    mask = (mod != 0)
    if not np.any(mask):
        print("No measurement hits found (sensitive_id != 0). Exiting.")
        sys.exit(1)

    x    = rx[mask]
    y    = ry[mask]
    z    = rz[mask]
    pids = pid[mask] if pid is not None else None
    vols = vol[mask]
    print(f"  {len(x)} measurement hits loaded.")

    # ---- mapped coordinates ------------------------------------------ #
    r1      = np.sqrt(x**2 + y**2 + z**2)
    safe_r1 = np.where(r1 == 0, 1.0, r1)
    x2      = x / safe_r1
    y2      = y / safe_r1

    # ---- volume labels ----------------------------------------------- #
    vol_labels = [vol_label(v) for v in vols]

    # ---- particle info (optional) ------------------------------------ #
    particle_info = {}
    if args.color_by == 'particle':
        print("Loading particle info …")
        particle_info = load_particle_info(hits_path)
        if particle_info:
            print(f"  Loaded info for {len(particle_info)} particles.")
        else:
            print("  particles.root not found — legend will show IDs only.")

    # ---- figure title ------------------------------------------------ #
    n_particles = len(np.unique(pids[pids > 0])) if pids is not None else 0
    title = f"{len(x)} hits"
    if n_particles:
        title += f"  —  {n_particles} particles"

    # ---- render ------------------------------------------------------ #
    out_path = Path(f"{args.out_prefix}_2d.png")
    print("Rendering plot …")
    try:
        make_side_by_side(
            x, y, x2, y2,
            color_by=args.color_by,
            pids=pids,
            vol_labels=vol_labels,
            particle_info=particle_info,
            title=title,
            out_path=out_path,
            dpi=args.dpi,
        )
    except Exception as e:
        import traceback
        print(f"Error creating plot: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()