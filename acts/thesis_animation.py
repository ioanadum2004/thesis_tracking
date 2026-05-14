"""
thesis_animation.py  -  Full pipeline animation for bachelor's thesis video.

Produces 4 animation clips (each saved as a separate MP4) plus one combined
MP4 that plays them back-to-back:

  1. act1_detector.mp4   - A muon flying through concentric detector layers
  2. act2_seeds.mp4      - Hit cloud appears, then true vs fake seed formation
  3. act3_filter.mp4     - ML filter scoring seeds (bouncer metaphor)
  4. act4_results.mp4    - Before/after comparison: fake rate per pT bin

Usage (run on the Nikhef cluster, pointing at one event's output):
  python thesis_animation.py \
      --hits  /path/to/event/hits.root \
      --particles /path/to/event/particles.root \
      --seeds /path/to/estimatedparams.root \
      --outdir ./animation_out

All arguments are optional - if a file is missing the script falls back to
synthetic toy data for that act so you can preview everything immediately.

Dependencies: uproot, numpy, matplotlib (with ffmpeg writer for MP4)
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch

# ── colour palette (dark physics aesthetic) ──────────────────────────────────
# BG        = '#0a0e1a'   # near-black blue
# GRID      = '#1a2035'
# CYAN      = '#00e5ff'
# ORANGE    = '#ff6b35'
# GREEN     = '#39ff14'
# MAGENTA   = '#ff00ff'
# YELLOW    = '#ffd600'
# GREY      = '#4a5568'
# WHITE     = '#e8eaf6'
# PANEL_BG  = '#111827'

BG        = '#ffffff'   # near-black blue
GRID      = '#0a0e1a'
CYAN      = '#a5dcd2'
ORANGE    = '#f79d1d'
GREEN     = '#c7e792'
MAGENTA   = '#cd376a'
YELLOW    = '#ffdd00'
GREY      = '#a6a6a6'
WHITE     = '#ffffff'
PANEL_BG  = '#fc998e'

plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor':   BG,
    'text.color':       WHITE,
    'axes.labelcolor':  WHITE,
    'xtick.color':      GREY,
    'ytick.color':      GREY,
    'axes.edgecolor':   GRID,
    'font.family':      'monospace',
})

# ── helpers ───────────────────────────────────────────────────────────────────

# def save_anim(anim_obj, path, fps=30, dpi=120):
#     writer = animation.FFMpegWriter(fps=fps, bitrate=2000,
#                                     extra_args=['-vcodec', 'libx264',
#                                                 '-pix_fmt', 'yuv420p'])
#     anim_obj.save(str(path), writer=writer, dpi=dpi)
#     print(f"  saved → {path}")

# def save_anim(anim_obj, path, fps=30, dpi=120):
#     path = Path(path)
#     # Try ffmpeg first
#     ffmpeg_candidates = [
#         '/usr/bin/ffmpeg',
#         '/usr/local/bin/ffmpeg',
#         '/cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc13-opt/bin/ffmpeg',
#     ]
#     ffmpeg_path = None
#     for candidate in ffmpeg_candidates:
#         if Path(candidate).is_file():
#             ffmpeg_path = candidate
#             break
#     # Also try whatever's in PATH
#     if ffmpeg_path is None:
#         import shutil
#         ffmpeg_path = shutil.which('ffmpeg')

#     if ffmpeg_path is not None:
#         matplotlib.rcParams['animation.ffmpeg_path'] = ffmpeg_path
#         out_path = path.with_suffix('.mp4')
#         writer = animation.FFMpegWriter(fps=fps, bitrate=2000,
#                                         extra_args=['-vcodec', 'libx264',
#                                                     '-pix_fmt', 'yuv420p'])
#         anim_obj.save(str(out_path), writer=writer, dpi=dpi)
#         print(f"  saved → {out_path}")
#     else:
#         # Fallback: Pillow GIF (no external binary needed)
#         out_path = path.with_suffix('.gif')
#         writer = animation.PillowWriter(fps=fps)
#         anim_obj.save(str(out_path), writer=writer, dpi=dpi)
#         print(f"  saved (gif fallback) → {out_path}")

def try_load_hits(hits_path):
    """Return (x, y, z, vol, pid) arrays or None on failure."""
    try:
        import uproot
        f = uproot.open(str(hits_path))
        keys = list(f.keys())
        chosen = next((k for k in keys if k.lower().startswith('hits')), keys[0])
        t = f[chosen]
        req = ['tx', 'ty', 'tz', 'sensitive_id', 'volume_id']
        if any(r not in t.keys() for r in req):
            return None
        tx  = np.asarray(t['tx'].array())
        ty  = np.asarray(t['ty'].array())
        tz  = np.asarray(t['tz'].array())
        mod = np.asarray(t['sensitive_id'].array())
        vol = np.asarray(t['volume_id'].array())
        pid = np.asarray(t['particle_id'].array()) if 'particle_id' in t.keys() else np.zeros_like(tx, dtype=int)
        mask = mod != 0
        return tx[mask], ty[mask], tz[mask], vol[mask], pid[mask]
    except Exception as e:
        warnings.warn(f"Could not load hits: {e}")
        return None

def save_anim(anim_obj, path, fps=30, dpi=120):
    path = Path(path)
    ffmpeg_candidates = [
        '/usr/bin/ffmpeg',
        '/usr/local/bin/ffmpeg',
        '/cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc13-opt/bin/ffmpeg',
    ]
    ffmpeg_path = None
    for candidate in ffmpeg_candidates:
        if Path(candidate).is_file():
            ffmpeg_path = candidate
            break
    if ffmpeg_path is None:
        import shutil
        ffmpeg_path = shutil.which('ffmpeg')

    if ffmpeg_path is not None:
        matplotlib.rcParams['animation.ffmpeg_path'] = ffmpeg_path
        out_path = path.with_suffix('.mp4')
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000,
                                        extra_args=['-vcodec', 'libx264',
                                                    '-pix_fmt', 'yuv420p'])
        anim_obj.save(str(out_path), writer=writer, dpi=dpi)
        print(f"  saved → {out_path}")
    else:
        out_path = path.with_suffix('.gif')
        writer = animation.PillowWriter(fps=fps)
        anim_obj.save(str(out_path), writer=writer, dpi=dpi)
        print(f"  saved (gif fallback) → {out_path}")

    return out_path 

def try_load_particles(particles_path):
    """Return dict {particle_id: (pt, eta)} or {}."""
    try:
        import uproot
        f = uproot.open(str(particles_path))
        keys = list(f.keys())
        chosen = next((k for k in keys if 'particle' in k.lower()), keys[0])
        t = f[chosen]
        pid_arr = np.asarray(t['particle_id'].array())
        pt_arr  = np.asarray(t['pt'].array())
        eta_arr = np.asarray(t['eta'].array()) if 'eta' in t.keys() else np.zeros_like(pt_arr)
        # flatten jagged if needed
        def flat(a):
            if a.dtype == object:
                return np.concatenate([np.atleast_1d(x) for x in a])
            return a
        pid_arr, pt_arr, eta_arr = flat(pid_arr), flat(pt_arr), flat(eta_arr)
        return {int(p): (float(pt), float(eta)) for p, pt, eta in zip(pid_arr, pt_arr, eta_arr)}
    except Exception as e:
        warnings.warn(f"Could not load particles: {e}")
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# ACT 1 – Detector layers + muon track
# ═══════════════════════════════════════════════════════════════════════════════

def make_act1(outdir, hits_data=None, particles=None, fps=30):
    """Animate a charged particle curving through detector barrel layers."""
    print("  building act1: detector + track…")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)
    ax.set_aspect('equal')
    ax.set_xlim(-220, 220)
    ax.set_ylim(-220, 220)
    ax.axis('off')

    # --- detector layer radii (generic detector approximate values) -----------
    layer_radii = [32, 72, 116, 172, 228]   # mm  (pixel + strip layers)
    layer_names = ['Pixel L1', 'Pixel L2', 'Pixel L3', 'Strip L1', 'Strip L2']
    layer_colors = [CYAN, CYAN, CYAN, ORANGE, ORANGE]

    total_frames = fps * 7   # 7 seconds

    # pre-draw static rings (they fade in during first second)
    rings = []
    ring_labels = []
    for r, name, col in zip(layer_radii, layer_names, layer_colors):
        circle = plt.Circle((0, 0), r, color=col, fill=False,
                             linewidth=1.5, alpha=0.0, linestyle='--')
        ax.add_patch(circle)
        rings.append(circle)
        lbl = ax.text(r * 0.707 + 4, r * 0.707 + 4, name,
                      color=col, fontsize=7, alpha=0.0,
                      path_effects=[pe.withStroke(linewidth=2, foreground=BG)])
        ring_labels.append(lbl)

    # beampipe
    bp = plt.Circle((0, 0), 23, color=GREY, fill=False, linewidth=1, alpha=0.0)
    ax.add_patch(bp)
    bp_lbl = ax.text(0, -28, 'beam pipe', color=GREY, fontsize=7,
                     ha='center', alpha=0.0)

    # vertex marker
    vtx = ax.plot(0, 0, 'o', color=YELLOW, markersize=6, alpha=0.0, zorder=10)[0]
    vtx_lbl = ax.text(4, 4, 'collision vertex', color=YELLOW, fontsize=8,
                      alpha=0.0,
                      path_effects=[pe.withStroke(linewidth=2, foreground=BG)])

    # title
    title_txt = ax.text(0, 205, 'ACTS Generic Detector  |  Transverse (x–y) View',
                        color=WHITE, fontsize=11, ha='center', alpha=0.0,
                        fontweight='bold')

    # --- generate curved track (helix projected to x-y) ----------------------
    # A muon with pT~0.3 GeV in B=2T has radius R = pT/(0.3*B) ≈ 0.5 m = 500 mm
    pt_gev = 0.28
    B_T    = 2.0
    R_mm   = pt_gev / (0.3 * B_T) * 1000   # ~467 mm

    phi0   = np.deg2rad(35)   # initial direction
    charge = +1

    def helix_xy(s_arr, R, phi0, charge):
        """s_arr: arc-length parameter in mm."""
        cx = -charge * R * np.sin(phi0)
        cy =  charge * R * np.cos(phi0)
        theta = phi0 + charge * s_arr / R
        x = cx + R * np.sin(theta)
        y = cy - R * np.cos(theta)
        return x, y

    s_total = np.linspace(0, 380, 1000)
    tx, ty = helix_xy(s_total, R_mm, phi0, charge)

    # hit positions: find where track crosses each layer ring
    hit_pts = []
    for r in layer_radii:
        dists = np.sqrt(tx**2 + ty**2)
        # find first crossing from inside
        cross = np.where((dists[:-1] < r) & (dists[1:] >= r))[0]
        if len(cross):
            idx = cross[0]
            hit_pts.append((tx[idx], ty[idx]))

    # drawn track line (built up frame by frame)
    track_line, = ax.plot([], [], color=GREEN, linewidth=2.0, alpha=0.9, zorder=8)
    glow_line,  = ax.plot([], [], color=GREEN, linewidth=6.0, alpha=0.15, zorder=7)

    # hit sparks
    hit_scatters = []
    for _ in hit_pts:
        sc = ax.plot([], [], 'o', color=YELLOW, markersize=10,
                     markerfacecolor='none', markeredgewidth=2,
                     alpha=0.0, zorder=9)[0]
        hit_scatters.append(sc)

    # annotation
    ann_txt = ax.text(0, -195,
                      f'μ⁺   pT = {pt_gev:.2f} GeV   B = {B_T} T   R_curve ≈ {R_mm:.0f} mm',
                      color=CYAN, fontsize=9, ha='center', alpha=0.0)

    subtitle = ax.text(0, -210,
                       'Charged particles curve in the magnetic field — '
                       'curvature encodes momentum',
                       color=GREY, fontsize=8, ha='center', alpha=0.0,
                       style='italic')

    ring_fade_frames = fps  # 1 s
    track_start      = fps  # track starts drawing at 1 s
    track_frames     = fps * 3  # 3 s to draw full track

    def init():
        track_line.set_data([], [])
        glow_line.set_data([], [])
        return ([track_line, glow_line, vtx, vtx_lbl, bp, bp_lbl,
                 ann_txt, subtitle, title_txt] +
                rings + ring_labels + hit_scatters)

    def update(frame):
        # 1) fade in rings
        ralpha = min(1.0, frame / ring_fade_frames)
        bp.set_alpha(ralpha * 0.6)
        bp_lbl.set_alpha(ralpha * 0.6)
        vtx.set_alpha(ralpha)
        vtx_lbl.set_alpha(ralpha * 0.8)
        title_txt.set_alpha(ralpha)
        for ring, lbl, col in zip(rings, ring_labels, layer_colors):
            ring.set_alpha(ralpha * 0.7)
            lbl.set_alpha(ralpha * 0.7)

        # 2) draw track
        if frame >= track_start:
            prog = min(1.0, (frame - track_start) / track_frames)
            n = max(2, int(prog * len(s_total)))
            track_line.set_data(tx[:n], ty[:n])
            glow_line.set_data(tx[:n], ty[:n])

            # reveal hit markers
            cur_r = np.sqrt(tx[n-1]**2 + ty[n-1]**2)
            for i, (hx, hy) in enumerate(hit_pts):
                hr = np.sqrt(hx**2 + hy**2)
                if cur_r >= hr:
                    hit_scatters[i].set_data([hx], [hy])
                    hit_scatters[i].set_alpha(0.9)

        # 3) fade in annotation near end
        ann_alpha = max(0.0, min(1.0, (frame - track_start - track_frames//2)
                                 / (fps * 0.8)))
        ann_txt.set_alpha(ann_alpha)
        subtitle.set_alpha(ann_alpha * 0.8)

        return ([track_line, glow_line, vtx, vtx_lbl, bp, bp_lbl,
                 ann_txt, subtitle, title_txt] +
                rings + ring_labels + hit_scatters)

    anim = animation.FuncAnimation(fig, update, frames=total_frames,
                                   init_func=init, blit=True)
    out = outdir / 'act1_detector.mp4'
    save_anim(anim, out, fps=fps)
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# ACT 2 – Hit cloud → seed formation (true vs fake)
# ═══════════════════════════════════════════════════════════════════════════════

def make_act2(outdir, hits_data=None, particles=None, fps=30):
    """Show spacepoints appearing, then highlight a true seed and a fake seed."""
    print("  building act2: hits + seed formation…")

    # --- choose a few particles from real data, or synthesise ----------------
    rng = np.random.default_rng(seed=7)

    if hits_data is not None:
        tx, ty, tz, vol, pid = hits_data
        # project to x-y; pick up to 300 hits for clarity
        x_all, y_all, pid_all = tx, ty, pid
        if len(x_all) > 400:
            idx = rng.choice(len(x_all), 400, replace=False)
            x_all, y_all, pid_all = x_all[idx], y_all[idx], pid_all[idx]

        unique_pids = [p for p in np.unique(pid_all) if p != 0]
        # pick first particle with >=3 hits as "true" track
        true_pid = None
        for p in unique_pids:
            if np.sum(pid_all == p) >= 3:
                true_pid = p
                break

        if true_pid is not None:
            mask_t = pid_all == true_pid
            true_hits = list(zip(x_all[mask_t], y_all[mask_t]))[:3]
            # fake: pick 3 hits from 3 different other particles
            fake_hits = []
            for p in unique_pids:
                if p != true_pid:
                    m = pid_all == p
                    if np.sum(m) >= 1:
                        idx0 = np.where(m)[0][0]
                        fake_hits.append((x_all[idx0], y_all[idx0]))
                    if len(fake_hits) == 3:
                        break
        else:
            true_hits, fake_hits = None, None
    else:
        x_all, y_all, pid_all = None, None, None
        true_hits, fake_hits  = None, None

    # fallback synthetic data
    n_particles = 8
    if x_all is None:
        layer_r = [32, 72, 116, 172]
        all_pts = []
        all_pids = []
        syn_true_hits = []
        syn_fake_hits = []
        for p in range(n_particles):
            phi0 = rng.uniform(0, 2 * np.pi)
            pt   = rng.uniform(0.1, 0.5)
            R    = pt / (0.3 * 2.0) * 1000
            charge = rng.choice([-1, 1])
            cx = -charge * R * np.sin(phi0)
            cy =  charge * R * np.cos(phi0)
            hits_p = []
            for r in layer_r:
                # analytic crossing
                # circle-circle intersection of (cx,cy,R) and (0,0,r)
                d = np.sqrt(cx**2 + cy**2)
                if abs(R - r) <= d <= R + r:
                    a = (R**2 - r**2 + d**2) / (2 * d)
                    h = np.sqrt(max(0, R**2 - a**2))
                    mx = cx * a / d
                    my = cy * a / d
                    px1 = mx + h * cy / d
                    py1 = my - h * cx / d
                    hits_p.append((px1 + rng.normal(0, 0.5),
                                   py1 + rng.normal(0, 0.5)))
            all_pts.extend(hits_p)
            all_pids.extend([p] * len(hits_p))
            if p == 0 and len(hits_p) >= 3:
                syn_true_hits = hits_p[:3]
            if p >= 1 and len(syn_fake_hits) < 3 and hits_p:
                syn_fake_hits.append(hits_p[0])

        x_all   = np.array([p[0] for p in all_pts])
        y_all   = np.array([p[1] for p in all_pts])
        pid_all = np.array(all_pids)

        if true_hits is None:
            true_hits = syn_true_hits
        if fake_hits is None:
            fake_hits = syn_fake_hits

    # ensure 3 hits each
    true_hits = true_hits[:3] if true_hits and len(true_hits) >= 3 else None
    fake_hits = fake_hits[:3] if fake_hits and len(fake_hits) >= 3 else None

    # ── figure setup ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)
    ax.set_aspect('equal')
    lim = 200
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axis('off')

    # detector rings
    for r, col in zip([32, 72, 116, 172], [CYAN, CYAN, ORANGE, ORANGE]):
        ax.add_patch(plt.Circle((0, 0), r, color=col, fill=False,
                                linewidth=1, alpha=0.2, linestyle='--'))

    title_txt = ax.text(0, 185, 'Spacepoints & Seed Formation',
                        color=WHITE, fontsize=12, ha='center',
                        fontweight='bold', alpha=0.0)

    # all hits scatter (start invisible, fade in)
    scat = ax.scatter(x_all, y_all, c=GREY, s=12, alpha=0.0, zorder=5)

    # true seed elements
    true_scat = ax.scatter([], [], c=GREEN, s=60, zorder=10,
                           edgecolors='white', linewidths=0.8, alpha=0.0)
    true_lines = LineCollection([], colors=GREEN, linewidths=2, alpha=0.0, zorder=9)
    ax.add_collection(true_lines)
    true_lbl = ax.text(0, 0, '', color=GREEN, fontsize=9, alpha=0.0,
                       path_effects=[pe.withStroke(linewidth=2, foreground=BG)])

    # fake seed elements
    fake_scat = ax.scatter([], [], c=ORANGE, s=60, zorder=10,
                           marker='X', edgecolors='white', linewidths=0.8,
                           alpha=0.0)
    fake_lines = LineCollection([], colors=ORANGE, linewidths=2,
                                linestyles='dashed', alpha=0.0, zorder=9)
    ax.add_collection(fake_lines)
    fake_lbl = ax.text(0, 0, '', color=ORANGE, fontsize=9, alpha=0.0,
                       path_effects=[pe.withStroke(linewidth=2, foreground=BG)])

    # legend patches
    leg_true = mpatches.Patch(color=GREEN,  label='True seed  (3 hits, same particle)')
    leg_fake = mpatches.Patch(color=ORANGE, label='Fake seed  (hits from different particles)')
    legend = ax.legend(handles=[leg_true, leg_fake], loc='lower right',
                       facecolor=PANEL_BG, edgecolor=GREY,
                       labelcolor=WHITE, fontsize=9)
    legend.set_alpha(0.0)

    subtitle = ax.text(0, -190,
                       'At low pT, tight curvature causes accidental hit overlap → '
                       'many fake seeds',
                       color=GREY, fontsize=8, ha='center',
                       style='italic', alpha=0.0)

    # ── frame schedule ────────────────────────────────────────────────────────
    # 0  –  30 : title fade in
    # 30 –  90 : hits fade in
    # 90 – 150 : true seed highlight
    # 150– 210 : fake seed highlight
    # 210– 240 : hold

    total_frames = fps * 9

    def seg(frame, start, dur):
        return max(0.0, min(1.0, (frame - start) / dur))

    def seed_segments(hits):
        """Return line segments bottom→mid, mid→top for a seed triplet."""
        if hits and len(hits) >= 3:
            return [[(hits[0][0], hits[0][1]), (hits[1][0], hits[1][1])],
                    [(hits[1][0], hits[1][1]), (hits[2][0], hits[2][1])]]
        return []

    def init():
        scat.set_alpha(0.0)
        true_scat.set_offsets(np.empty((0, 2)))
        fake_scat.set_offsets(np.empty((0, 2)))
        true_lines.set_segments([])
        fake_lines.set_segments([])
        return [scat, true_scat, fake_scat, true_lines, fake_lines,
                true_lbl, fake_lbl, title_txt, subtitle]

    def update(frame):
        # title
        title_txt.set_alpha(seg(frame, 0, fps))

        # hits cloud
        hit_alpha = seg(frame, fps, fps * 2)
        scat.set_alpha(hit_alpha * 0.55)

        # true seed
        t_alpha = seg(frame, fps * 3, fps * 1.5)
        if true_hits and t_alpha > 0:
            true_scat.set_offsets(np.array(true_hits))
            true_scat.set_alpha(t_alpha)
            segs = seed_segments(true_hits)
            prog_segs = segs[:max(0, int(t_alpha * 2 + 0.5))]
            true_lines.set_segments(prog_segs)
            true_lines.set_alpha(t_alpha * 0.9)
            if t_alpha > 0.5:
                cx = np.mean([h[0] for h in true_hits])
                cy = np.mean([h[1] for h in true_hits])
                true_lbl.set_position((cx + 8, cy + 8))
                true_lbl.set_text('✓ True seed')
                true_lbl.set_alpha((t_alpha - 0.5) * 2)

        # fake seed
        f_alpha = seg(frame, fps * 5, fps * 1.5)
        if fake_hits and f_alpha > 0:
            fake_scat.set_offsets(np.array(fake_hits))
            fake_scat.set_alpha(f_alpha)
            segs = seed_segments(fake_hits)
            prog_segs = segs[:max(0, int(f_alpha * 2 + 0.5))]
            fake_lines.set_segments(prog_segs)
            fake_lines.set_alpha(f_alpha * 0.9)
            if f_alpha > 0.5:
                cx = np.mean([h[0] for h in fake_hits])
                cy = np.mean([h[1] for h in fake_hits])
                fake_lbl.set_position((cx + 8, cy - 14))
                fake_lbl.set_text('✗ Fake seed')
                fake_lbl.set_alpha((f_alpha - 0.5) * 2)

        # legend + subtitle
        leg_alpha = seg(frame, fps * 5, fps)
        for lh in legend.legend_handles:
            lh.set_alpha(leg_alpha)
        for txt in legend.get_texts():
            txt.set_alpha(leg_alpha)
        legend.get_frame().set_alpha(leg_alpha * 0.8)
        subtitle.set_alpha(leg_alpha * 0.8)

        return [scat, true_scat, fake_scat, true_lines, fake_lines,
                true_lbl, fake_lbl, title_txt, subtitle]

    anim = animation.FuncAnimation(fig, update, frames=total_frames,
                                   init_func=init, blit=True)
    out = outdir / 'act2_seeds.mp4'
    save_anim(anim, out, fps=fps)
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# ACT 3 – ML seed filter (bouncer metaphor)
# ═══════════════════════════════════════════════════════════════════════════════

def make_act3(outdir, fps=30):
    """Animate seeds arriving at the ML filter; green pass, red blocked."""
    print("  building act3: ML filter…")

    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # ── static layout ─────────────────────────────────────────────────────────
    # pipeline boxes: [Simulation] → [Seeding] → [ML Filter] → [CKF]
    boxes = [
        (1.2, 3.5, 'Simulation\n(ACTS)', CYAN),
        (3.8, 3.5, 'Seeding\n(GridTriplet)', ORANGE),
        (6.5, 3.5, 'ML Filter\n(MLP / LightGBM)', MAGENTA),
        (9.8, 3.5, 'CKF\n(Track Fit)', GREEN),
    ]

    box_patches = []
    box_texts   = []
    for bx, by, label, col in boxes:
        rect = mpatches.FancyBboxPatch((bx - 0.9, by - 0.55), 1.8, 1.1,
                                       boxstyle='round,pad=0.08',
                                       facecolor=PANEL_BG,
                                       edgecolor=col, linewidth=2,
                                       alpha=0.0, zorder=5)
        ax.add_patch(rect)
        box_patches.append(rect)
        txt = ax.text(bx, by, label, color=col, fontsize=8,
                      ha='center', va='center', fontweight='bold',
                      alpha=0.0, zorder=6)
        box_texts.append(txt)

    # arrows between boxes
    arrow_xs = [(2.1, 2.9), (4.7, 5.6), (7.4, 8.9)]
    arrow_objs = []
    for x0, x1 in arrow_xs:
        arr = ax.annotate('', xy=(x1, 3.5), xytext=(x0, 3.5),
                          arrowprops=dict(arrowstyle='->', color=GREY,
                                         lw=1.5),
                          annotation_clip=False, alpha=0.0)
        arrow_objs.append(arr)

    title_txt = ax.text(6, 6.5, 'ML Seed Filter Pipeline',
                        color=WHITE, fontsize=13, ha='center',
                        fontweight='bold', alpha=0.0)

    filter_lbl = ax.text(6.5, 4.8,
                         '"Bouncer" — scores each seed before the\n'
                         'expensive CKF runs',
                         color=MAGENTA, fontsize=8, ha='center',
                         alpha=0.0, style='italic')

    # ── animated seed dots ────────────────────────────────────────────────────
    # seeds travel from seeding box (3.8) to filter (6.5)
    # then either pass (→ CKF, green) or get blocked (drop down, red)

    n_seeds      = 18
    fake_indices = {2, 5, 8, 11, 14, 16}   # which are fake
    seed_colors  = [ORANGE if i in fake_indices else GREEN
                    for i in range(n_seeds)]

    # stagger seeds: each starts at a different frame
    stagger = 8   # frames between seeds
    travel_frames = fps   # frames to cross from seeding to filter

    # each seed: phase = 'travel' | 'pass' | 'block'
    seed_dots = []
    seed_labels_dots = []
    for i in range(n_seeds):
        dot, = ax.plot([], [], 'o', markersize=8,
                       color=seed_colors[i], alpha=0.0, zorder=10)
        seed_dots.append(dot)

    blocked_xs = []
    pass_xs    = []

    stat_lbl = ax.text(6.5, 0.4, '', color=WHITE, fontsize=9,
                       ha='center', alpha=0.8)

    total_frames = fps * 10

    # compute per-seed position at each frame
    def seed_pos(i, frame):
        start_f = 2 * fps + i * stagger   # travel begins
        if frame < start_f:
            return None, 'wait'
        elapsed = frame - start_f
        # phase 1: travel to filter (3.8 → 6.5)
        if elapsed < travel_frames:
            prog = elapsed / travel_frames
            x = 3.8 + (6.5 - 3.8) * prog
            y = 3.5 + rng.uniform(-0.3, 0.3) * np.sin(prog * np.pi)
            return (x, y), 'travel'
        # phase 2: judged at filter
        if i not in fake_indices:
            # pass: continue to CKF
            elapsed2 = elapsed - travel_frames
            if elapsed2 < travel_frames:
                prog = elapsed2 / travel_frames
                x = 6.5 + (9.8 - 6.5) * prog
                y = 3.5
                return (x, y), 'pass'
            return None, 'done'
        else:
            # blocked: drop down
            elapsed2 = elapsed - travel_frames
            drop_frames = int(fps * 0.6)
            if elapsed2 < drop_frames:
                prog = elapsed2 / drop_frames
                x = 6.5
                y = 3.5 - 2.0 * prog
                return (x, y), 'block'
            return None, 'done'

    def init():
        for d in seed_dots:
            d.set_data([], [])
        stat_lbl.set_text('')
        return seed_dots + [stat_lbl, title_txt, filter_lbl]

    def update(frame):
        # fade in layout
        layout_alpha = min(1.0, frame / fps)
        title_txt.set_alpha(layout_alpha)
        for bp2, bt in zip(box_patches, box_texts):
            bp2.set_alpha(layout_alpha * 0.9)
            bt.set_alpha(layout_alpha)
        for arr in arrow_objs:
            arr.set_alpha(layout_alpha * 0.8)

        # filter label
        filter_lbl.set_alpha(min(1.0, max(0.0,
                                          (frame - fps * 2) / fps)))

        # move seeds
        n_passed = 0
        n_blocked = 0
        for i, dot in enumerate(seed_dots):
            pos, phase = seed_pos(i, frame)
            if pos is None:
                dot.set_data([], [])
            else:
                dot.set_data([pos[0]], [pos[1]])
                dot.set_alpha(0.9)
            if phase == 'done':
                if i not in fake_indices:
                    n_passed += 1
                else:
                    n_blocked += 1

        # stats label
        total_done = n_passed + n_blocked
        if total_done > 0:
            n_total = n_seeds
            stat_lbl.set_text(
                f'{n_total} seeds → {n_total - len(fake_indices)} passed  '
                f'({100*(n_total-len(fake_indices))//n_total}%)   '
                f'{len(fake_indices)} blocked  '
                f'({100*len(fake_indices)//n_total}%)'
            )

        return seed_dots + [stat_lbl, title_txt, filter_lbl]

    anim = animation.FuncAnimation(fig, update, frames=total_frames,
                                   init_func=init, blit=True)
    out = outdir / 'act3_filter.mp4'
    save_anim(anim, out, fps=fps)
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# ACT 4 – Before / after results (fake rate per pT bin)
# ═══════════════════════════════════════════════════════════════════════════════

def make_act4(outdir, fps=30):
    """Animate a bar chart comparison: baseline vs ML filter fake rate per bin."""
    print("  building act4: results…")

    # Representative numbers from your thesis (edit as needed)
    pt_bins   = ['0.10–0.15', '0.15–0.20', '0.20–0.25',
                 '0.25–0.30', '0.30–0.40', '0.40–0.50']
    fake_base = np.array([36.2, 28.4, 21.1, 15.3, 10.2, 7.4])   # %
    fake_mlp  = np.array([14.8, 11.2,  8.6,  6.1,  4.3, 3.2])   # %  (example)
    fake_lgbm = np.array([12.1,  9.8,  7.2,  5.0,  3.6, 2.8])   # %  (example)

    eff_base  = np.array([78.0, 82.0, 85.0, 88.0, 91.0, 93.0])   # %
    eff_mlp   = np.array([74.0, 79.0, 83.0, 86.5, 90.0, 92.5])   # %
    eff_lgbm  = np.array([73.0, 78.0, 82.0, 86.0, 89.5, 92.0])   # %

    x = np.arange(len(pt_bins))
    w = 0.26

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)
        ax.tick_params(colors=GREY, labelsize=8)
        ax.spines[:].set_color(GRID)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
    fig.patch.set_facecolor(BG)

    # pre-create bars at height 0
    bars_base1 = ax1.bar(x - w, np.zeros(len(x)), w, color=GREY,
                         label='Baseline (no filter)', alpha=0.85)
    bars_mlp1  = ax1.bar(x,     np.zeros(len(x)), w, color=CYAN,
                         label='MLP filter', alpha=0.85)
    bars_lgbm1 = ax1.bar(x + w, np.zeros(len(x)), w, color=MAGENTA,
                         label='LightGBM filter', alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels(pt_bins, rotation=30, ha='right', color=GREY, fontsize=7)
    ax1.set_ylabel('Fake rate  (%)', color=WHITE, fontsize=9)
    ax1.set_xlabel('pT bin (GeV)', color=WHITE, fontsize=9)
    ax1.set_ylim(0, 42)
    ax1.set_title('Fake Rate per pT Bin', color=WHITE, fontsize=10,
                  fontweight='bold', pad=10)
    ax1.legend(facecolor=PANEL_BG, edgecolor=GREY, labelcolor=WHITE,
               fontsize=8)
    ax1.axhline(0, color=GRID, linewidth=0.5)

    bars_base2 = ax2.bar(x - w, np.zeros(len(x)), w, color=GREY,
                         label='Baseline', alpha=0.85)
    bars_mlp2  = ax2.bar(x,     np.zeros(len(x)), w, color=CYAN,
                         label='MLP filter', alpha=0.85)
    bars_lgbm2 = ax2.bar(x + w, np.zeros(len(x)), w, color=MAGENTA,
                         label='LightGBM filter', alpha=0.85)

    ax2.set_xticks(x)
    ax2.set_xticklabels(pt_bins, rotation=30, ha='right', color=GREY, fontsize=7)
    ax2.set_ylabel('Track efficiency  (%)', color=WHITE, fontsize=9)
    ax2.set_xlabel('pT bin (GeV)', color=WHITE, fontsize=9)
    ax2.set_ylim(60, 100)
    ax2.set_title('Track Efficiency per pT Bin', color=WHITE, fontsize=10,
                  fontweight='bold', pad=10)
    ax2.legend(facecolor=PANEL_BG, edgecolor=GREY, labelcolor=WHITE,
               fontsize=8)

    fig.suptitle('ML Seed Filter  -  Results', color=WHITE, fontsize=13,
                 fontweight='bold', y=1.01)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    grow_frames = fps * 2   # bars grow over 2 s
    total_frames = fps * 6

    def update(frame):
        prog = min(1.0, frame / grow_frames)
        # ease-in-out
        prog_e = prog * prog * (3 - 2 * prog)

        for bar, h in zip(bars_base1, fake_base * prog_e):
            bar.set_height(h)
        for bar, h in zip(bars_mlp1, fake_mlp * prog_e):
            bar.set_height(h)
        for bar, h in zip(bars_lgbm1, fake_lgbm * prog_e):
            bar.set_height(h)
        # for bar, h in zip(bars_base2, eff_base * prog_e + 60 * (1 - prog_e)):
        #     bar.set_height(max(0, h - 60))
        # for bar, h in zip(bars_mlp2, eff_mlp * prog_e + 60 * (1 - prog_e)):
        #     bar.set_height(max(0, h - 60))
        # for bar, h in zip(bars_lgbm2, eff_lgbm * prog_e + 60 * (1 - prog_e)):
        #     bar.set_height(max(0, h - 60))
        for bar, h in zip(bars_base2, (eff_base - 60) * prog_e):
            bar.set_height(h)
        for bar, h in zip(bars_mlp2, (eff_mlp - 60) * prog_e):
            bar.set_height(h)
        for bar, h in zip(bars_lgbm2, (eff_lgbm - 60) * prog_e):
            bar.set_height(h)

        return (list(bars_base1) + list(bars_mlp1) + list(bars_lgbm1) +
                list(bars_base2) + list(bars_mlp2) + list(bars_lgbm2))

    anim = animation.FuncAnimation(fig, update, frames=total_frames,
                                   blit=True)
    out = outdir / 'act4_results.mp4'
    save_anim(anim, out, fps=fps)
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINE all acts into one MP4
# ═══════════════════════════════════════════════════════════════════════════════

# def combine(clips, outdir):
#     """Use ffmpeg to concatenate clips."""
#     import subprocess
#     list_file = outdir / 'concat_list.txt'
#     with open(list_file, 'w') as f:
#         for c in clips:
#             f.write(f"file '{c.resolve()}'\n")

#     out = outdir / 'thesis_full_animation.mp4'
#     cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0',
#            '-i', str(list_file),
#            '-c', 'copy', str(out)]
#     result = subprocess.run(cmd, capture_output=True, text=True)
#     if result.returncode != 0:
#         print("  ffmpeg concat failed:", result.stderr[-500:])
#         print("  individual clips are still in", outdir)
#     else:
#         print(f"  combined → {out}")
#     return out

def combine(clips, outdir):
    """Concatenate gif clips using Pillow (no ffmpeg needed)."""
    from PIL import Image
    out = outdir / 'thesis_full_animation.gif'

    frames = []
    durations = []

    for clip in clips:
        # clips list may have .mp4 paths but files are actually .gif
        gif_path = Path(clip).with_suffix('.gif')
        if not gif_path.exists():
            print(f"  warning: could not find {gif_path}, skipping")
            continue
        img = Image.open(gif_path)
        try:
            while True:
                frames.append(img.copy().convert('RGBA'))
                durations.append(img.info.get('duration', 33))  # default ~30fps
                img.seek(img.tell() + 1)
        except EOFError:
            pass

    if not frames:
        print("  no frames collected, aborting combine")
        return out

    frames[0].save(
        str(out),
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )
    print(f"  combined → {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description='Thesis pipeline animation')
    ap.add_argument('--hits',      default=None, help='path to hits.root')
    ap.add_argument('--particles', default=None, help='path to particles.root')
    ap.add_argument('--seeds',     default=None, help='path to estimatedparams.root (unused for now)')
    ap.add_argument('--outdir',    default='./animation_out', help='output directory')
    ap.add_argument('--fps',       type=int, default=30, help='frames per second (default 30; use 24 for smaller files)')
    ap.add_argument('--acts',      default='1,2,3,4', help='comma-separated acts to render, e.g. 1,3')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    acts = [int(a.strip()) for a in args.acts.split(',')]

    # load data if provided
    hits_data = None
    particles = {}
    if args.hits and Path(args.hits).exists():
        print("Loading hits.root …")
        hits_data = try_load_hits(args.hits)
    if args.particles and Path(args.particles).exists():
        print("Loading particles.root …")
        particles = try_load_particles(args.particles)

    clips = []
    if 1 in acts:
        clips.append(make_act1(outdir, hits_data, particles, fps=args.fps))
    if 2 in acts:
        clips.append(make_act2(outdir, hits_data, particles, fps=args.fps))
    if 3 in acts:
        clips.append(make_act3(outdir, fps=args.fps))
    if 4 in acts:
        clips.append(make_act4(outdir, fps=args.fps))

    if len(clips) > 1:
        combine(clips, outdir)

    print("\nDone! Files written to:", outdir)


if __name__ == '__main__':
    main()
