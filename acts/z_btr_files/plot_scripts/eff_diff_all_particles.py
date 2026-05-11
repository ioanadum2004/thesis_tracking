"""
eff_diff_all_particles.py

Computes and plots the **difference** (new - baseline) of tracking metrics
(efficiency, fake rate, duplicate rate, true matched efficiency) as a function
of transverse momentum (pT), aggregated over all particle types.

Files expected:
  new config  : tracksummary_ckf.root      + particles.root
  baseline    : tracksummary_ckf_baseline.root + particles_baseline.root

Output plots are saved to z_btr_files/efficiency_plots/ (or --out-dir).
A positive value means the new configuration is *higher* than baseline.

Usage:
  python eff_diff_all_particles.py
  python eff_diff_all_particles.py \\
      --tracks       tracksummary_ckf.root \\
      --particles    particles.root \\
      --tracks-base  tracksummary_ckf_baseline.root \\
      --particles-base particles_baseline.root \\
      --out-dir      my_plots/diff
"""

import os
import argparse

import uproot
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Internal helper: build per-bin histograms for one run
# ---------------------------------------------------------------------------

def _compute_histograms(tracks_file, particles_file,
                        track_tree="tracksummary",
                        particle_tree="particles",
                        pt_bins=None):
    """Return (hist_all, hist_good, hist_fake, hist_duplicate, hist_matched)
    as numpy arrays of length len(pt_bins)-1.

    Classification codes (ACTS convention):
      0 = fake
      1 = matched (best match, "good")
      2 = duplicate
    """
    if pt_bins is None:
        pt_bins = np.linspace(0.1, 0.5, 50)

    n_bins = len(pt_bins) - 1

    hist_all      = np.zeros(n_bins)
    hist_good     = np.zeros(n_bins)
    hist_fake     = np.zeros(n_bins)
    hist_duplicate = np.zeros(n_bins)
    hist_matched  = np.zeros(n_bins)

    tracks = uproot.open(tracks_file)[track_tree].arrays(
        ["majorityParticleId", "t_pT", "trackClassification",
         "eQOP_fit", "eTHETA_fit"],
        library="ak",
    )

    n_events = len(tracks["majorityParticleId"])
    print(f"  [{tracks_file}] {n_events} events")

    for event in range(n_events):
        track_pts   = tracks["t_pT"][event]
        track_class = tracks["trackClassification"][event]
        track_qop   = tracks["eQOP_fit"][event]
        track_theta = tracks["eTHETA_fit"][event]

        for pt, classification, qop, theta in zip(
            track_pts, track_class, track_qop, track_theta
        ):
            # For fakes there is no truth pT; reconstruct from fitted params
            if classification == 0:
                pt = np.sin(theta) / abs(qop)

            if pt < pt_bins[0]:
                continue

            bin_index = np.digitize(pt, pt_bins) - 1
            if not (0 <= bin_index < n_bins):
                continue

            hist_all[bin_index] += 1

            if classification in (1, 2):
                hist_matched[bin_index] += 1
            if classification == 1:
                hist_good[bin_index] += 1
            if classification == 2:
                hist_duplicate[bin_index] += 1
            if classification == 0:
                hist_fake[bin_index] += 1

    return hist_all, hist_good, hist_fake, hist_duplicate, hist_matched


def _safe_rate(numerator, denominator):
    """Element-wise division; return 0 where denominator == 0."""
    return np.where(denominator > 0, numerator / denominator, 0.0)


def _binomial_error(rate, denominator):
    """Standard binomial uncertainty sqrt(p(1-p)/N); 0 where N == 0."""
    return np.where(
        denominator > 0,
        np.sqrt(rate * (1 - rate) / np.maximum(denominator, 1)),
        0.0,
    )


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def track_metrics_diff_all_particles(
    tracks_file="tracksummary_ckf.root",
    particles_file="particles.root",
    tracks_file_base="tracksummary_ckf_baseline.root",
    particles_file_base="particles_baseline.root",
    track_tree="tracksummary",
    particle_tree="particles",
    pt_bins=None,
    output_efficiency_diff="efficiency_diff_vs_pt.png",
    output_fake_diff="fake_diff_vs_pt.png",
    output_duplicate_diff="duplicate_diff_vs_pt.png",
    output_matched_diff="matched_diff_vs_pt.png",
    out_dir=None,
):
    """
    Plot (new - baseline) for each tracking metric vs pT.

    Parameters
    ----------
    tracks_file, particles_file      : new-configuration ROOT files
    tracks_file_base, particles_file_base : baseline ROOT files
    pt_bins   : numpy array of bin edges (default: np.linspace(0.1, 0.5, 50))
    output_*  : filenames for the four difference plots
    out_dir   : directory to write plots (default: z_btr_files/efficiency_plots)
    """
    if pt_bins is None:
        pt_bins = np.linspace(0.1, 0.5, 50)

    output_dir = out_dir if out_dir else os.path.join("z_btr_files", "efficiency_plots")
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Build histograms for both runs
    # ------------------------------------------------------------------
    print("Reading new configuration...")
    all_new, good_new, fake_new, dup_new, matched_new = _compute_histograms(
        tracks_file, particles_file, track_tree, particle_tree, pt_bins
    )

    print("Reading baseline configuration...")
    all_base, good_base, fake_base, dup_base, matched_base = _compute_histograms(
        tracks_file_base, particles_file_base, track_tree, particle_tree, pt_bins
    )

    # ------------------------------------------------------------------
    # Compute per-bin rates
    # ------------------------------------------------------------------
    efficiency_new   = _safe_rate(matched_new, all_new)
    efficiency_base  = _safe_rate(matched_base, all_base)

    fake_new_rate    = _safe_rate(fake_new, all_new)
    fake_base_rate   = _safe_rate(fake_base, all_base)

    dup_new_rate     = _safe_rate(dup_new, all_new)
    dup_base_rate    = _safe_rate(dup_base, all_base)

    matched_new_rate  = _safe_rate(good_new, all_new)
    matched_base_rate = _safe_rate(good_base, all_base)

    # ------------------------------------------------------------------
    # Differences: new - baseline
    # ------------------------------------------------------------------
    diff_efficiency = efficiency_new  - efficiency_base
    diff_fake       = fake_new_rate   - fake_base_rate
    diff_duplicate  = dup_new_rate    - dup_base_rate
    diff_matched    = matched_new_rate - matched_base_rate

    # ------------------------------------------------------------------
    # Propagated errors (add in quadrature)
    # ------------------------------------------------------------------
    err_efficiency = np.sqrt(
        _binomial_error(efficiency_new,  all_new)  ** 2 +
        _binomial_error(efficiency_base, all_base) ** 2
    )
    err_fake = np.sqrt(
        _binomial_error(fake_new_rate,  all_new)  ** 2 +
        _binomial_error(fake_base_rate, all_base) ** 2
    )
    err_duplicate = np.sqrt(
        _binomial_error(dup_new_rate,  all_new)  ** 2 +
        _binomial_error(dup_base_rate, all_base) ** 2
    )
    err_matched = np.sqrt(
        _binomial_error(matched_new_rate,  all_new)  ** 2 +
        _binomial_error(matched_base_rate, all_base) ** 2
    )

    bin_centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    def _global(hist_new, hist_base):
        rate_n = hist_new.sum() / max(all_new.sum(), 1)
        rate_b = hist_base.sum() / max(all_base.sum(), 1)
        return rate_n, rate_b, rate_n - rate_b

    rn, rb, rd = _global(matched_new, matched_base)
    print(f"\nGlobal efficiency   — new: {rn:.4f}  baseline: {rb:.4f}  diff: {rd:+.4f}")
    rn, rb, rd = _global(fake_new, fake_base)
    print(f"Global fake rate    — new: {rn:.4f}  baseline: {rb:.4f}  diff: {rd:+.4f}")
    rn, rb, rd = _global(dup_new, dup_base)
    print(f"Global dup rate     — new: {rn:.4f}  baseline: {rb:.4f}  diff: {rd:+.4f}")
    rn, rb, rd = _global(good_new, good_base)
    print(f"Global matched eff  — new: {rn:.4f}  baseline: {rb:.4f}  diff: {rd:+.4f}")

    # ------------------------------------------------------------------
    # Helper to draw one difference plot
    # ------------------------------------------------------------------
    def _plot_diff(diff, err, title, ylabel, filename, color):
        plt.figure()
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
        #plt.errorbar(bin_centers, diff, yerr=err, fmt='o', color=color)
        plt.plot(bin_centers, diff, color=color)
        plt.scatter(bin_centers, diff, color=color, s=20)
        plt.title(title)
        plt.xlabel("pT [GeV]")
        plt.ylabel(ylabel)
        plt.grid()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        print(f"Saved: {os.path.join(output_dir, filename)}")

    # ------------------------------------------------------------------
    # Save plots
    # ------------------------------------------------------------------
    _plot_diff(
        diff_efficiency, err_efficiency,
        "Track Efficiency Difference vs pT (new − baseline)",
        "Δ Efficiency",
        output_efficiency_diff,
        color='blue',
    )
    _plot_diff(
        diff_fake, err_fake,
        "Fake Rate Difference vs pT (new − baseline)",
        "Δ Fake Rate",
        output_fake_diff,
        color='green',
    )
    _plot_diff(
        diff_duplicate, err_duplicate,
        "Duplicate Rate Difference vs pT (new − baseline)",
        "Δ Duplicate Rate",
        output_duplicate_diff,
        color='orange',
    )
    _plot_diff(
        diff_matched, err_matched,
        "True Matched Efficiency Difference vs pT (new − baseline)",
        "Δ Matched Efficiency",
        output_matched_diff,
        color='red',
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Plot (new - baseline) tracking metric differences vs pT."
    )
    p.add_argument("--tracks",          default="tracksummary_ckf.root",
                   help="New-config tracks ROOT file (default: tracksummary_ckf.root)")
    p.add_argument("--particles",       default="particles.root",
                   help="New-config particles ROOT file (default: particles.root)")
    p.add_argument("--tracks-base",     default="tracksummary_ckf_baseline.root",
                   help="Baseline tracks ROOT file (default: tracksummary_ckf_baseline.root)")
    p.add_argument("--particles-base",  default="particles_baseline.root",
                   help="Baseline particles ROOT file (default: particles_baseline.root)")
    p.add_argument("--out-dir",         default=None,
                   help="Output directory (default: z_btr_files/efficiency_plots)")
    args = p.parse_args()

    track_metrics_diff_all_particles(
        tracks_file=args.tracks,
        particles_file=args.particles,
        tracks_file_base=args.tracks_base,
        particles_file_base=args.particles_base,
        out_dir=args.out_dir,
    )
