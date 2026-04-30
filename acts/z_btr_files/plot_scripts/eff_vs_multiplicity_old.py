# eff_vs_multiplicity_less_features.py
#
# Plots tracking efficiency, fake rate, and seed type counts vs. multiplicity
# for two conditions: baseline (no filter) and seed filter. Reads
# tracksummary_ckf.root files from the multiplicity_sweep_less_features
# directory and writes plots to z_btr_files/.
#
# Run from the acts/ directory:
#   python z_btr_files/plot_scripts/eff_vs_multiplicity_less_features.py
#
# The script must be run from acts/ because all input/output paths are
# relative to that directory.

import uproot
import numpy as np
import matplotlib.pyplot as plt

def eff_vs_mult():

    multiplicities = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
    efficiencies_baseline = []
    efficiencies_sf = []

    fake_baseline = []
    fake_sf = []

    fake_count_baseline = []
    total_seeds_baseline = []
    truth_matched_baseline = []
    duplicate_baseline = []
    matched_baseline = []

    fake_count_sf = []
    total_seeds_sf = []
    truth_matched_sf = []
    duplicate_sf = []
    matched_sf = []

    for mult in multiplicities:

        print(f"\n{'='*55}")
        print(f"  Multiplicity = {mult}")
        print(f"{'='*55}")

        for label, results_list, fake_list, fake_count, total_seeds, truth_matched, duplicate, matched in [
                ("baseline",    efficiencies_baseline, fake_baseline, fake_count_baseline, total_seeds_baseline, truth_matched_baseline, duplicate_baseline, matched_baseline),
                ("seedfilter",  efficiencies_sf,       fake_sf,       fake_count_sf,       total_seeds_sf,       truth_matched_sf,       duplicate_sf,       matched_sf),
        ]:
            path = f"z_btr_files/multiplicity_sweep_less_features/mult_{mult}/{label}/tracksummary_ckf.root"
            
            f_tracks = uproot.open(path)
            t_tracks = f_tracks["tracksummary"]
            tracks = t_tracks.arrays(["majorityParticleId", "t_pT", "trackClassification", "eQOP_fit", "eTHETA_fit"], library="ak")
            n_events = len(tracks["majorityParticleId"])

            zero_count = 0
            total_count = 0
            unu_count = 0
            doi_count = 0
            trei_count = 0

            for event in range(n_events):
                for ids in tracks["trackClassification"][event]:
                    total_count += 1
                    if ids == 0:
                        zero_count += 1
                    elif ids == 1:
                        unu_count += 1
                    elif ids == 2:
                        doi_count += 1
                    elif ids == 3:
                        trei_count += 1

            print(f"\n{label} \n")
            print("fake:", zero_count)
            print("matched:", unu_count + doi_count)
            print("true matched:", unu_count)
            print("duplicate:", doi_count)
            print("total seeds:", total_count)

            fake_count.append(zero_count)
            total_seeds.append(total_count)
            truth_matched.append(unu_count)
            duplicate.append(doi_count)
            matched.append(unu_count + doi_count)

            track_eff = (unu_count + doi_count) / total_count
            fake_rate = zero_count / total_count
            results_list.append(track_eff)
            fake_list.append(fake_rate)
            print("track efficiency:", track_eff)
            print("particle efficiency:", unu_count / total_count)
            print("track fake rate:", fake_rate)

    particles_per_event = [m * 5 * 5 for m in multiplicities]

    colors = {
        "Baseline":    "blue",
        "Seed Filter": "orange",
    }

    # --- TRACKING EFFICIENCY ---

    plt.figure()
    plt.plot(particles_per_event, efficiencies_baseline, "o-", color=colors["Baseline"],    label="Baseline")
    plt.plot(particles_per_event, efficiencies_sf,       "o-", color=colors["Seed Filter"], label="Seed Filter")
    plt.title("Track Efficiency vs Multiplicity")
    plt.xlabel("Particles per event")
    plt.ylabel("Track Efficiency")
    plt.ylim(0, 1.05)
    plt.grid()
    plt.legend()
    plt.savefig("z_btr_files/efficiency_vs_multiplicity.png", dpi=150)
    plt.close()

    # --- SEED TYPE COMPARISON (2x3 grid, last panel unused) ---

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)

    seed_types = [
        ("Fake Seeds",    fake_count_baseline,    fake_count_sf),
        ("Matched",       matched_baseline,        matched_sf),
        ("Truth Matched", truth_matched_baseline,  truth_matched_sf),
        ("Duplicate",     duplicate_baseline,      duplicate_sf),
        ("Total Seeds",   total_seeds_baseline,    total_seeds_sf),
    ]

    for ax, (title, baseline, sf) in zip(axes.flat, seed_types):
        ax.plot(particles_per_event, baseline, "o-", color=colors["Baseline"],    label="Baseline")
        ax.plot(particles_per_event, sf,       "o-", color=colors["Seed Filter"], label="Seed Filter")
        ax.set_title(title)
        ax.set_xlabel("Particles per event")
        ax.set_ylabel("Seed Count")
        ax.grid(True)
        ax.legend()

    fig.delaxes(axes[1, 2])
    fig.suptitle("Seed Type Comparison vs Multiplicity", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("z_btr_files/seed_type_comparison_vs_multiplicity.png", dpi=150)
    plt.close()

    # --- FAKE RATE ---

    plt.figure()
    plt.plot(particles_per_event, fake_baseline, "o-", color=colors["Baseline"],    label="Baseline")
    plt.plot(particles_per_event, fake_sf,       "o-", color=colors["Seed Filter"], label="Seed Filter")
    plt.title("Fake Rate vs Multiplicity")
    plt.xlabel("Particles per event")
    plt.ylabel("Fake Rate")
    plt.ylim(0, 1.05)
    plt.grid()
    plt.legend()
    plt.savefig("z_btr_files/fake_rate_vs_multiplicity.png", dpi=150)
    plt.close()

    # --- EFFICIENCY AND FAKE RATE COMBINED ---

    plt.figure()
    plt.plot(particles_per_event, efficiencies_baseline, "o-", color=colors["Baseline"],    label="Baseline Efficiency")
    plt.plot(particles_per_event, efficiencies_sf,       "o-", color=colors["Seed Filter"], label="Seed Filter Efficiency")
    plt.plot(particles_per_event, fake_baseline, "s--", color=colors["Baseline"],    label="Baseline Fake Rate")
    plt.plot(particles_per_event, fake_sf,       "s--", color=colors["Seed Filter"], label="Seed Filter Fake Rate")
    plt.title("Efficiency and Fake Rate vs Multiplicity")
    plt.xlabel("Particles per event")
    plt.ylabel("Rate")
    plt.ylim(0, 1.05)
    plt.grid()
    plt.legend()
    plt.savefig("z_btr_files/both_vs_multiplicity.png", dpi=150)
    plt.close()

    # --- EFFICIENCY RATIO (SF / Baseline) ---

    efficiencies_sf_baseline = [efficiencies_sf[i] / efficiencies_baseline[i] for i in range(len(multiplicities))]
    efficiencies_sf_baseline_minus = [efficiencies_sf[i] - efficiencies_baseline[i] for i in range(len(multiplicities))]

    plt.figure()
    plt.plot(particles_per_event, efficiencies_sf_baseline, "o-", color="red", label="SF / Baseline")
    plt.title("Track Efficiency Seed Filter / Baseline vs Multiplicity")
    plt.xlabel("Particles per event")
    plt.ylabel("Efficiency Ratio")
    plt.ylim(0, 2.05)
    plt.grid()
    plt.legend()
    plt.savefig("z_btr_files/efficiency_vs_multiplicity_division.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(particles_per_event, efficiencies_sf_baseline_minus, "o-", color="green", label="SF - Baseline")
    plt.title("Track Efficiency Seed Filter - Baseline vs Multiplicity")
    plt.xlabel("Particles per event")
    plt.ylabel("Efficiency Difference")
    plt.grid()
    plt.legend()
    plt.savefig("z_btr_files/efficiency_vs_multiplicity_minus.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    eff_vs_mult()
