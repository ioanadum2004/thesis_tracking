"""
eff_vs_multilicity.py

Reads tracksummary_ckf.root and run.log from each multiplicity directory
produced by run_multiplicity.py and generates plots of tracking performance
as a function of the number of particles per event, for the baseline, MLP,
and LightGBM configurations.

How to Run
----------
Run from the acts/ working directory after completing the multiplicity sweep.

1. Run with default multiplicity range:
   $ python eff_vs_multiplicity.py

Requirements
------------
- ROOT files must be present at:
  z_btr_files/multiplicity_sweep/mult_{m}/{label}/tracksummary_ckf.root
  for each multiplicity m and label in {baseline, mlp, tree}
- run.log files must be present in the same directories for timing extraction
  (generated automatically by the updated run_multiplicity.py)

Output
------
    efficiency_vs_multiplicity.png        : track efficiency vs particles per event
    fake_rate_vs_multiplicity.png         : fake rate vs particles per event
    both_vs_multiplicity.png              : efficiency and fake rate combined
    efficiency_vs_multiplicity_division.png : MLP/baseline efficiency ratio
    efficiency_vs_multiplicity_minus.png  : MLP - baseline efficiency difference
    timing_vs_multiplicity.png            : total CKF computation time vs particles per event
    seed_type_comparison_vs_multiplicity.png : fake, matched, truth matched, duplicate,
                                              and total seed counts vs particles per event

    All plots saved to z_btr_files/efficiency_plots/multiplicity_plots/

Module structure
----------------
    eff_vs_mult → parse_timing → [plot functions]
"""

import uproot
import numpy as np
import matplotlib.pyplot as plt
import re
import os

output_dir = "z_btr_files/efficiency_plots/multiplicity_plots/conf_B/seed_types"
os.makedirs(output_dir, exist_ok=True)

def delete_parse_timing(log_path):
    """Extract total CKF wall time in ms from run.log."""
    try:
        with open(log_path) as f:
            for line in f:
                if "TOTAL" in line:
                    match = re.search(r'\|\s+([\d.]+)\s+\|', line)
                    if match:
                        return float(match.group(1))
    except FileNotFoundError:
        pass
    return None

def parse_timing(log_path):
    print(f"  looking for log: {log_path}")
    try:
        with open(log_path) as f:
            for line in f:
                if "TOTAL" in line:
                    numbers = re.findall(r'[\d]+\.[\d]+', line)
                    if numbers:
                        return float(numbers[0])
    except FileNotFoundError:
        print(f"  log not found: {log_path}")
    return None

def eff_vs_mult():

#    multiplicities = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]
    multiplicities = list(range(1, 150))
    efficiencies_baseline = []
    efficiencies_mlp = []
    efficiencies_tree = []

    fake_baseline = []
    fake_mlp = []
    fake_tree = []

    fake_count_baseline = []
    total_seeds_baseline = []
    truth_matched_baseline = []
    duplicate_baseline = []
    matched_baseline = []

    fake_count_mlp = []
    total_seeds_mlp = []
    truth_matched_mlp = []
    duplicate_mlp = []
    matched_mlp = []

    fake_count_tree = []
    total_seeds_tree = []
    truth_matched_tree = []
    duplicate_tree = []
    matched_tree = []

    timing_baseline = []
    timing_mlp = []
    timing_tree = []

    for mult in multiplicities:

        print(f"\n{'='*55}")
        print(f"  Multiplicity = {mult}")
        print(f"{'='*55}")

        #for label, results_list, fake_list in [("baseline", efficiencies_baseline, fake_baseline), 
        #                            ("seedfilter", efficiencies_sf, fake_sf)]:
        for label, results_list, fake_list, fake_count, total_seeds, truth_matched, duplicate, matched in [
                ("baseline", efficiencies_baseline, fake_baseline, fake_count_baseline, total_seeds_baseline, truth_matched_baseline, duplicate_baseline, matched_baseline),
                ("mlp",      efficiencies_mlp,      fake_mlp, fake_count_mlp, total_seeds_mlp, truth_matched_mlp, duplicate_mlp, matched_mlp),
                ("tree",     efficiencies_tree,      fake_tree, fake_count_tree, total_seeds_tree, truth_matched_tree, duplicate_tree, matched_tree),
        ]:
            path = f"z_btr_files/multiplicity_sweep_conf_B/mult_{mult}/{label}/tracksummary_ckf.root"
            
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
            
            track_eff = (unu_count + doi_count)/total_count
            fake_rate = zero_count/total_count
            results_list.append(track_eff)
            fake_list.append(fake_rate)
            print("track efficiency:", track_eff)
            print("particle efficiency:", unu_count/total_count)
            print("track fake rate:", fake_rate)

            log_path = f"z_btr_files/multiplicity_sweep_conf_B/mult_{mult}/{label}/run.log"
            t = parse_timing(log_path)
            if label == "baseline":
                timing_baseline.append(t)
            elif label == "mlp":
                timing_mlp.append(t)
            elif label == "tree":
                timing_tree.append(t)
            print(f"timing: {t} ms" if t is not None else "timing: not found")
                
    particles_per_event = [m * 3 * 5 for m in multiplicities]

    # --- TRACKING EFFICIENCY---

    plt.figure()
    plt.plot(particles_per_event, efficiencies_baseline, "-", color="blue",  label="Baseline")
    plt.plot(particles_per_event, efficiencies_mlp,       "-", color="orange", label="MLP")
    plt.plot(particles_per_event, efficiencies_tree,       "-", color="purple", label="LightGBM")
    # plt.plot(particles_per_event, fake_baseline, "o-", color="green",  label="Baseline Fake Rate")
    # plt.plot(particles_per_event, fake_sf,       "o-", color="red", label="Seed Filter Fake Rate")
    plt.title("Track Efficiency vs Multiplicity")
    plt.xlabel("Particles per event")
    plt.ylabel("Track Efficiency")
    plt.ylim(0, 1.05)
    plt.grid()
    plt.legend()
    plt.savefig("z_btr_files/efficiency_plots/multiplicity_plots/conf_B/efficiency_vs_multiplicity_B.png", dpi=150)
    plt.close()

   # # --- TYPES OF SEEDS ---
    
   # plt.figure()
   # plt.plot(particles_per_event, fake, "o-", color="blue",  label="Fake Seeds")
    #plt.plot(particles_per_event, matched,       "o-", color="green", label="Matched")
    #plt.plot(particles_per_event, truth_matched,       "o-", color="red", label="Truth Matched")
    #plt.plot(particles_per_event, duplicate,       "o-", color="orange", label="Duplicate")
    #plt.plot(particles_per_event, total_seeds,       "o-", color="purple", label="Total Seeds")
    # plt.plot(particles_per_event, fake_baseline, "o-", color="green",  label="Baseline Fake Rate")                                                                                               
    # plt.plot(particles_per_event, fake_sf,       "o-", color="red", label="Seed Filter Fake Rate")                                                                                               
    #plt.title("Types of seeds vs multiplicity")
    #plt.xlabel("Particles per event")
    #plt.ylabel("Seed Count")
    #plt.ylim(0, 1.05)
    #plt.grid()
    #plt.legend()
    #plt.savefig("z_btr_files/types_of_seeds_count.png", dpi=150)
    #plt.close()

    # --- TYPES OF SEEDS vs MULTIPLICITY ---

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)

    seed_types = [
        ("Fake Seeds", fake_count_baseline, fake_count_mlp, fake_count_tree),
        ("Matched", matched_baseline, matched_mlp, matched_tree),
        ("Truth Matched", truth_matched_baseline, truth_matched_mlp, truth_matched_tree),
        ("Duplicate", duplicate_baseline, duplicate_mlp, duplicate_tree),
        ("Total Seeds", total_seeds_baseline, total_seeds_mlp, total_seeds_tree),
    ]
    
    colors = {
        "Baseline": "blue",
        "MLP": "orange",
        "LightGBM": "purple"
    }
    
    for ax, (title, baseline, mlp, tree) in zip(axes.flat, seed_types):
        ax.plot(particles_per_event, baseline, "-", color=colors["Baseline"], label="Baseline")
        ax.plot(particles_per_event, mlp, "-", color=colors["MLP"], label="MLP")
        ax.plot(particles_per_event, tree, "-", color=colors["LightGBM"], label="LightGBM")

        ax.set_title(title)
        ax.set_xlabel("Particles per event")
        ax.set_ylabel("Seed Count")
        ax.grid(True)
        ax.legend()

        # Remove the unused sixth subplot
    fig.delaxes(axes[1, 2])

    fig.suptitle("Seed Type Comparison vs Multiplicity", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("z_btr_files/efficiency_plots/multiplicity_plots/conf_B/seed_type_comparison_vs_multiplicity_B.png", dpi=150)
    plt.close()

    for title, baseline, mlp, tree in seed_types:
        plt.figure()

        plt.plot(particles_per_event, baseline, "-", color=colors["Baseline"], label="Baseline")
        plt.plot(particles_per_event, mlp, "-", color=colors["MLP"], label="MLP")
        plt.plot(particles_per_event, tree, "-", color=colors["LightGBM"], label="LightGBM")

        plt.title(title)
        plt.xlabel("Particles per event")
        plt.ylabel("Seed Count")
        plt.grid(True)
        plt.legend()

        # safe filename (optional but recommended)
        filename = title.lower().replace(" ", "_") + ".png"
        filepath = os.path.join(output_dir, filename)

        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

    # --- FAKE RATE ---

    plt.figure()
   # plt.plot(particles_per_event, efficiencies_baseline, "o-", color="blue",  label="Baseline")
   # plt.plot(particles_per_event, efficiencies_sf,       "o-", color="orange", label="Seed Filter")
    plt.plot(particles_per_event, fake_baseline, "-", color=colors["Baseline"],  label="Baseline")
    plt.plot(particles_per_event, fake_mlp,       "-", color=colors["MLP"], label="MLP")
    plt.plot(particles_per_event, fake_tree,       "-", color=colors["LightGBM"], label="LightGBM")
    plt.title("Fake Rate vs Multiplicity")
    plt.xlabel("Particles per event")
    plt.ylabel("Track Efficiency")
    plt.ylim(0, 1.05)
    plt.grid()
    plt.legend()
    plt.savefig("z_btr_files/efficiency_plots/multiplicity_plots/conf_B/fake_rate_vs_multiplicity_B.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(particles_per_event, efficiencies_baseline, "-", color="blue",  label="Baseline")                                                                                         
    #plt.plot(particles_per_event, efficiencies_sf,       "o-", color="orange", label="Seed Filter")                                                                                     
    plt.plot(particles_per_event, efficiencies_mlp,       "-", color="orange", label="MLP")
    plt.plot(particles_per_event, efficiencies_tree,       "-", color="purple", label="LightGBM")
    plt.plot(particles_per_event, fake_baseline, "-", color="#81C784",  label="Baseline")
    plt.plot(particles_per_event, fake_mlp,       "-", color="red", label="MLP")
    plt.plot(particles_per_event, fake_tree,       "-", color="deepskyblue", label="LightGBM")
    #plt.plot(particles_per_event, fake_sf,       "o-", color="red", label="Seed Filter Fake Rate")
    plt.title("Fake Rate vs Multiplicity")
    plt.xlabel("Particles per event")
    plt.ylabel("Track Efficiency")
    plt.ylim(0, 1.05)
    plt.grid()
    plt.legend()
    plt.savefig("z_btr_files/efficiency_plots/multiplicity_plots/conf_B/both_vs_multiplicity_B.png", dpi=150)
    plt.close()
    
    efficiencies_sf_baseline = []
    efficiencies_sf_baseline_minus = []
    counter = 1

    # print("\n Baseline \n")
    # for i, mult in enumerate(multiplicities):
    #     efficiencies_sf_baseline.append(efficiencies_mlp[i] / efficiencies_baseline[i])
    #     efficiencies_sf_baseline_minus.append(efficiencies_mlp[i] - efficiencies_baseline[i])
    #     # print(counter," - ", efficiencies_sf_baseline[mult-1])
    #     counter += 1


    # plt.figure()
    # plt.plot(particles_per_event, efficiencies_sf_baseline, "o-", color="red",  label="Division")
    # plt.title("Track Efficiency Seed Filter/Baseline vs Multiplicity")
    # plt.xlabel("Particles per event")
    # plt.ylabel("Track Efficiency")
    # plt.ylim(0, 2.05)
    # plt.grid()
    # plt.legend()
    # plt.savefig("z_btr_files/efficiency_plots/multiplicity_plots/efficiency_vs_multiplicity_division.png", dpi=150)
    # plt.close()

    # plt.figure()
    # plt.plot(particles_per_event, efficiencies_sf_baseline_minus, "o-", color="green",  label="Minus")
    # plt.title("Track Efficiency Seed Filter - Baseline vs Multiplicity")
    # plt.xlabel("Particles per event")
    # plt.ylabel("Track Efficiency")
    # # plt.ylim(0, 0.0002)
    # plt.grid()
    # plt.legend()
    # plt.savefig("z_btr_files/efficiency_plots/multiplicity_plots/efficiency_vs_multiplicity_minus.png", dpi=150)
    # plt.close()

    # --- TIME ---

    plt.figure()
    plt.plot(particles_per_event, timing_baseline, "-", color=colors["Baseline"],   label="Baseline")
    plt.plot(particles_per_event, timing_mlp,      "-", color=colors["MLP"], label="MLP")
    plt.plot(particles_per_event, timing_tree,     "-", color=colors["LightGBM"], label="LightGBM")
    plt.title("Computation Time vs Multiplicity")
    plt.xlabel("Particles per event")
    plt.ylabel("Total CKF time (ms)")
    plt.grid()
    plt.legend()
    plt.savefig("z_btr_files/efficiency_plots/multiplicity_plots/conf_B/timing_vs_multiplicity_B.png", dpi=150)
    plt.close()

    print("\n" + "="*55)
    print(" ALL PLOTS SUCCESSFULLY GENERATED AND SAVED!")
    print("="*55)
    print("Main Performance Plots:")
    print("  - Track Efficiency : z_btr_files/efficiency_plots/multiplicity_plots/conf_B/efficiency_vs_multiplicity_B.png")
    print("  - Fake Rate        : z_btr_files/efficiency_plots/multiplicity_plots/conf_B/fake_rate_vs_multiplicity_B.png")
    print("  - Combined (Both)  : z_btr_files/efficiency_plots/multiplicity_plots/conf_B/both_vs_multiplicity_B.png")
    print("  - Computation Time : z_btr_files/efficiency_plots/multiplicity_plots/conf_B/timing_vs_multiplicity_B.png")
    print("\nSeed Comparison Plots:")
    print("  - Grid Overview    : z_btr_files/efficiency_plots/multiplicity_plots/seed_type_comparison_vs_multiplicity_B.png")
    print(f"  - Individual Types : Saved in '{output_dir}/'")
    print("                       (fake_seeds.png, matched.png, truth_matched.png, duplicate.png, total_seeds.png)")
    print("="*55 + "\n")

if __name__ == "__main__":

    eff_vs_mult()
