"""
eff_plot_multiple_species.py

Computes and plots tracking metrics (efficiency, fake rate, duplicate rate,
true matched efficiency) as a function of transverse momentum (pT).

Two modes are available:
  - all_particles : aggregates all particle types into a single histogram
  - each_particle : breaks down metrics per particle type

Output plots are saved to z_btr_files/efficiency_plots/.

Usage:
  python eff_plot_multiple_species.py --mode all_particles
  python eff_plot_multiple_species.py --mode each_particle
  python eff_plot_multiple_species.py --mode all_particles --tracks tracksummary_ckf.root --particles particles.root
  python eff_plot_multiple_species.py --mode all_particles --out-dir my_plots/run1

Arguments:
  --mode        Which mode to run: 'all_particles' or 'each_particle' (required)
  --tracks      Path to the tracks ROOT file (default: tracksummary_ckf.root)
  --particles   Path to the particles ROOT file (default: particles.root)
  --out-dir     Directory to save output plots (default: z_btr_files/efficiency_plots)
"""

import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def track_metrics_all_particles(
    tracks_file,
    particles_file,
    track_tree="tracksummary",
    particle_tree="particles",
    pt_bins=np.linspace(0.1, 0.5, 50),
    output_purity="track_efficiency_vs_pt_all_part.png",
    output_fake_track="fake_track_vs_pt_all_part.png",
    output_duplicate_track="duplicate_ratio_vs_pt_all_part.png",
    output_matched_track="matched_efficiency_vs_pt_all_part.png",
    out_dir=None,
):
    
    # ---------- Create output directory ----------


    output_dir = out_dir if out_dir is not None else os.path.join("z_btr_files", "efficiency_plots")
    os.makedirs(output_dir, exist_ok=True)

    # ---------- READ PARTICLES ----------

    f_particles = uproot.open(particles_file)
    t_particles = f_particles[particle_tree]
    particles = t_particles.arrays(["particle_id", "pt", "particle_type"], library="ak")
    
    #print the particle types and their counts
    unique_types, counts = np.unique(particles["particle_type"], return_counts=True)
    print("Particle types and counts:")
    for t, c in zip(unique_types, counts):
        print(f"Type: {t}, Count: {c}")


    # ---------- READ TRACKS ----------

    f_tracks = uproot.open(tracks_file)
    t_tracks = f_tracks[track_tree]

    #also load track pt to plot against
    #tracks = t_tracks.arrays(["majorityParticleId", "t_pT"], library="ak" )
    tracks = t_tracks.arrays(["majorityParticleId", "t_pT", "trackClassification", "eQOP_fit", "eTHETA_fit"], library="ak")

    n_events = len(tracks["majorityParticleId"])
    # t_tracks = f_tracks[track_tree]


    # ---------- Histogram ----------
    
    #makes all histograms zero
    hist_all = np.zeros(len(pt_bins) - 1)
    hist_good = np.zeros(len(pt_bins) - 1)
    hist_fake = np.zeros(len(pt_bins) - 1)
    hist_duplicate = np.zeros(len(pt_bins) - 1)
    hist_matched = np.zeros(len(pt_bins) - 1)

    print("Number of events:", n_events)

     # ---------- LOOP THROUGH EVENTS ----------

    countfake = 0
    countgood = 0
    counttotal = 0
    countduplicate = 0
    countmatched = 0

    for event in range(n_events):

        track_truth = tracks["majorityParticleId"][event]
        track_pts = tracks["t_pT"][event]
        track_class = tracks["trackClassification"][event]
        track_qop = tracks["eQOP_fit"][event]
        track_theta = tracks["eTHETA_fit"][event]

        for ids, pt, classification, qop_fit, theta_fit in zip(track_truth, track_pts, track_class, track_qop, track_theta):
            # if len(ids) < 3:
            #     continue  # skip broken entries

            counttotal += 1

            if classification == 0:
                pt = np.sin(theta_fit) / abs(qop_fit)

            if pt < 0.1:
                continue

            bin_index = np.digitize(pt, pt_bins) - 1 # theres a problem here for some reason
            if not (0 <= bin_index < len(hist_all)):
                countfake += 1
                continue

            hist_all[bin_index] += 1
            # counttotal += 1
            if classification == 1 or classification == 2:   # truth-matched track
                hist_matched[bin_index] += 1
                countmatched += 1
            if classification == 1:   # truth-matched track
                hist_good[bin_index] += 1
                countgood += 1
            if classification == 2: # duplicate track
                hist_duplicate[bin_index] += 1
                countduplicate += 1
            if classification == 0: # fake track
                hist_fake[bin_index] += 1
                countfake += 1

    # ---------- Compute tracking efficiency ----------
    efficiency = np.zeros_like(hist_all)

    # print("Track efficiency:", np.sum(hist_good) / np.sum(hist_all))

    for i in range(len(hist_all)):
        if hist_all[i] > 0:
            efficiency[i] = hist_matched[i] / hist_all[i]

    bin_centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])

    # ---------- Errors ----------

    errors_purity = np.zeros_like(efficiency)

    for i in range(len(hist_all)):
        if hist_all[i] > 0:
            p = efficiency[i]
            N = hist_all[i]
            errors_purity[i] = np.sqrt(p * (1 - p) / N)

    # ---------- Compute fake track ----------

    fakeTrack = np.zeros_like(hist_all)

    for i in range(len(hist_all)):
        if hist_all[i] > 0:
            fakeTrack[i] = hist_fake[i] / hist_all[i]

    # ---------- Errors ----------

    errors_fake = np.zeros_like(fakeTrack)

    for i in range(len(hist_all)):
        if hist_all[i] > 0:
            p = fakeTrack[i]
            N = hist_all[i]
            errors_fake[i] = np.sqrt(p * (1 - p) / N)

    # ---------- Compute duplicate rate ----------

    duplicateRate = np.zeros_like(hist_all)

    for i in range(len(hist_all)): 
        if hist_all[i] > 0:
            duplicateRate[i] = hist_duplicate[i] / hist_all[i]

    # ---------- Errors ----------

    errors_duplicate = np.zeros_like(duplicateRate)

    for i in range(len(hist_all)):
        if hist_all[i] > 0:
            p = duplicateRate[i]
            N = hist_all[i]
            errors_duplicate[i] = np.sqrt(p * (1 - p) / N)

    # ---------- Compute true matched efficiency ----------

    matchedEfficiency = np.zeros_like(hist_all)
    for i in range(len(hist_all)):
        if hist_all[i] > 0:
            matchedEfficiency[i] = hist_good[i] / hist_all[i]

    # ---------- Errors ----------

    errors_matched = np.zeros_like(matchedEfficiency)
    for i in range(len(hist_all)):
        if hist_all[i] > 0:
            p = matchedEfficiency[i]
            N = hist_all[i]
            errors_matched[i] = np.sqrt(p * (1 - p) / N)

    # ----- DEBUG ----

    #see how many tracks are in each bin and how many are good
    # for i in range(len(hist_all)):
    #     print(f"Bin {i}: total={hist_all[i]}, good={hist_good[i]}")

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

    #for event in range(5):
    #     for ids in tracks["majorityParticleId"][event]:
    #         print(ids)

    print("\n")
    
    print("For loop - TRUTH - need to match this")
    print("Total tracks:", total_count)
    print("fake:", zero_count) 
    print("matched:", unu_count + doi_count)
    print("true matched:", unu_count)
    print("duplicate:", doi_count)
    print("douplicate - classification:", doi_count/total_count)
    print("truth-matched - classification:", unu_count/total_count)
    print("matced - classification:", (unu_count + doi_count)/total_count)
    print("fake - classification:", zero_count/total_count)

    print("\nHistogram counters:")
    print("Total tracks:", counttotal)
    print("Fake tracks:", countfake)
    print("Matched tracks:", countgood)
    print("Duplicate tracks:", countduplicate)
    print("Track Efficiency:", countmatched/counttotal)
    print("Fake Rate:", countfake/counttotal)
    print("Duplicate Rate:", countduplicate/counttotal)

    print("\nHistograms:")
    print("Total tracks in histogram:", np.sum(hist_all))
    print("Fake tracks:", np.sum(hist_fake))
    print("Matched tracks:", np.sum(hist_good))
    print("Duplicate tracks:", np.sum(hist_duplicate))
    print("Track Efficiency - histogram:", np.sum(hist_matched) / np.sum(hist_all))
    print("Fake rate - histogram:", np.sum(hist_fake) / np.sum(hist_all))
    print("Duplicate Rate - histogram:", np.sum(hist_duplicate) / np.sum(hist_all))

    print("\n")

    # ---------- Plot efficiency ----------
    plt.figure()
    # plt.plot(bin_centers, efficiency, "o-", color='blue')
    plt.errorbar(bin_centers, efficiency, yerr=errors_purity, fmt='o', color='blue')
    #if i want to make the error bars blue
    plt.title("Track Efficiency vs pT")
    plt.xlabel("pT [GeV]")
    plt.ylabel("Track Efficiency")
    plt.ylim(0, 1.05)
    plt.grid()
    # plt.legend()
    # plt.savefig(output_purity)
    plt.savefig(os.path.join(output_dir, output_purity))
    #plt.show()

    # print(f"Saved: {output_purity}")
    plt.close()

    # ---------- Plot fake ----------
    plt.figure()
    # plt.plot(bin_centers, fakeTrack, "o-", color='green')
    plt.errorbar(bin_centers, fakeTrack, yerr=errors_fake, fmt='o', color='green')
    #if i want to make the error bars blue
    plt.title("Fake Track Rate vs pT")
    plt.xlabel("pT [GeV]")
    plt.ylabel("fake track")
    plt.ylim(0, 1.05)
    plt.grid()
    # plt.legend()
    plt.savefig(os.path.join(output_dir, output_fake_track))
    #plt.show()

    # print(f"Saved: {output_fake_track}")
    plt.close()

    # ---------- Plot duplicate ----------
    plt.figure()
    # plt.plot(bin_centers, duplicateRate, "o-", color='orange')
    plt.errorbar(bin_centers, duplicateRate, yerr=errors_duplicate, fmt='o', color='orange')
    plt.title("Duplicate Track Rate vs pT")
    plt.xlabel("pT [GeV]")
    plt.ylabel("duplicate ration")
    plt.ylim(0, 1.05)
    plt.grid()
    # plt.legend()
    plt.savefig(os.path.join(output_dir, output_duplicate_track))
    #plt.show()

    # print(f"Saved: {output_duplicate_track}")
    plt.close()

    # ---------- Plot true matched ----------
    plt.figure()
    # plt.plot(bin_centers, matchedEfficiency, "o-", color='red')
    plt.errorbar(bin_centers, matchedEfficiency, yerr=errors_matched, fmt='o', color='red')
    plt.title("True Matched Efficiency vs pT")
    plt.xlabel("pT [GeV]")
    plt.ylabel("true matched efficiency")
    plt.ylim(0, 1.05)
    plt.grid()
    # plt.legend()
    plt.savefig(os.path.join(output_dir, output_matched_track))
    # #plt.show()

    # print(f"Saved: {output_matched_track}")
    plt.close()

    # --- Plot all of them as scatter plots on the same canvas ---

    # plt.figure(figsize=(10, 6))
    # plt.errorbar(bin_centers, efficiency, yerr=errors_purity, fmt='o', color='blue', label='Track Efficiency')
    # plt.errorbar(bin_centers, fakeTrack, yerr=errors_fake, fmt='o', color='green', label='Fake Rate')
    # plt.errorbar(bin_centers, duplicateRate, yerr=errors_duplicate, fmt='o', color='orange', label='Duplicate Rate')
    # plt.errorbar(bin_centers, matchedEfficiency, yerr=errors_matched, fmt='o', color='red', label='True Matched Efficiency')
    # plt.xlabel("pT [GeV]")
    # plt.ylabel("Fraction")
    # plt.ylim(0, 1.05)

def track_metrics_each_particle_old(
    tracks_file,
    particles_file,
    track_tree="tracksummary",
    particle_tree="particles",
    pt_bins=np.linspace(0.1, 0.5, 50),
    output_purity="track_efficiency_vs_pt_each.png",
    output_fake_track="fake_track_vs_pt_each.png",
    output_duplicate_track="duplicate_ratio_vs_pt_each.png",
    output_matched_track="matched_efficiency_vs_pt_each.png",
    out_dir=None,
):
    
    # ---------- Create output directory ----------


    output_dir = out_dir if out_dir is not None else os.path.join("z_btr_files", "efficiency_plots")
    os.makedirs(output_dir, exist_ok=True)

    # ---------- READ PARTICLES ----------

    f_particles = uproot.open(particles_file)
    t_particles = f_particles[particle_tree]
    particles = t_particles.arrays(["particle_id", "pt", "particle_type"], library="ak")
    
    #print the particle types and their counts
    unique_types, counts = np.unique(particles["particle_type"], return_counts=True)
    print("Particle types and counts:")
    for t, c in zip(unique_types, counts):
        print(f"Type: {t}, Count: {c}")


    # ---------- READ TRACKS ----------

    f_tracks = uproot.open(tracks_file)
    t_tracks = f_tracks[track_tree]

    #also load track pt to plot against
    #tracks = t_tracks.arrays(["majorityParticleId", "t_pT"], library="ak" )
    tracks = t_tracks.arrays(["majorityParticleId", "t_pT", "trackClassification", "eQOP_fit", "eTHETA_fit"], library="ak")

    # ---------- Histogram ----------
    
    #makes all histograms zero
    # hist_all = np.zeros(len(pt_bins) - 1)
    # hist_good = np.zeros(len(pt_bins) - 1)
    # hist_fake = np.zeros(len(pt_bins) - 1)
    # hist_duplicate = np.zeros(len(pt_bins) - 1)
    # hist_matched = np.zeros(len(pt_bins) - 1)

    hist_all_protons = np.zeros(len(pt_bins) - 1) # total count
    hist_all_pions = np.zeros(len(pt_bins) - 1)
    hist_all_electrons = np.zeros(len(pt_bins) - 1)

    hist_good_protons = np.zeros(len(pt_bins) - 1) # truth matched
    hist_good_pions = np.zeros(len(pt_bins) - 1)
    hist_good_electrons = np.zeros(len(pt_bins) - 1)

    hist_fake_protons = np.zeros(len(pt_bins) - 1) # fake, not matched to any particle
    hist_fake_pions = np.zeros(len(pt_bins) - 1)
    hist_fake_electrons = np.zeros(len(pt_bins) - 1)

    hist_duplicate_protons = np.zeros(len(pt_bins) - 1) # duplicates
    hist_duplicate_pions = np.zeros(len(pt_bins) - 1)
    hist_duplicate_electrons = np.zeros(len(pt_bins) - 1)

    hist_matched_protons = np.zeros(len(pt_bins) - 1) # matched to a particle, either good or duplicate
    hist_matched_pions = np.zeros(len(pt_bins) - 1)
    hist_matched_electrons = np.zeros(len(pt_bins) - 1)

    n_events = len(tracks["majorityParticleId"])

    print("Number of events:", n_events)

     # ---------- LOOP THROUGH EVENTS ----------

    # countfake = 0
    # countgood = 0
    # counttotal = 0
    # countduplicate = 0
    # countmatched = 0

    countfake_protons = 0
    countfake_pions = 0
    countfake_electrons = 0

    countgood_protons = 0
    countgood_pions = 0
    countgood_electrons = 0

    counttotal_protons = 0
    counttotal_pions = 0
    counttotal_electrons = 0

    countduplicate_protons = 0
    countduplicate_pions = 0
    countduplicate_electrons = 0

    countmatched_protons = 0
    countmatched_pions = 0
    countmatched_electrons = 0

    for event in range(n_events):

        track_truth = tracks["majorityParticleId"][event]
        track_pts = tracks["t_pT"][event]
        track_class = tracks["trackClassification"][event]
        track_qop = tracks["eQOP_fit"][event]
        track_theta = tracks["eTHETA_fit"][event]
        particle_type = particles["particle_type"][event]

        for ids, pt, classification, qop_fit, theta_fit, particle_type in zip(track_truth, track_pts, track_class, track_qop, track_theta, particle_type):
            # if len(ids) < 3:
            #     continue  # skip broken entries

            if particle_type == -2212 or particle_type == 2212:
                counttotal_protons += 1
            elif particle_type == -211 or particle_type == 211:
                counttotal_pions += 1
            else:
                counttotal_electrons += 1

            if classification == 0:
                pt = np.sin(theta_fit) / abs(qop_fit)

            if pt < 0.1:
                continue

            bin_index = np.digitize(pt, pt_bins) - 1
            if not (0 <= bin_index < len(hist_all)):
                if particle_type == -2212 or particle_type == 2212:
                    countfake_protons += 1
                elif particle_type == -211 or particle_type == 211:
                    countfake_pions += 1
                else:
                    countfake_electrons += 1
                continue

            if particle_type == -2212 or particle_type == 2212:
                hist_all_protons[bin_index] += 1
            elif particle_type == -211 or particle_type == 211:
                hist_all_pions[bin_index] += 1
            else:
                hist_all_electrons[bin_index] += 1

            if classification == 1 or classification == 2:   # truth-matched track
                if particle_type == -2212 or particle_type == 2212:
                    hist_matched_protons[bin_index] += 1
                    countmatched_protons += 1
                elif particle_type == -211 or particle_type == 211:
                    hist_matched_pions[bin_index] += 1
                    countmatched_pions += 1
                else:
                    hist_matched_electrons[bin_index] += 1
                    countmatched_electrons += 1
            
            if classification == 1:   # truth-matched track
                if particle_type == -2212 or particle_type == 2212:
                    hist_good_protons[bin_index] += 1
                    countgood_protons += 1
                elif particle_type == -211 or particle_type == 211:
                    hist_good_pions[bin_index] += 1
                    countgood_pions += 1
                else:
                    hist_good_electrons[bin_index] += 1
                    countgood_electrons += 1
                
            if classification == 2: # duplicate track
                if particle_type == -2212 or particle_type == 2212:
                    hist_duplicate_protons[bin_index] += 1
                    countduplicate_protons += 1
                elif particle_type == -211 or particle_type == 211:
                    hist_duplicate_pions[bin_index] += 1
                    countduplicate_pions += 1
                else:
                    hist_duplicate_electrons[bin_index] += 1
                    countduplicate_electrons += 1

            if classification == 0: # fake track
                if particle_type == -2212 or particle_type == 2212:
                    hist_fake_protons[bin_index] += 1
                    countfake_protons += 1
                elif particle_type == -211 or particle_type == 211:
                    hist_fake_pions[bin_index] += 1
                    countfake_pions += 1
                else:
                    hist_fake_electrons[bin_index] += 1
                    countfake_electrons += 1

    # ---------- Compute tracking efficiency ----------
    efficiency_protons = np.zeros_like(hist_all_protons)

    for i in range(len(hist_all_protons)):
        if hist_all_protons[i] > 0:
            efficiency_protons[i] = hist_matched_protons[i] / hist_all_protons[i]

    bin_centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])

    efficiency_pions = np.zeros_like(hist_all_pions)

    for i in range(len(hist_all_pions)):
        if hist_all_pions[i] > 0:
            efficiency_pions[i] = hist_matched_pions[i] / hist_all_pions[i]


    efficiency_electrons = np.zeros_like(hist_all_electrons)

    for i in range(len(hist_all_electrons)):
        if hist_all_electrons[i] > 0:
            efficiency_electrons[i] = hist_matched_electrons[i] / hist_all_electrons[i]


    # ---------- Errors ----------

    errors_purity = np.zeros_like(efficiency)

    for i in range(len(hist_all)):
        if hist_all[i] > 0:
            p = efficiency[i]
            N = hist_all[i]
            errors_purity[i] = np.sqrt(p * (1 - p) / N)

    # ---------- Compute fake track ----------

    fakeTrack = np.zeros_like(hist_all)

    for i in range(len(hist_all)):
        if hist_all[i] > 0:
            fakeTrack[i] = hist_fake[i] / hist_all[i]

    # ---------- Errors ----------

    errors_fake = np.zeros_like(fakeTrack)

    for i in range(len(hist_all)):
        if hist_all[i] > 0:
            p = fakeTrack[i]
            N = hist_all[i]
            errors_fake[i] = np.sqrt(p * (1 - p) / N)

    # ---------- Compute duplicate rate ----------

    duplicateRate = np.zeros_like(hist_all)

    for i in range(len(hist_all)): 
        if hist_all[i] > 0:
            duplicateRate[i] = hist_duplicate[i] / hist_all[i]

    # ---------- Errors ----------

    errors_duplicate = np.zeros_like(duplicateRate)

    for i in range(len(hist_all)):
        if hist_all[i] > 0:
            p = duplicateRate[i]
            N = hist_all[i]
            errors_duplicate[i] = np.sqrt(p * (1 - p) / N)

    # ---------- Compute true matched efficiency ----------

    matchedEfficiency = np.zeros_like(hist_all)
    for i in range(len(hist_all)):
        if hist_all[i] > 0:
            matchedEfficiency[i] = hist_good[i] / hist_all[i]

    # ---------- Errors ----------

    errors_matched = np.zeros_like(matchedEfficiency)
    for i in range(len(hist_all)):
        if hist_all[i] > 0:
            p = matchedEfficiency[i]
            N = hist_all[i]
            errors_matched[i] = np.sqrt(p * (1 - p) / N)

    # ----- DEBUG ----

    #see how many tracks are in each bin and how many are good
    # for i in range(len(hist_all)):
    #     print(f"Bin {i}: total={hist_all[i]}, good={hist_good[i]}")

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

    #for event in range(5):
    #     for ids in tracks["majorityParticleId"][event]:
    #         print(ids)

    print("\n")
    
    print("For loop - TRUTH - need to match this")
    print("Total tracks:", total_count)
    print("fake:", zero_count) 
    print("matched:", unu_count + doi_count)
    print("true matched:", unu_count)
    print("duplicate:", doi_count)
    print("douplicate - classification:", doi_count/total_count)
    print("truth-matched - classification:", unu_count/total_count)
    print("matced - classification:", (unu_count + doi_count)/total_count)
    print("fake - classification:", zero_count/total_count)

    print("\nHistogram counters:")
    print("Total tracks:", counttotal)
    print("Fake tracks:", countfake)
    print("Matched tracks:", countgood)
    print("Duplicate tracks:", countduplicate)
    print("Track Efficiency:", countmatched/counttotal)
    print("Fake Rate:", countfake/counttotal)
    print("Duplicate Rate:", countduplicate/counttotal)

    print("\nHistograms:")
    print("Total tracks in histogram:", np.sum(hist_all))
    print("Fake tracks:", np.sum(hist_fake))
    print("Matched tracks:", np.sum(hist_good))
    print("Duplicate tracks:", np.sum(hist_duplicate))
    print("Track Efficiency - histogram:", np.sum(hist_matched) / np.sum(hist_all))
    print("Fake rate - histogram:", np.sum(hist_fake) / np.sum(hist_all))
    print("Duplicate Rate - histogram:", np.sum(hist_duplicate) / np.sum(hist_all))

    print("\n")

    # ---------- Plot efficiency ----------
    plt.figure()
    # plt.plot(bin_centers, efficiency, "o-", color='blue')
    plt.errorbar(bin_centers, efficiency, yerr=errors_purity, fmt='o', color='blue')
    #if i want to make the error bars blue
    plt.title("Track Efficiency vs pT")
    plt.xlabel("pT [GeV]")
    plt.ylabel("Track Efficiency")
    plt.ylim(0, 1.05)
    plt.grid()
    # plt.legend()
    # plt.savefig(output_purity)
    plt.savefig(os.path.join(output_dir, output_purity))
    #plt.show()

    # print(f"Saved: {output_purity}")
    plt.close()

    # ---------- Plot fake ----------
    plt.figure()
    # plt.plot(bin_centers, fakeTrack, "o-", color='green')
    plt.errorbar(bin_centers, fakeTrack, yerr=errors_fake, fmt='o', color='green')
    #if i want to make the error bars blue
    plt.title("Fake Track Rate vs pT")
    plt.xlabel("pT [GeV]")
    plt.ylabel("fake track")
    plt.ylim(0, 1.05)
    plt.grid()
    # plt.legend()
    plt.savefig(os.path.join(output_dir, output_fake_track))
    #plt.show()

    # print(f"Saved: {output_fake_track}")
    plt.close()

    # ---------- Plot duplicate ----------
    plt.figure()
    # plt.plot(bin_centers, duplicateRate, "o-", color='orange')
    plt.errorbar(bin_centers, duplicateRate, yerr=errors_duplicate, fmt='o', color='orange')
    plt.title("Duplicate Track Rate vs pT")
    plt.xlabel("pT [GeV]")
    plt.ylabel("duplicate ration")
    plt.ylim(0, 1.05)
    plt.grid()
    # plt.legend()
    plt.savefig(os.path.join(output_dir, output_duplicate_track))
    #plt.show()

    # print(f"Saved: {output_duplicate_track}")
    plt.close()

    # ---------- Plot true matched ----------
    plt.figure()
    # plt.plot(bin_centers, matchedEfficiency, "o-", color='red')
    plt.errorbar(bin_centers, matchedEfficiency, yerr=errors_matched, fmt='o', color='red')
    plt.title("True Matched Efficiency vs pT")
    plt.xlabel("pT [GeV]")
    plt.ylabel("true matched efficiency")
    plt.ylim(0, 1.05)
    plt.grid()
    # plt.legend()
    plt.savefig(os.path.join(output_dir, output_matched_track))
    # #plt.show()

    # print(f"Saved: {output_matched_track}")
    plt.close()

    # --- Plot all of them as scatter plots on the same canvas ---

    # plt.figure(figsize=(10, 6))
    # plt.errorbar(bin_centers, efficiency, yerr=errors_purity, fmt='o', color='blue', label='Track Efficiency')
    # plt.errorbar(bin_centers, fakeTrack, yerr=errors_fake, fmt='o', color='green', label='Fake Rate')
    # plt.errorbar(bin_centers, duplicateRate, yerr=errors_duplicate, fmt='o', color='orange', label='Duplicate Rate')
    # plt.errorbar(bin_centers, matchedEfficiency, yerr=errors_matched, fmt='o', color='red', label='True Matched Efficiency')
    # plt.xlabel("pT [GeV]")
    # plt.ylabel("Fraction")
    # plt.ylim(0, 1.05)

def track_metrics_each_particle(
    tracks_file,
    particles_file,
    track_tree="tracksummary",
    particle_tree="particles",
    pt_bins=np.linspace(0.1, 0.5, 50),
    output_purity="track_efficiency_vs_pt_each.png",
    output_fake_track="fake_track_vs_pt_each.png",
    output_duplicate_track="duplicate_ratio_vs_pt_each.png",
    output_matched_track="matched_efficiency_vs_pt_each.png",
    out_dir=None,
):

    # ---------- Output directory ----------
    output_dir = out_dir if out_dir else os.path.join("z_btr_files", "efficiency_plots")
    os.makedirs(output_dir, exist_ok=True)

    # ---------- Particle categories ----------
    particle_map = {
        "proton": [2212, -2212],
        "pion": [211, -211],
        "electron": [11, -11],
    }

    def get_particle_name(p_type):
        for name, codes in particle_map.items():
            if p_type in codes:
                return name
        return None

    def make_hist_dict():
        return {k: np.zeros(len(pt_bins) - 1) for k in particle_map}

    def make_counter_dict():
        return {k: 0 for k in particle_map}

    # ---------- Read particles ----------
    particles = uproot.open(particles_file)[particle_tree].arrays(
        ["particle_id", "pt", "particle_type"], library="ak"
    )

    # ---------- Read tracks ----------
    tracks = uproot.open(tracks_file)[track_tree].arrays(
        ["majorityParticleId", "t_pT", "trackClassification", "eQOP_fit", "eTHETA_fit"],
        library="ak",
    )

    # ---------- Histograms ----------
    hist_all = make_hist_dict()
    hist_good = make_hist_dict()
    hist_fake = make_hist_dict()
    hist_duplicate = make_hist_dict()
    hist_matched = make_hist_dict()

    # ---------- Counters ----------
    counttotal = make_counter_dict()
    countgood = make_counter_dict()
    countfake = make_counter_dict()
    countduplicate = make_counter_dict()
    countmatched = make_counter_dict()

    n_events = len(tracks["majorityParticleId"])
    print("Number of events:", n_events)

    # ---------- Loop ----------
    for event in range(n_events):

        track_truth = tracks["majorityParticleId"][event]
        track_pts = tracks["t_pT"][event]
        track_class = tracks["trackClassification"][event]
        track_qop = tracks["eQOP_fit"][event]
        track_theta = tracks["eTHETA_fit"][event]
        particle_types = particles["particle_type"][event]

        for pid, pt, classification, qop, theta, p_type in zip(
            track_truth, track_pts, track_class, track_qop, track_theta, particle_types
        ):

            pname = get_particle_name(p_type)
            if pname is None:
                continue

            counttotal[pname] += 1

            if classification == 0:
                pt = np.sin(theta) / abs(qop)

            if pt < 0.1:
                continue

            bin_index = np.digitize(pt, pt_bins) - 1
            if not (0 <= bin_index < len(pt_bins) - 1):
                countfake[pname] += 1
                continue

            hist_all[pname][bin_index] += 1

            if classification in [1, 2]:
                hist_matched[pname][bin_index] += 1
                countmatched[pname] += 1

            if classification == 1:
                hist_good[pname][bin_index] += 1
                countgood[pname] += 1

            if classification == 2:
                hist_duplicate[pname][bin_index] += 1
                countduplicate[pname] += 1

            if classification == 0:
                hist_fake[pname][bin_index] += 1
                countfake[pname] += 1

    # ---------- Compute metrics ----------
    efficiency = {}
    fakeTrack = {}
    duplicateRate = {}
    matchedEfficiency = {}

    for pname in particle_map:
        efficiency[pname] = np.divide(
            hist_matched[pname],
            hist_all[pname],
            out=np.zeros_like(hist_all[pname]),
            where=hist_all[pname] > 0,
        )

        fakeTrack[pname] = np.divide(
            hist_fake[pname],
            hist_all[pname],
            out=np.zeros_like(hist_all[pname]),
            where=hist_all[pname] > 0,
        )

        duplicateRate[pname] = np.divide(
            hist_duplicate[pname],
            hist_all[pname],
            out=np.zeros_like(hist_all[pname]),
            where=hist_all[pname] > 0,
        )

        matchedEfficiency[pname] = np.divide(
            hist_good[pname],
            hist_all[pname],
            out=np.zeros_like(hist_all[pname]),
            where=hist_all[pname] > 0,
        )

    bin_centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])

    # ---------- Plot ----------
    colors = {"proton": "red", "pion": "blue", "electron": "green"}

    def plot_metric(metric_dict, title, ylabel, filename):
        plt.figure()
        for pname in particle_map:
            plt.plot(
                bin_centers,
                metric_dict[pname],
                "o-",
                label=pname,
                color=colors[pname],
            )
        plt.legend()
        plt.title(title)
        plt.xlabel("pT [GeV]")
        plt.ylabel(ylabel)
        plt.ylim(0, 1.05)
        plt.grid()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    plot_metric(efficiency, "Track Efficiency vs pT", "Efficiency", output_purity)
    plot_metric(fakeTrack, "Fake Rate vs pT", "Fake Rate", output_fake_track)
    plot_metric(duplicateRate, "Duplicate Rate vs pT", "Duplicate Rate", output_duplicate_track)
    plot_metric(matchedEfficiency, "True Matched Efficiency vs pT", "Matched Efficiency", output_matched_track)

    # ---------- Print summary ----------
    print("\nSummary per particle:")
    for pname in particle_map:
        if counttotal[pname] > 0:
            print(f"\n{pname}:")
            print("  Total:", counttotal[pname])
            print("  Efficiency:", countmatched[pname] / counttotal[pname])
            print("  Fake rate:", countfake[pname] / counttotal[pname])
            print("  Duplicate rate:", countduplicate[pname] / counttotal[pname])

# if __name__ == "__main__":

#     track_metrics_all_particles(
#         tracks_file="tracksummary_ckf.root",
#         particles_file="particles.root",
#     )

if __name__ == "__main__":

    p = argparse.ArgumentParser(description="Plot tracking metrics vs pT.")
    p.add_argument("--mode", choices=["all_particles", "each_particle"], required=True,
                   help="Which function to run")
    p.add_argument("--tracks", default="tracksummary_ckf.root",
                   help="Path to tracks ROOT file (default: tracksummary_ckf.root)")
    p.add_argument("--particles", default="particles.root",
                   help="Path to particles ROOT file (default: particles.root)")
    p.add_argument("--out-dir", default=None,
               help="Directory to save output plots (default: z_btr_files/efficiency_plots)")
    args = p.parse_args()

    if args.mode == "all_particles":
        track_metrics_all_particles(
            tracks_file=args.tracks,
            particles_file=args.particles,
            out_dir=args.out_dir,
        )
    elif args.mode == "each_particle":
        track_metrics_each_particle(
            tracks_file=args.tracks,
            particles_file=args.particles,
            out_dir=args.out_dir,
        )