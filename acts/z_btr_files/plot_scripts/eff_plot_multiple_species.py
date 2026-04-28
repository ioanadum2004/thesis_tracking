import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import os

# def true_efficiency_vs_pt_3(
#     particles_file,
#     tracks_file,
#     particle_tree="particles",
#     track_tree="tracksummary",
#     pt_bins=np.linspace(0.01, 10, 50),
#     output="b_particle_efficiency_vs_pt.png",
# ):

    # ---------- READ PARTICLES ----------

    # f_particles = uproot.open(particles_file)
    # t_particles = f_particles[particle_tree]
    # # particles = t_particles.arrays(["particle", "pt"], library="ak")
    # particles = t_particles.arrays(["particle_id", "pt"], library="ak")
    

    # print("Nr of events:", len(particles["particle_id"]))

def track_metrics_classification_all_particles(
    tracks_file,
    particles_file,
    track_tree="tracksummary",
    particle_tree="particles",
    pt_bins=np.linspace(0.1, 0.5, 50),
    output_purity="track_efficiency_vs_pt.png",
    output_fake_track="fake_track_vs_pt.png",
    output_duplicate_track="duplicate_ratio_vs_pt.png",
    output_matched_track="matched_efficiency_vs_pt.png",
):
    
    # ---------- Create output directory ----------


    output_dir = os.path.join("z_btr_files", "efficiency_plots")
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

    n_events = len(tracks["majorityParticleId"])
    t_tracks = f_tracks[track_tree]

    #also load track pt to plot against
    #tracks = t_tracks.arrays(["majorityParticleId", "t_pT"], library="ak" )
    tracks = t_tracks.arrays(["majorityParticleId", "t_pT", "trackClassification", "eQOP_fit", "eTHETA_fit"], library="ak")

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

            bin_index = np.digitize(pt, pt_bins) - 1
            if not (0 <= bin_index < len(hist_all)):
                countfake += 1
                continue

            hist_all[bin_index] += 1
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

def track_metrics_classification_all_particles(
    tracks_file,
    particles_file,
    track_tree="tracksummary",
    particle_tree="particles",
    pt_bins=np.linspace(0.1, 0.5, 50),
    output_purity="track_efficiency_vs_pt.png",
    output_fake_track="fake_track_vs_pt.png",
    output_duplicate_track="duplicate_ratio_vs_pt.png",
    output_matched_track="matched_efficiency_vs_pt.png",
):
    
    # ---------- Create output directory ----------


    output_dir = os.path.join("z_btr_files", "efficiency_plots")
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
    hist_all = np.zeros(len(pt_bins) - 1)
    hist_good = np.zeros(len(pt_bins) - 1)
    hist_fake = np.zeros(len(pt_bins) - 1)
    hist_duplicate = np.zeros(len(pt_bins) - 1)
    hist_matched = np.zeros(len(pt_bins) - 1)

    hist_all_protons = np.zeros(len(pt_bins) - 1)

    n_events = len(tracks["majorityParticleId"])

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

            bin_index = np.digitize(pt, pt_bins) - 1
            if not (0 <= bin_index < len(hist_all)):
                countfake += 1
                continue

            hist_all[bin_index] += 1
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

if __name__ == "__main__":

    track_metrics_classification_all_particles(
        tracks_file="tracksummary_ckf.root",
        particles_file="particles.root",
    )