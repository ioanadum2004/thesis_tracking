import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

#python z_btr_files/old_plots/plots_latest.py

# TO DO - add function to read particles

# TO DO - add function to read tracks

#includes distribution plot
# UPDATE: it does work, but doesnt match the ckf efficiency
# Efficiency = reconstructed truth / total truth
def true_efficiency_vs_pt_2(
    particles_file,
    tracks_file,
    particle_tree="particles",
    track_tree="tracksummary",
    pt_bins=np.linspace(0.01, 10, 50),
    #pt_bins=np.linspace(0, 10, 50),
    output="maybe_true_efficiency_vs_pt.png",
):

    # ---------- READ PARTICLES ----------


    f_particles = uproot.open(particles_file)
    t_particles = f_particles[particle_tree]

    particles = t_particles.arrays(["particle", "pt"], library="ak")

    # particles["particle"] =
    # [
    # [101, 102, 103],     # event 0
    # [201, 202]           # event 1
    # ]

    # particles["pt"] =
    # [
    # [0.5, 1.2, 1.8],
    # [0.4, 1.0]
    # ]
    # len particles will be 1000 events

    print("particle length")
    print(len(particles["particle"]))

    #check number of particles in first event
    # print(len(particles["particle"][0]))
    # print(len(particles["particle"][1]))
    # print(len(particles["particle"][2]))


    # ---------- READ TRACKS ----------


    f_tracks = uproot.open(tracks_file)
    t_tracks = f_tracks[track_tree]

    tracks = t_tracks.arrays(["majorityParticleId"], library="ak")

    # tracks["majorityParticleId"] =
    # [
    # [101, 0, 103],    # event 0
    # [201]             # event 1
    # ]

    # ---------- Histogram ----------
    
    #makes all histograms zero
    hist_all = np.zeros(len(pt_bins) - 1)
    hist_rec = np.zeros(len(pt_bins) - 1)
    #these store total truth particles per bin and total reconstructed particles per bin

    n_events = len(particles["particle"])

    total_tracks = 0
    for event in range(n_events):
        total_tracks += len(tracks["majorityParticleId"][event])
    print("Average tracks per event:", total_tracks / n_events)


     # ---------- LOOP THROUGH EVENT ----------


    #  also filter through then again if pt<0.5

    # 0 = 
    # 1 = vertex - related to multiplicity - how many particles are generated in a collision
    # 2 = majority id (the one we want to use to match to truth)
    # 3 = 

    # print("with 0")
    # print("without 0")

    for event in range(n_events):

        # true particles in this event
        particle_ids = particles["particle"][event].to_list()
        particle_pts = particles["pt"][event].to_list()

            # reconstructed track matches in this event
            #matched_ids = tracks["majorityParticleId"][event].to_list()

        # --- v1 ---

        # matched_ids = tracks["majorityParticleId"][event]

        # print("match ids", matched_ids)

            # remove fake tracks (id = 0)
            #matched_ids = [mid for mid in matched_ids if mid != 0]
        # matched_ids = matched_ids[matched_ids != 0]

        # print(matched_ids)

        #theres going to be an error here if no track is reconstructed
        #if instead of deleting the 0s i just ignore them in the loop, then we can still get the efficiency for events where no tracks were reconstructed.

        # --- v2 ---

        #     # get track → truth links for this event
        # track_truth = tracks["majorityParticleId"][event]

        #     # flatten everything
        # all_truth_ids = ak.flatten(track_truth)

        #     # remove zeros
        # all_truth_ids = all_truth_ids[all_truth_ids != 0]

        #     # make unique set
        # matched_ids = set(all_truth_ids.to_list())

        # --v3--

        track_truth = tracks["majorityParticleId"][event]

        # take only second entry of each track
        majority_ids = [ids[2] for ids in track_truth if len(ids) > 0]

        # remove zeros
        # I COMMENTED THIS OUT, ADD BACK IN IF IT MAKES SENSE
        majority_ids = [mid for mid in majority_ids if mid != 0] # this doesnt make any differece
    
        matched_ids = set(majority_ids)

        # debug: eff should be 0
        #matched_ids = set()

        # debug: eff should be 1
        #matched_ids = set(particle_ids)

        # if event == 1:
        #     print("Truth IDs:", particle_ids)
        #     print("Track majority raw:", track_truth)

        #     majority_ids = [tid[0] for tid in track_truth if tid[0] != 0]
        #     print("Majority IDs cleaned:", majority_ids)

        #     for pid in particle_ids:
        #         print(f"Particle {pid} reconstructed? {pid in majority_ids}")

        if event == 304:
            print("Matched IDs 304:", majority_ids)
            print("Truth IDs:", particle_ids)


        # example v3:

        #particle_ids = [100, 101, 102, 103]
        # majorityParticleId =
        # [
        #     [100, 0, 0],   # track 0
        #     [102, 101],    # track 1 (merged)
        #     [0, 0, 0],     # track 2 (fake)
        #     [100],         # track 3 (duplicate)
        # ]

        # particle_ids = [1,2,3]
        # particle_pts = [0.5,1.2,1.8]
        # matched_ids = [1, 1, 3] - removes the 0 that was in the majorityParticleId array

        # ---------- LOOP OVER PARTICLES ----------
        for pid, pt in zip(particle_ids, particle_pts):

            #filter for paritcles under 0.5 GeV
            if pt <= 0.01:
                continue

            # find which bin this pt belongs to
            bin_index = np.digitize(pt, pt_bins) - 1
            #digitize is used to find the index of the bin that each pt value belongs to. 
            #It returns an array of indices corresponding to the bins defined by pt_bins. 
            #The -1 is used to convert from 1-based indexing (which digitize uses) to 0-based indexing (which Python uses).

            if 0 <= bin_index < len(hist_all):

                # count total particle
                
                hist_all[bin_index] += 1
                #so this knows how many particles are in each pt bin
                #asta e numitorul

                # check if reconstructed
                #and this created a histogram of how many particles in each pt bin were reconstructed 
                for mid in matched_ids:
                    if mid != 0: # ignore fake tracks
                        if pid == mid:
                            hist_rec[bin_index] += 1
                            break   # stop once found
                    # if pid == mid:
                    #     hist_rec[bin_index] += 1
                    #     break   # stop once found

    # ---------- Compute efficiency ----------
    efficiency = np.zeros_like(hist_all)

    for i in range(len(hist_all)):
        if hist_all[i] > 0:
            efficiency[i] = hist_rec[i] / hist_all[i]

    bin_centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])

    # ---------- Errors ----------

    errors = np.zeros_like(efficiency)

    for i in range(len(hist_all)):
        if hist_all[i] > 0:
            eff = efficiency[i]
            N = hist_all[i]
            errors[i] = np.sqrt(eff * (1 - eff) / N)


    # ---------- Debug ----------
    print("Total truth particles:", np.sum(hist_all))
    print("Total reconstructed truth particles:", np.sum(hist_rec))
    print("Global efficiency:", np.sum(hist_rec) / np.sum(hist_all))
    # print("Efficiency per bin:", efficiency)
    print("total efficiency, average efficiency:", )

    # Total truth particles: 4000.0 -> hist all is calculated correct
    # Total reconstructed truth particles: 1762.0
    # Global efficiency: 0.4405

    #hist_all is counting how many particles are in each pt bin, and hist_rec is counting how many of those were reconstructed.

    # print efficiency for each bin
    # for i in range(len(hist_all)):
    #     if hist_all[i] > 0:
    #         print(f"Bin {i}: all={hist_all[i]}, rec={hist_rec[i]}, eff={hist_rec[i]/hist_all[i]:.3f}")

    low_pt = 0
    high_pt = 0

    for event in range(n_events):
        for pt in particles["pt"][event]:
            if pt < 0.01:
                low_pt += 1
            else:
                high_pt += 1

    print("Below 0.5 GeV:", low_pt)
    print("Above 0.5 GeV:", high_pt)

    #this doest print anything so there are no particles reconstructed below, 0.5, but the graph shows there are
    # for event in range(n_events):
    #     particle_ids = particles["particle"][event].to_list()
    #     particle_pts = particles["pt"][event].to_list()
    #     matched_ids = tracks["majorityParticleId"][event].to_list()
    #     matched_ids = [mid for mid in matched_ids if mid != 0]

    #     for pid, pt in zip(particle_ids, particle_pts):
    #         if pt < 0.5:
    #             if pid in matched_ids:
    #                 print(f"Event {event}: low-pT particle {pid} matched!")

    # for event in range(5):
    #     print("Particle pts:", particles["pt"][event].to_list())
    #     print("Binned indices:", np.digitize(particles["pt"][event].to_list(), pt_bins)-1)

    # for i, (all_, rec) in enumerate(zip(hist_all, hist_rec)):
    #     print(f"Bin {i}: edges=({pt_bins[i]:.3f},{pt_bins[i+1]:.3f}), all={all_}, rec={rec}")

    # for i in range(3):  # first few bins
    #     print(f"LOW BIN {i}:")
    #     print(f"  edges=({pt_bins[i]:.3f},{pt_bins[i+1]:.3f})")
    #     print(f"  all={hist_all[i]}")
    #     print(f"  rec={hist_rec[i]}")

    all_pts = []

    for event in range(n_events):
        all_pts.extend(particles["pt"][event].to_list())

    print("Minimum pT in dataset:", min(all_pts))

    for i in range(5):
        print(
            f"Bin {i}: "
            f"range=({pt_bins[i]:.3f},{pt_bins[i+1]:.3f}) "
            f"all={hist_all[i]}, "
            f"rec={hist_rec[i]}, "
            f"eff={efficiency[i]:.3f}"
        )

    count_very_low = 0

    for pt in all_pts:
        if pt < 0.1:
            count_very_low += 1

    print("Particles below 0.1 GeV:", count_very_low)

    #plot nr of truth particles per bin

    # plt.figure()
    # plt.plot(bin_centers, hist_all, "o-")
    # plt.xlabel("pT")
    # plt.ylabel("Number of truth particles")
    # plt.savefig("truth_per_bin.png")
    # plt.close()

    #debugging: print particles with pt < 0.2 and check if they are reconstructed, should be 0, but the graph shows there are some reconstructed particles below 0.2, so maybe we can find out which ones are they and why they are reconstructed
    
    # for event in range(n_events):
    #     particle_ids = particles["particle"][event].to_list()
    #     particle_pts = particles["pt"][event].to_list()
    #     track_truth = tracks["majorityParticleId"][event]

    #     #third element in the list is the particle id
    #     majority_ids = [ids[2] for ids in track_truth if len(ids) > 0]
    #     majority_ids = [mid for mid in majority_ids if mid != 0]
    #     matched_ids = set(majority_ids)

    #     for pid, pt in zip(particle_ids, particle_pts):
    #         if pt < 0.2:
    #             print(
    #                 f"event={event}, pid={pid}, pt={pt:.3f}, "
    #                 f"reconstructed={pid in matched_ids}"
    #             )


    # ---------- Plot efficiency ----------
    plt.figure()
    plt.plot(bin_centers, efficiency, "o-", color='blue')
    plt.errorbar(bin_centers, efficiency, yerr=errors, fmt='o', color='blue')
    #if i want to make the error bars blue
    plt.xlabel("pT [GeV]")
    plt.ylabel("True Efficiency")
    plt.ylim(0, 1.05)
    plt.grid()
    plt.legend()
    plt.savefig(output)
    #plt.show()

    print(f"Saved: {output}")
    plt.close()

    # ---------- pT distribution ----------

    all_pts = []

    for event in range(n_events):
        pts = particles["pt"][event].to_list()
        all_pts.extend(pts)

    plt.figure()
    plt.hist(all_pts, bins=50)
    plt.xlabel("pT")
    plt.ylabel("Truth particles")
    plt.savefig("truth_pt_distribution.png")
    plt.close()

def true_efficiency_vs_pt_3(
    particles_file,
    tracks_file,
    particle_tree="particles",
    track_tree="tracksummary",
    pt_bins=np.linspace(0.01, 10, 50),
    output="b_particle_efficiency_vs_pt.png",
):

    # ---------- READ PARTICLES ----------

    f_particles = uproot.open(particles_file)
    t_particles = f_particles[particle_tree]
    # particles = t_particles.arrays(["particle", "pt"], library="ak")
    particles = t_particles.arrays(["particle_id", "pt"], library="ak")
    

    print("Nr of events:", len(particles["particle_id"]))

    # ---------- READ TRACKS ----------

    f_tracks = uproot.open(tracks_file)
    t_tracks = f_tracks[track_tree]
    tracks = t_tracks.arrays(["majorityParticleId"], library="ak")

    # ---------- Histogram ----------
    
    #makes all histograms zero
    hist_all = np.zeros(len(pt_bins) - 1)
    hist_rec = np.zeros(len(pt_bins) - 1)

    n_events = len(particles["particle_id"])

    total_tracks = 0
    for event in range(n_events):
        total_tracks += len(tracks["majorityParticleId"][event])
    print("Average tracks per event:", total_tracks / n_events)


     # ---------- LOOP THROUGH EVENT ----------

    counter_total_particles_after_pt_if = 0
    counter_total_particles_before_pt_if = 0
    counter_total_particles_after_bin_index = 0
    counter_total_tracks_before_pt_if = 0
    counter_total_tracks_after_pt_if = 0

    for event in range(n_events):

        # true particles in this event
        particle_ids = particles["particle_id"][event].to_list()
        particle_pts = particles["pt"][event].to_list()

        track_truth = tracks["majorityParticleId"][event]

        # create a set of reconstructed particle IDs in this event

        track_ids = [tid[2] for tid in track_truth]

        counter_total_tracks_before_pt_if += len(track_ids)


        # ---------- DEBUG PRINTS ----------
        if event == 304:

            print("\n====================================================")
            print(f"EVENT {event}")
            print("====================================================")

            # ---------- PARTICLE TABLE ----------
            print("\nGenerated Particles:")
            print(f"{'Index':<6} {'ParticleID':<20} {'pT (GeV)':<10}")

            for i, (pid, pt) in enumerate(zip(particle_ids, particle_pts)):
                print(f"{i:<6} {pid:<20} {pt:<10.4f}")

            # ---------- TRACK TABLE ----------
            print("\nTracks (majorityParticleId):")
            print(f"{'TrackIndex':<10} {'MajorityParticleID':<20}")

            for i, tid in enumerate(track_ids):
                print(f"{i:<10} {tid:<20}")


            reconstructed_particles = set()

            for tid in track_ids:
                if tid != 0:
                    reconstructed_particles.add(tid)

            print("\nReconstructed particle IDs (excluding fakes):", reconstructed_particles)

            # ---------- MATCHING TABLE ----------
            print("\nMatching attempt:")
            print(f"{'ParticleID':<20} {'pT':<10} {'In track list?':<15}")

            for pid, pt in zip(particle_ids, particle_pts):

                if pt < 0.01:
                    status = "Skipped (low pT)"
                if pid in track_ids:
                    status = "MATCH"
                    track_ids.remove(pid)

                else:
                    status = "NO MATCH"

                print(f"{pid:<20} {pt:<10.4f} {status:<15}")

            print("\nTrack IDs before matching removal:")
            print(track_ids)
        
        # if event == 304:
        #     # print("Matched IDs 304:", track_truth[2])
        #     print("AFTER")
        #     print("IDs for event", 304, ":", track_ids)

        #     for ids in track_truth:
        #         print("Track majority raw:", ids)
        #     print("Truth IDs:", particle_ids)

        track_ids = [tid[2] for tid in track_truth  if tid[2] != 0]

        # ---------- LOOP OVER PARTICLES 2 ----------
        for pid, pt in zip(particle_ids, particle_pts):
            
            counter_total_particles_before_pt_if += 1
            
            #filter for paritcles under 0.01 GeV
            if pt < 0.5:
                continue

            #add eta value maybe

            counter_total_particles_after_pt_if += 1

            # find which bin this pt belongs to
            bin_index = np.digitize(pt, pt_bins) - 1
            #digitize is used to find the index of the bin that each pt value belongs to. 
            #It returns an array of indices corresponding to the bins defined by pt_bins. 
            #The -1 is used to convert from 1-based indexing (which digitize uses) to 0-based indexing (which Python uses).

            if 0 <= bin_index < len(hist_all):

                # count total particle
                counter_total_particles_after_bin_index += 1
                hist_all[bin_index] += 1

                # reconstructed_particles = set()

                # for tid in track_ids:
                #     if tid != 0:
                #         reconstructed_particles.add(tid)

                # # if pid is in track_ids, remove pid from track_ids and count as reconstructed
                # if pid in reconstructed_particles:
                #     hist_rec[bin_index] += 1
                #     reconstructed_particles.remove(pid)
                #     counter_total_tracks_after_pt_if += 1
                    # if event == 1:
                    #     print("Matched IDs 1:", pid)
                    #     print("IDs for event", 1, ":", track_ids)

                if pid in track_ids:
                    hist_rec[bin_index] += 1
                    track_ids.remove(pid)
                    counter_total_tracks_after_pt_if += 1
                    if event == 1:
                        print("Matched IDs 1:", pid)
                        print("IDs for event", 1, ":", track_ids)


    print("reconstructed particles set: ", reconstructed_particles)

    # ---------- Compute efficiency ----------
    efficiency = np.zeros_like(hist_all)

    for i in range(len(hist_all)):
        if hist_all[i] > 0:
            efficiency[i] = hist_rec[i] / hist_all[i]

    bin_centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])

    # ---------- Errors ----------

    errors = np.zeros_like(efficiency)

    for i in range(len(hist_all)):
        if hist_all[i] > 0:
            eff = efficiency[i]
            N = hist_all[i]
            errors[i] = np.sqrt(eff * (1 - eff) / N)


    # ---------- Debug ----------
    print("Total truth particles:", np.sum(hist_all))
    print("Total reconstructed truth particles:", np.sum(hist_rec))
    print("Global efficiency:", np.sum(hist_rec) / np.sum(hist_all))
    
    low_pt = 0
    high_pt = 0

    for event in range(n_events):
        for pt in particles["pt"][event]:
            if pt < 0.5:
                low_pt += 1
            else:
                high_pt += 1

    print("Below 0.5 GeV:", low_pt)
    print("Above 0.5 GeV:", high_pt)

    all_pts = []

    for event in range(n_events):
        all_pts.extend(particles["pt"][event].to_list())

    print("Minimum pT in dataset:", min(all_pts))

    # ---------- Plot efficiency ----------
    plt.figure()
    plt.plot(bin_centers, efficiency, "o-", color='blue')
    plt.errorbar(bin_centers, efficiency, yerr=errors, fmt='o', color='blue')
    #if i want to make the error bars blue
    plt.xlabel("pT [GeV]")
    plt.ylabel("True Efficiency")
    plt.ylim(0, 1.05)
    plt.grid()
    plt.legend()
    plt.savefig(output)
    #plt.show()

    print(f"Saved: {output}")
    plt.close()

    # ---------- pT distribution ----------

    all_pts = []

    for event in range(n_events):
        pts = particles["pt"][event].to_list()
        all_pts.extend(pts)

    plt.figure()
    plt.hist(all_pts, bins=50)
    plt.xlabel("pT")
    plt.ylabel("Truth particles")
    plt.savefig("truth_pt_distribution.png")
    plt.close()

def debug_0(
    particles_file,
    tracks_file,
    details_file,
    particle_tree="particles",
    track_tree="tracksummary",
    details_tree="matchingdetails",
    pt_bins=np.linspace(0.01, 10, 50),
    output="b_particle_efficiency_vs_pt.png",
):

    # ---------- READ PARTICLES ----------

    f_particles = uproot.open(particles_file)
    t_particles = f_particles[particle_tree]
    particles = t_particles.arrays(["particle", "pt", "particle_id"], library="ak")

    print("Nr of events:", len(particles["particle"]))

    # ---------- READ TRACKS ----------

    f_tracks = uproot.open(tracks_file)
    t_tracks = f_tracks[track_tree]
    tracks = t_tracks.arrays(["majorityParticleId", "trackClassification", "t_pT"], library="ak")

    f_details = uproot.open(details_file)
    t_details = f_details[details_tree]
    details = t_details.arrays(["event_nr", "particle_id", "matched", "track_pt"], library="ak")

    # ---------- Histogram ----------
    
    #makes all histograms zero
    hist_all = np.zeros(len(pt_bins) - 1)
    hist_rec = np.zeros(len(pt_bins) - 1)

    n_events = len(particles["particle"])

    total_tracks = 0
    for event in range(n_events):
        total_tracks += len(tracks["majorityParticleId"][event])
    print("Average tracks per event:", total_tracks / n_events)

    n_particle = len(details["particle_id"])

    print("\n n_particle", n_particle)

     # ---------- LOOP THROUGH EVENT ----------

    # print("with 0")
    # print("without 0")

    counter_particles = 0
    counter_tracks = 0

    counter_matched = 0
    counter_fake = 0

    for event in range(n_events):

        particle_ids = particles["particle"][event].to_list()
        particle_pts = particles["pt"][event].to_list()
        ids = particles["particle_id"][event].to_list()

        track_truth = tracks["majorityParticleId"][event].to_list()
        track_class = tracks["trackClassification"][event].to_list()
        track_pt = tracks["t_pT"][event].to_list()

        details_tpT = details["track_pt"][event]
        details_matched = details["matched"][event]
        details_particle_id = details["particle_id"][event]
        details_event_nr = details["event_nr"][event]

        for pid in zip(particle_ids):
            counter_particles += 1

        for tid in zip(track_truth):
            counter_tracks += 1
        
        if event == 1:
            print("\n==============================")
            print("EVENT", event)
            print("==============================")

            # ----- TRUE PARTICLES -----

            print("\nTRUE PARTICLES")
            print("particle:", particle_ids)
            print("particle_id:", ids)
            print("pt:", particle_pts)

            # ----- RECONSTRUCTED TRACKS -----

            print("\nTRACKS")
            print("majorityParticleId:", track_truth)
            print("trackClassification:", track_class)
            print("t_pT:", track_pt)

            # ----- MATCHING DETAILS -----

            print("\nMATCHING DETAILS")
            print("event_nr:", details_event_nr)
            print("particle_id:", details_particle_id)
            print("matched:", details_matched)
            print("track_pt:", details_tpT)

    print("\nTotal particles counted in loop:", counter_particles)
    print("Total tracks counted in loop:", counter_tracks)

    for p in range(n_particle):

        details_tpT = details["track_pt"][p]
        details_matched = details["matched"][p]
        details_particle_id = details["particle_id"][p]
        details_event_nr = details["event_nr"][p]

        if details_particle_id == 0:

            print("\n==============================")
            print("Particle", p)
            print("==============================")

            print("\nMATCHING DETAILS")
            print("event_nr:", details_event_nr)
            print("particle_id:", details_particle_id)
            print("matched:", details_matched)
            print("track_pt:", details_tpT)

        if details_matched == 1:
            counter_matched += 1
        else:
            counter_fake += 1

    print("\nTotal matched particles in details:", counter_matched)
    print("Total fake particles in details:", counter_fake)

    n_events = len(particles["particle"])
    matched_per_event = []

    for evt in range(n_events):
        true_ids = particles["particle_id"][evt].to_list()
        
        # get matching details for this event
        mask_event = details["event_nr"] == evt
        details_event_ids = details["particle_id"][mask_event].to_list()
        details_event_matched = details["matched"][mask_event].to_list()
        
        # check which true particles are matched
        matched_particles = 0
        for pid in true_ids:
            # find all rows in details for this particle in this event
            matches = [m for p, m in zip(details_event_ids, details_event_matched) if p == pid]
            if any(matches):
                matched_particles += 1
        
        eff_event = matched_particles / len(true_ids)
        matched_per_event.append(eff_event)

    # final particle-level efficiency = average over events
    particle_efficiency = sum(matched_per_event) / n_events
    print("Particle-level efficiency:", particle_efficiency)

    matched_particle_ids = set(details["particle_id"][details["matched"] == True])
    particle_efficiency = len(matched_particle_ids) / counter_particles
    print("Particle-level efficiency (set method):", particle_efficiency)

    #     good_track_ids = [
    #         tid[2]
    #         for tid, cls in zip(track_truth, track_class)
    #         if len(tid) > 2 and tid[2] != 0 and cls == 1
    #     ]

    #     # ---------- LOOP OVER PARTICLES 2 ----------
    #     for pid, pt in zip(particle_ids, particle_pts):

            
    #         #filter for paritcles under 0.5 GeV
    #         if pt <= 0.01:
    #             continue

    #         bin_index = np.digitize(pt, pt_bins) - 1

    #         if 0 <= bin_index < len(hist_all):
                
    #             hist_all[bin_index] += 1

    #             if pid in good_track_ids:
    #                 hist_rec[bin_index] += 1
    #                 good_track_ids.remove(pid)

    # # ---------- Compute efficiency ----------
    # efficiency = np.zeros_like(hist_all)

    # for i in range(len(hist_all)):
    #     if hist_all[i] > 0:
    #         efficiency[i] = hist_rec[i] / hist_all[i]

    # bin_centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])

    # # ---------- Errors ----------

    # errors = np.zeros_like(efficiency)

    # for i in range(len(hist_all)):
    #     if hist_all[i] > 0:
    #         eff = efficiency[i]
    #         N = hist_all[i]
    #         errors[i] = np.sqrt(eff * (1 - eff) / N)


    # # ---------- Debug ----------
    # print("Total truth particles:", np.sum(hist_all))
    # print("Total reconstructed truth particles:", np.sum(hist_rec))
    # print("Global efficiency:", np.sum(hist_rec) / np.sum(hist_all))
    
    # low_pt = 0
    # high_pt = 0

    # for event in range(n_events):
    #     for pt in particles["pt"][event]:
    #         if pt < 0.01:
    #             low_pt += 1
    #         else:
    #             high_pt += 1

    # print("Below 0.01 GeV:", low_pt)
    # print("Above 0.01 GeV:", high_pt)

    # all_pts = []

    # for event in range(n_events):
    #     all_pts.extend(particles["pt"][event].to_list())

    # print("Minimum pT in dataset:", min(all_pts))

    # # ---------- Plot efficiency ----------
    # plt.figure()
    # plt.plot(bin_centers, efficiency, "o-", color='blue')
    # plt.errorbar(bin_centers, efficiency, yerr=errors, fmt='o', color='blue')
    # #if i want to make the error bars blue
    # plt.xlabel("pT [GeV]")
    # plt.ylabel("True Efficiency")
    # plt.ylim(0, 1.05)
    # plt.grid()
    # plt.legend()
    # plt.savefig(output)
    # #plt.show()

    # print(f"Saved: {output}")
    # plt.close()

    # # ---------- pT distribution ----------

    # all_pts = []

    # for event in range(n_events):
    #     pts = particles["pt"][event].to_list()
    #     all_pts.extend(pts)

    # plt.figure()
    # plt.hist(all_pts, bins=50)
    # plt.xlabel("pT")
    # plt.ylabel("Truth particles")
    # plt.savefig("truth_pt_distribution.png")
    # plt.close()

def debug(
    particles_file,
    tracks_file,
    particle_tree="particles",
    track_tree="tracksummary",
    pt_bins=np.linspace(0.1, 0.5, 40),
    output="b_particle_efficiency_vs_pt.png",
):
    # ---------- READ PARTICLES ----------
    f_particles = uproot.open(particles_file)
    t_particles = f_particles[particle_tree]
    particles = t_particles.arrays(["particle", "pt", "particle_id"], library="ak")

    print("Nr of events:", len(particles["particle"]))

    # ---------- READ TRACKS ----------
    f_tracks = uproot.open(tracks_file)
    t_tracks = f_tracks[track_tree]
    tracks = t_tracks.arrays(["majorityParticleId", "trackClassification", "t_pT"], library="ak")

    # ---------- HISTOGRAMS ----------
    hist_all = np.zeros(len(pt_bins) - 1)   # total particles per pt bin
    hist_rec = np.zeros(len(pt_bins) - 1)   # matched particles per pt bin
    hist_fake = np.zeros(len(pt_bins) - 1)  # fake tracks per pt bin
    hist_total_tracks = np.zeros(len(pt_bins) - 1)  # total tracks per pt bin

    n_events = len(particles["particle"])
    total_tracks = sum(len(tracks["majorityParticleId"][event]) for event in range(n_events))
    print("Average tracks per event:", total_tracks / n_events)

    # ---------- LOOP THROUGH EVENTS ----------
    for event in range(n_events):
        particle_pts = particles["pt"][event].to_list()
        particle_ids = particles["particle_id"][event].to_list()

        track_truth = tracks["majorityParticleId"][event].to_list()
        track_class = tracks["trackClassification"][event].to_list()

        # ----- PARTICLE EFFICIENCY -----
        for pid, pt in zip(particle_ids, particle_pts):
            # find bin
            bin_idx = np.digitize(pt, pt_bins) - 1
            if bin_idx < 0 or bin_idx >= len(hist_all):
                continue  # skip out-of-range pts

            hist_all[bin_idx] += 1

            # is there a matched track?
            # matched = any((t == pid) and (cls == 1) for t, cls in zip(track_truth, track_class))
            # if matched:
            #     hist_rec[bin_idx] += 1

            # if pid in track_truth, then hist_rec[bin_idx] += 1
            for t, cls in zip(track_truth, track_class):
                if t == pid and cls == 1:
                    hist_rec[bin_idx] += 1
                    break  # stop after first match

        # ----- FAKE TRACKS -----
        for t, cls in zip(track_truth, track_class):
            if cls == 0:  # fake
                # assign to bin of nearest particle pt or simple approach: bin by t_pT if available, else skip
                # since t_pT is NaN for fakes, we can skip NaN or count them in a separate bin
                pass  # For simplicity, we will compute **overall fake fraction later**

            # count total tracks per bin using **true particle they are matched to**
            if t != 0:
                try:
                    idx = particle_ids.index(t)
                    pt = particle_pts[idx]
                    bin_idx = np.digitize(pt, pt_bins) - 1
                    if 0 <= bin_idx < len(hist_total_tracks):
                        hist_total_tracks[bin_idx] += 1
                except ValueError:
                    pass  # unmatched track, ignore for total_tracks histogram

   # ---------- Compute efficiency ----------
    efficiency = np.zeros_like(hist_all)

    for i in range(len(hist_all)):
        if hist_all[i] > 0:
            efficiency[i] = hist_rec[i] / hist_all[i]

    bin_centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])

    # ---------- PLOT RESULTS ----------
    # bin_centers = 0.5 * (pt_bins[1:] + pt_bins[:-1])

    # plt.figure(figsize=(8,6))
    # plt.step(bin_centers, efficiency, where='mid', label="Particle efficiency")
    # plt.step(bin_centers, fake_rate, where='mid', label="Fake rate")
    # plt.xlabel("pT [GeV]")
    # plt.ylabel("Fraction")
    # plt.title("Particle efficiency and Fake rate vs pT")
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(output)
    # # plt.show()

    # print("Done. Plot saved to:", output)

    # ---------- Plot efficiency ----------
    plt.figure()
    plt.plot(bin_centers, efficiency, "o-", color='blue')
    # plt.errorbar(bin_centers, efficiency, yerr=errors, fmt='o', color='blue')
    #if i want to make the error bars blue
    plt.xlabel("pT [GeV]")
    plt.ylabel("True Efficiency")
    plt.ylim(0, 1.05)
    plt.grid()
    plt.legend()
    plt.savefig(output)
    #plt.show()

    print(f"Saved: {output}")
    plt.close()

# Efficiency with tracks (nMatchedTracks/ nAllTracks) = 0.999794
# asta e buna pastreaza
# calculates efficiency right but not fake track - uses majority particle iD
def track_eff_and_fake_vs_pt(
    tracks_file,
    track_tree="tracksummary",
    pt_bins=np.linspace(0.01, 10, 50),
    output_purity="track_efficiency_vs_pt.png",
    output_fake_track="fake_track_vs_pt.png",
):

    # ---------- READ TRACKS ----------

    f_tracks = uproot.open(tracks_file)
    t_tracks = f_tracks[track_tree]

    #also load track pt to plot against
    #tracks = t_tracks.arrays(["majorityParticleId", "t_pT"], library="ak" )
    tracks = t_tracks.arrays(["majorityParticleId", "t_pT", "trackClassification"], library="ak")

    # ---------- Histogram ----------
    
    #makes all histograms zero
    hist_all = np.zeros(len(pt_bins) - 1)
    hist_good = np.zeros(len(pt_bins) - 1)
    hist_fake = np.zeros(len(pt_bins) - 1)

    n_events = len(tracks["majorityParticleId"])


     # ---------- LOOP THROUGH EVENTS ----------

    countfake = 0
    countgood = 0
    counttotal = 0

    for event in range(n_events):

        track_truth = tracks["majorityParticleId"][event]
        track_pts = tracks["t_pT"][event]
        track_class = tracks["trackClassification"][event]

        # trackClassification for this track
        # classification = tracks["trackClassification"][event][i]

        # decide if it's "good"
        # is_good = (classification == 1)  # or whatever code corresponds to "truth-matched"

        #loop through tracks - using ids[2] as the majority id, if its 0 then its fake, if its not 0 then its good, and we count how many of each we have in each pt bin
        # for ids, pt in zip(track_truth, track_pts):

        #     # get majority truth id (3rd element)
        #     if len(ids) > 2:
        #         majority_id = ids[2]
        #     else:
        #         continue

        #     # find which bin this pt belongs to
        #     bin_index = np.digitize(pt, pt_bins) - 1
        #     #digitize is used to find the index of the bin that each pt value belongs to. 

        #     if 0 <= bin_index < len(hist_all):

        #         # count total tracks
        #         hist_all[bin_index] += 1

        #         # count good tracks (majority id != 0)
        #         if majority_id != 0:
        #             hist_good[bin_index] += 1

        #loops through tracks using ids[0]
        # for ids, pt in zip(track_truth, track_pts):
        #     if len(ids) < 3:
        #         continue  # skip broken entries

        #     #check if its fake
        #     is_fake = ids[0] == 0 or all(x == 0 for x in ids[2:])
            
        #     # find which pT bin this track belongs to
        #     bin_index = np.digitize(pt, pt_bins) - 1

        #     if 0 <= bin_index < len(hist_all):
        #         # count total tracks in this bin
        #         hist_all[bin_index] += 1

        #         # count good tracks in this bin
        #         if not is_fake:
        #             hist_good[bin_index] += 1
        #         else:
        #             hist_fake[bin_index] += 1

        #loops through tracks using trackClassification to decide if its good or fake
        for ids, pt, classification in zip(track_truth, track_pts, track_class):
            # if len(ids) < 3:
            #     continue  # skip broken entries

            if pt < 0.01:
                continue

            # print("Track classification:", classification)

            # 1 = truth-matched, 2 = fake track

            bin_index = np.digitize(pt, pt_bins) - 1

            # if ids[2] == 0:
            #     print("Fake track pT:", pt)

            # if ids[2] != 0:
            #     print("Good track pt:", pt)

            counttotal += 1

            #problem is here - cred ca trece cumva peste alea care nu sunt bune or smth
            if not (0 <= bin_index < len(hist_all)):
                continue

            hist_all[bin_index] += 1
            if ids[2] != 0:   # truth-matched track
                hist_good[bin_index] += 1
                countgood += 1
            else: # fake track
                hist_fake[bin_index] += 1
                countfake += 1

        # for ids, pts, classes in zip(track_truth, track_pts, track_class):
        #     if len(ids) < 3:
        #         continue  # skip broken entries

        #     for id_, pt, cl in zip(ids, pts, classes):
        #         bin_index = np.digitize(pt, pt_bins) - 1
        #         if not (0 <= bin_index < len(hist_all)):
        #             continue

        #         hist_all[bin_index] += 1
        #         if cl == 1:   # truth-matched track
        #             hist_good[bin_index] += 1
        #         else:
        #             hist_fake[bin_index] += 1

    # ---------- Compute purity ----------
    purity = np.zeros_like(hist_all)

    # print("Track efficiency:", np.sum(hist_good) / np.sum(hist_all))

    for i in range(len(hist_all)):
        if hist_all[i] > 0:
            purity[i] = hist_good[i] / hist_all[i]

    bin_centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])

    # ---------- Errors ----------

    errors_purity = np.zeros_like(purity)

    for i in range(len(hist_all)):
        if hist_all[i] > 0:
            p = purity[i]
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

    # ----- DEBUG ----

    #see how many tracks are in each bin and how many are good
    # for i in range(len(hist_all)):
    #     print(f"Bin {i}: total={hist_all[i]}, good={hist_good[i]}")

    fake_count = 0
    total_count = 0
    good_count = 0

    for event in range(n_events):
        for ids in tracks["majorityParticleId"][event]:
            if len(ids) > 2:
                total_count += 1
                if ids[2] == 0:
                    fake_count += 1
                else:
                    good_count += 1

    #for event in range(5):
    #     for ids in tracks["majorityParticleId"][event]:
    #         print(ids)

    print("For loop")
    print("Total tracks:", total_count)
    print("Fake tracks:", fake_count) 
    print("good tracks:", good_count)
    print("Fake Rate - ids[2]:", fake_count/total_count)
    print("Track Efficiency - ids[2]:", good_count/total_count)

    print("\nHistogram counters:")
    print("Total tracks:", counttotal)
    print("Fake tracks:", countfake)
    print("Matched tracks:", countgood)
    print("Fake Rate:", countfake/counttotal)
    print("Track Efficiency:", countgood/counttotal)

    print("\nHistograms:")
    print("Total tracks in histogram:", np.sum(hist_all))
    print("Track Efficiency - histogram:", np.sum(hist_good) / np.sum(hist_all))
    print("Fake rate - histogram:", np.sum(hist_fake) / np.sum(hist_all))

    # for event in range(n_events):
    #     for ids, cl in zip(tracks["majorityParticleId"][event],
    #                     tracks["trackClassification"][event]):

    #         if len(ids) > 2:
    #             majority_id = ids[2]

    #             if majority_id == 0 and cl != 2:
    #                 print("Majority=0 but classification=", cl)

    #             if majority_id != 0 and cl == 2:
    #                 print("Has majority but classified fake")

    # ---------- Plot purity ----------
    plt.figure()
    plt.plot(bin_centers, purity, "o-", color='blue')
    plt.errorbar(bin_centers, purity, yerr=errors_purity, fmt='o', color='blue')
    #if i want to make the error bars blue
    plt.xlabel("pT [GeV]")
    plt.ylabel("Track Efficiency")
    plt.ylim(0, 1.05)
    plt.grid()
    plt.legend()
    plt.savefig(output_purity)
    #plt.show()

    print(f"Saved: {output_purity}")
    plt.close()

    # ---------- Plot fake ----------
    plt.figure()
    plt.plot(bin_centers, fakeTrack, "o-", color='blue')
    plt.errorbar(bin_centers, fakeTrack, yerr=errors_fake, fmt='o', color='blue')
    #if i want to make the error bars blue
    plt.xlabel("pT [GeV]")
    plt.ylabel("fake track")
    plt.ylim(0, 0.25)
    plt.grid()
    plt.legend()
    plt.savefig(output_fake_track)
    #plt.show()

    print(f"Saved: {output_fake_track}")
    plt.close()


def track_metrics_classification_without_generated(
    tracks_file,
    track_tree="tracksummary",
    pt_bins=np.linspace(0.01, 10, 50),
    output_purity="a_track_efficiency_vs_pt.png",
    output_fake_track="a_fake_track_vs_pt.png",
    output_duplicate_track="a_duplicate_ratio_vs_pt.png",
    output_matched_track="a_matched_efficiency_vs_pt.png",
):

    # ---------- READ TRACKS ----------

    f_tracks = uproot.open(tracks_file)
    t_tracks = f_tracks[track_tree]

    #also load track pt to plot against
    #tracks = t_tracks.arrays(["majorityParticleId", "t_pT"], library="ak" )
    tracks = t_tracks.arrays(["majorityParticleId", "t_pT", "trackClassification"], library="ak")

    # ---------- Histogram ----------
    
    #makes all histograms zero
    hist_all = np.zeros(len(pt_bins) - 1)
    hist_good = np.zeros(len(pt_bins) - 1)
    hist_fake = np.zeros(len(pt_bins) - 1)
    hist_duplicate = np.zeros(len(pt_bins) - 1)
    hist_matched = np.zeros(len(pt_bins) - 1)

    n_events = len(tracks["majorityParticleId"])


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

        for ids, pt, classification in zip(track_truth, track_pts, track_class):
            if len(ids) < 3:
                continue  # skip broken entries

            if pt < 0.01:
                continue

            counttotal += 1

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
    print("Track Efficiency:", countgood/counttotal)
    print("Fake Rate:", countfake/counttotal)
    print("Duplicate Rate:", countduplicate/counttotal)

    print("\nHistograms:")
    print("Total tracks in histogram:", np.sum(hist_all))
    print("Fake tracks:", np.sum(hist_fake))
    print("Matched tracks:", np.sum(hist_good))
    print("Duplicate tracks:", np.sum(hist_duplicate))
    print("Track Efficiency - histogram:", np.sum(hist_good) / np.sum(hist_all))
    print("Fake rate - histogram:", np.sum(hist_fake) / np.sum(hist_all))
    print("Duplicate Rate - histogram:", np.sum(hist_duplicate) / np.sum(hist_all))

    print("\n")

    # ---------- Plot efficiency ----------
    plt.figure()
    plt.plot(bin_centers, efficiency, "o-", color='blue')
    plt.errorbar(bin_centers, efficiency, yerr=errors_purity, fmt='o', color='blue')
    #if i want to make the error bars blue
    plt.xlabel("pT [GeV]")
    plt.ylabel("Track Efficiency")
    plt.ylim(0, 1.05)
    plt.grid()
    # plt.legend()
    plt.savefig(output_purity)
    #plt.show()

    # print(f"Saved: {output_purity}")
    plt.close()

    # ---------- Plot fake ----------
    plt.figure()
    plt.plot(bin_centers, fakeTrack, "o-", color='green')
    plt.errorbar(bin_centers, fakeTrack, yerr=errors_fake, fmt='o', color='green')
    #if i want to make the error bars blue
    plt.xlabel("pT [GeV]")
    plt.ylabel("fake track")
    plt.ylim(0, 0.25)
    plt.grid()
    # plt.legend()
    plt.savefig(output_fake_track)
    #plt.show()

    # print(f"Saved: {output_fake_track}")
    plt.close()

    # ---------- Plot duplicate ----------
    plt.figure()
    plt.plot(bin_centers, duplicateRate, "o-", color='orange')
    plt.errorbar(bin_centers, duplicateRate, yerr=errors_duplicate, fmt='o', color='orange')
    plt.xlabel("pT [GeV]")
    plt.ylabel("duplicate ration")
    plt.ylim(0, 0.25)
    plt.grid()
    # plt.legend()
    plt.savefig(output_duplicate_track)
    #plt.show()

    # print(f"Saved: {output_duplicate_track}")
    plt.close()

    # ---------- Plot true matched ----------
    plt.figure()
    plt.plot(bin_centers, matchedEfficiency, "o-", color='red')
    plt.errorbar(bin_centers, matchedEfficiency, yerr=errors_matched, fmt='o', color='red')
    plt.xlabel("pT [GeV]")
    plt.ylabel("true matched efficiency")
    plt.ylim(0, 1.05)
    plt.grid()
    # plt.legend()
    plt.savefig(output_matched_track)
    # #plt.show()

    # print(f"Saved: {output_matched_track}")
    plt.close()



# Purity = correctly reconstructed tracks / total reconstructed tracks
# fake fraction = tells you the fraction of reconstructed tracks that are fake
# here i was calculating it with trackClassification - not the way to go
def purity_and_fake_vs_pt(
    tracks_file,
    track_tree="tracksummary",
    pt_bins=np.linspace(0.5, 10, 50),
    output_purity="maybe_purity_vs_pt.png",
    output_fake_track="maybe_fake_track_vs_pt.png",
):

    # ---------- READ TRACKS ----------

    f_tracks = uproot.open(tracks_file)
    t_tracks = f_tracks[track_tree]

    #also load track pt to plot against
    #tracks = t_tracks.arrays(["majorityParticleId", "t_pT"], library="ak" )
    tracks = t_tracks.arrays(["majorityParticleId", "t_pT", "trackClassification"], library="ak")

    # ---------- Histogram ----------
    
    #makes all histograms zero
    hist_all = np.zeros(len(pt_bins) - 1)
    hist_good = np.zeros(len(pt_bins) - 1)
    hist_fake = np.zeros(len(pt_bins) - 1)

    n_events = len(tracks["majorityParticleId"])


     # ---------- LOOP THROUGH EVENTS ----------

    countfake = 0
    count_trei = 0
    count_zero = 0
    count_unu = 0

    for event in range(n_events):

        track_truth = tracks["majorityParticleId"][event]
        track_pts = tracks["t_pT"][event]
        track_class = tracks["trackClassification"][event]

        # trackClassification for this track
        # classification = tracks["trackClassification"][event][i]

        # decide if it's "good"
        # is_good = (classification == 1)  # or whatever code corresponds to "truth-matched"

        #loop through tracks - using ids[2] as the majority id, if its 0 then its fake, if its not 0 then its good, and we count how many of each we have in each pt bin
        # for ids, pt in zip(track_truth, track_pts):

        #     # get majority truth id (3rd element)
        #     if len(ids) > 2:
        #         majority_id = ids[2]
        #     else:
        #         continue

        #     # find which bin this pt belongs to
        #     bin_index = np.digitize(pt, pt_bins) - 1
        #     #digitize is used to find the index of the bin that each pt value belongs to. 

        #     if 0 <= bin_index < len(hist_all):

        #         # count total tracks
        #         hist_all[bin_index] += 1

        #         # count good tracks (majority id != 0)
        #         if majority_id != 0:
        #             hist_good[bin_index] += 1

        #loops through tracks using ids[0]
        # for ids, pt in zip(track_truth, track_pts):
        #     if len(ids) < 3:
        #         continue  # skip broken entries

        #     #check if its fake
        #     is_fake = ids[0] == 0 or all(x == 0 for x in ids[2:])
            
        #     # find which pT bin this track belongs to
        #     bin_index = np.digitize(pt, pt_bins) - 1

        #     if 0 <= bin_index < len(hist_all):
        #         # count total tracks in this bin
        #         hist_all[bin_index] += 1

        #         # count good tracks in this bin
        #         if not is_fake:
        #             hist_good[bin_index] += 1
        #         else:
        #             hist_fake[bin_index] += 1

        #loops through tracks using trackClassification to decide if its good or fake
        for ids, pt, classification in zip(track_truth, track_pts, track_class):
            if len(ids) < 3:
                continue  # skip broken entries

            if pt <= 0.5:
                continue

            print("Track classification:", classification)

            # 1 = truth-matched, 2 = fake track

            bin_index = np.digitize(pt, pt_bins) - 1
            if not (0 <= bin_index < len(hist_all)):
                continue

            hist_all[bin_index] += 1
            if classification == 1:   # truth-matched track
                hist_good[bin_index] += 1
                count_unu += 1
            elif classification == 2: # fake track
                hist_fake[bin_index] += 1
                countfake += 1
            elif classification == 3: # test
                count_trei += 1
            elif classification == 0:
                count_zero += 1

        # for ids, pts, classes in zip(track_truth, track_pts, track_class):
        #     if len(ids) < 3:
        #         continue  # skip broken entries

        #     for id_, pt, cl in zip(ids, pts, classes):
        #         bin_index = np.digitize(pt, pt_bins) - 1
        #         if not (0 <= bin_index < len(hist_all)):
        #             continue

        #         hist_all[bin_index] += 1
        #         if cl == 1:   # truth-matched track
        #             hist_good[bin_index] += 1
        #         else:
        #             hist_fake[bin_index] += 1

    # ---------- Compute purity ----------
    purity = np.zeros_like(hist_all)

    print("Track efficiency:", np.sum(hist_good) / np.sum(hist_all))

    for i in range(len(hist_all)):
        if hist_all[i] > 0:
            purity[i] = hist_good[i] / hist_all[i]

    bin_centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])

    # ---------- Errors ----------

    errors_purity = np.zeros_like(purity)

    for i in range(len(hist_all)):
        if hist_all[i] > 0:
            p = purity[i]
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

    # ----- DEBUG ----

    #see how many tracks are in each bin and how many are good
    # for i in range(len(hist_all)):
    #     print(f"Bin {i}: total={hist_all[i]}, good={hist_good[i]}")

    fake_count = 0
    total_count = 0

    for event in range(n_events):
        for ids in tracks["majorityParticleId"][event]:
            if len(ids) > 2:
                total_count += 1
                if ids[2] == 0:
                    fake_count += 1

    #for event in range(5):
    #     for ids in tracks["majorityParticleId"][event]:
    #         print(ids)

    print("Total tracks:", total_count)
    print("Fake tracks - with ids[2] == 0 - matches ckf:", fake_count) 
    print("Fake tracks 2 with trackClassification == 2 - matches my result:", countfake)
    print("Track Classification == 3 (test):", count_trei)
    print("Track Classification == 0:", count_zero)
    print("Track Classification == 1 (truth-matched):", count_unu)

    print("Track Efficiency :", np.sum(hist_good) / np.sum(hist_all))
    print("Track Efficiency with ids[2]:", 1 - fake_count/total_count)
    print("Fake Rate - ids[2]:", fake_count/total_count)
    print("Fake Rate - trackclass:", countfake/total_count)
    print("Global fake:", np.sum(hist_fake) / np.sum(hist_all))

    # for event in range(n_events):
    #     for ids, cl in zip(tracks["majorityParticleId"][event],
    #                     tracks["trackClassification"][event]):

    #         if len(ids) > 2:
    #             majority_id = ids[2]

    #             if majority_id == 0 and cl != 2:
    #                 print("Majority=0 but classification=", cl)

    #             if majority_id != 0 and cl == 2:
    #                 print("Has majority but classified fake")

    # ---------- Plot purity ----------
    plt.figure()
    plt.plot(bin_centers, purity, "o-", color='blue')
    plt.errorbar(bin_centers, purity, yerr=errors_purity, fmt='o', color='blue')
    #if i want to make the error bars blue
    plt.xlabel("pT [GeV]")
    plt.ylabel("Purity")
    plt.ylim(0, 1.05)
    plt.grid()
    plt.legend()
    plt.savefig(output_purity)
    #plt.show()

    print(f"Saved: {output_purity}")
    plt.close()

    # ---------- Plot fake ----------
    plt.figure()
    plt.plot(bin_centers, fakeTrack, "o-", color='blue')
    plt.errorbar(bin_centers, fakeTrack, yerr=errors_fake, fmt='o', color='blue')
    #if i want to make the error bars blue
    plt.xlabel("pT [GeV]")
    plt.ylabel("fake track")
    plt.ylim(0, 0.25)
    plt.grid()
    plt.legend()
    plt.savefig(output_fake_track)
    #plt.show()

    print(f"Saved: {output_fake_track}")
    plt.close()

def duplicate_rate_vs_pt(
    tracks_file,
    track_tree="tracksummary",
    pt_bins=np.linspace(0, 10, 50),
    output="maybe_duplicate_vs_pt.png",
):

    # ---------- READ TRACKS ----------

    f_tracks = uproot.open(tracks_file)
    t_tracks = f_tracks[track_tree]

    #also load track pt to plot against
    tracks = t_tracks.arrays(["majorityParticleId", "t_pT"], library="ak" )
    #tracks = t_tracks.arrays(["majorityParticleId", "t_pT", "trackClassification"], library="ak")


    # ---------- Histogram ----------
    

    #makes all histograms zero
    hist_all = np.zeros(len(pt_bins) - 1)
    hist_dup = np.zeros(len(pt_bins) - 1)

    n_events = len(tracks["majorityParticleId"])


     # ---------- LOOP THROUGH EVENTS ----------


    # for event in range(n_events):

    #     track_truth = tracks["majorityParticleId"][event]
    #     track_pts = tracks["t_pT"][event]

    #     #loop through tracks - using ids[2] as the majority id, if its 0 then its fake, if its not 0 then its good, and we count how many of each we have in each pt bin
    #     for ids, pt in zip(track_truth, track_pts):

    #         # get majority truth id (3rd element)
    #         if len(ids) > 2:
    #             majority_id = ids[2]
    #         else:
    #             continue

    #         # find which bin this pt belongs to
    #         bin_index = np.digitize(pt, pt_bins) - 1
    #         #digitize is used to find the index of the bin that each pt value belongs to. 

    #         if 0 <= bin_index < len(hist_all):

    #             # count total tracks
    #             hist_all[bin_index] += 1

    #             # counts duplicates in this bin (majority id != 0 and appears more than once)
    #             if majority_id != 0 and any(ids[2] == other_ids[2] for other_ids in track_truth if other_ids is not ids):
    #                 hist_dup[bin_index] += 1


    for event in range(n_events):
        majority_ids = [ids[2] for ids in tracks["majorityParticleId"][event] if len(ids) > 2 and ids[2] != 0]
        track_pts = tracks["t_pT"][event]

        # count duplicates: ids that appear more than once
        dup_counts = {tid: majority_ids.count(tid) for tid in set(majority_ids) if majority_ids.count(tid) > 1}

        for ids, pt in zip(tracks["majorityParticleId"][event], track_pts):
            if len(ids) <= 2:
                continue
            majority_id = ids[2]
            if majority_id == 0:
                continue

            bin_index = np.digitize(pt, pt_bins) - 1
            if 0 <= bin_index < len(hist_all):
                hist_all[bin_index] += 1
                if majority_id in dup_counts:
                    hist_dup[bin_index] += 1

    # ---------- Compute purity ----------
    duplicate_rate = np.zeros_like(hist_all)

    for i in range(len(hist_all)):
        if hist_all[i] > 0:
            duplicate_rate[i] = hist_dup[i] / hist_all[i]

    bin_centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])

    # ---------- Errors ----------

    errors = np.zeros_like(duplicate_rate)

    for i in range(len(hist_all)):
        if hist_all[i] > 0:
            d = duplicate_rate[i]
            N = hist_all[i]
            errors[i] = np.sqrt(d * (1 - d) / N)

    print("Duplicate ratio:", np.sum(hist_dup) / np.sum(hist_all))

    # ---------- Plot purity ----------
    plt.figure()
    plt.plot(bin_centers, duplicate_rate, "o-", color='blue')
    plt.errorbar(bin_centers, duplicate_rate, yerr=errors, fmt='o', color='blue')
    #if i want to make the error bars blue
    plt.xlabel("pT [GeV]")
    plt.ylabel("Purity")
    plt.ylim(0, 0.25)
    plt.grid()
    plt.legend()
    plt.savefig(output)
    #plt.show()

    print(f"Saved: {output}")
    plt.close()

# TO DO: Reconstructed pT distribution

# TO DO: Efficiency vs η (pseudo-rapidity)
# TO DO: Fake rate vs η
def track_eff_and_fake_vs_eta(
    tracks_file,
    track_tree="tracksummary",
    eta_bins=np.linspace(-2, 2, 30),
    output_purity="track_efficiency_vs_eta.png",
    output_fake_track="fake_track_vs_eta.png",
):

    # ---------- READ TRACKS ----------

    f_tracks = uproot.open(tracks_file)
    t_tracks = f_tracks[track_tree]
    tracks = t_tracks.arrays(["majorityParticleId", "t_eta"], library="ak" )

    # tracks = t_tracks.arrays(["majorityParticleId", "t_pT", "trackClassification"], library="ak")

    # ---------- Histogram ----------
    
    #makes all histograms zero
    hist_all = np.zeros(len(eta_bins) - 1)
    hist_good = np.zeros(len(eta_bins) - 1)
    hist_fake = np.zeros(len(eta_bins) - 1)

    n_events = len(tracks["majorityParticleId"])


     # ---------- LOOP THROUGH EVENTS ----------

    countfake = 0
    countgood = 0
    totalcount = 0

    for event in range(n_events):

        track_truth = tracks["majorityParticleId"][event]
        track_etas = tracks["t_eta"][event]

        #loops through tracks using trackClassification to decide if its good or fake
        for ids, eta in zip(track_truth, track_etas):
            # if len(ids) < 3:
            #     continue  # skip broken entries

            if len(ids) <= 2:
                continue

            # if ids[2] == 0:
            #     print("Fake track eta:", eta)

            # if ids[2] != 0:
            #     print("Good track eta:", eta)

            totalcount += 1

            bin_index = np.digitize(eta, eta_bins) - 1
            if not (0 <= bin_index < len(hist_all)):
                continue

            # try to bin based on generated eta?
            hist_all[bin_index] += 1

            if ids[2] != 0:   # truth-matched track
                hist_good[bin_index] += 1
                countgood += 1
            else: # fake track
                hist_fake[bin_index] += 1
                countfake += 1

    # ---------- Compute track efficiency ----------
    purity = np.zeros_like(hist_all)

    # print("Track efficiency:", np.sum(hist_good) / np.sum(hist_all))

    for i in range(len(hist_all)):
        if hist_all[i] > 0:
            purity[i] = hist_good[i] / hist_all[i]

    bin_centers = 0.5 * (eta_bins[:-1] + eta_bins[1:])

    # ---------- Errors ----------

    errors_purity = np.zeros_like(purity)

    for i in range(len(hist_all)):
        if hist_all[i] > 0:
            p = purity[i]
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

    # ----- DEBUG ----

    fake_count = 0
    total_count = 0
    good_count = 0

    for event in range(n_events):
        for ids in tracks["majorityParticleId"][event]:
            if len(ids) > 2:
                total_count += 1
                if ids[2] == 0:
                    fake_count += 1
                else:
                    good_count += 1


    print("Total tracks:", total_count)
    print("Fake tracks - with ids[2] == 0 - matches ckf:", fake_count) 
    print("Matched tracks: ids[2] != 0, in the histogram if:", countgood)
    print("good tracks:", good_count)

    print("---","\n")

    print("countgood:", countgood)
    print("totalcount:", totalcount)
    print("countfake:", countfake)
    print("Track Efficiency - using counts, :", countgood / totalcount)

    print("---\n")

    print("Track Efficiency - using the histogra, :", np.sum(hist_good) / np.sum(hist_all))
    print("1 Track Efficiency - using the histogra, :", countgood / total_count)
    print("Track Efficiency with ids[2]:", 1 - fake_count/total_count)
    print("Fake Rate - ids[2]:", fake_count/total_count)
    print("Fake Rate - trackclass:", countfake/total_count)
    print("Global fake - using hist:", np.sum(hist_fake) / np.sum(hist_all))

    # ---------- Plot efficiency ----------
    plt.figure()
    plt.plot(bin_centers, purity, "o-", color='blue')
    plt.errorbar(bin_centers, purity, yerr=errors_purity, fmt='o', color='blue')
    #if i want to make the error bars blue
    plt.xlabel("eta")
    plt.ylabel("Track Efficiency")
    plt.ylim(0, 1.05)
    plt.grid()
    plt.legend()
    plt.savefig(output_purity)
    #plt.show()

    print(f"Saved: {output_purity}")
    plt.close()

    # ---------- Plot fake ----------
    plt.figure()
    plt.plot(bin_centers, fakeTrack, "o-", color='blue')
    plt.errorbar(bin_centers, fakeTrack, yerr=errors_fake, fmt='o', color='blue')
    #if i want to make the error bars blue
    plt.xlabel("eta")
    plt.ylabel("Fake Rate")
    plt.ylim(0, 0.25)
    plt.grid()
    plt.legend()
    plt.savefig(output_fake_track)
    #plt.show()

    print(f"Saved: {output_fake_track}")
    plt.close()

# to do - particle interation on
# pt to 10 mev to 10 gev

if __name__ == "__main__":

    # true_efficiency_vs_pt_3(
    #     particles_file="particles.root",
    #     tracks_file="tracksummary_ckf.root",
    # )

    debug_0(
        particles_file="particles.root",
        tracks_file="tracksummary_ckf.root",
        details_file="performance_finding_ckf_matchingdetails.root",
    )

    # purity_and_fake_vs_pt(
    #     tracks_file="tracksummary_ckf.root",
    # )

    # track_eff_and_fake_vs_pt(
    #     tracks_file="tracksummary_ckf.root",
    # )

    # track_metrics_classification_without_generated(
    #     tracks_file="tracksummary_ckf.root",
    # )

    # track_eff_and_fake_vs_eta(
    #     tracks_file="tracksummary_ckf.root",
    # )

    # duplicate_rate_vs_pt(
    #     tracks_file="tracksummary_ckf.root",
    # )