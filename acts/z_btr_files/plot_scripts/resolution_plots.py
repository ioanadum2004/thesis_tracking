import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import os

#to do: make function to retireve files - pt and eta and ids

#to do: make funtion to compute resoultion 

#or take pt,p, theta as inputs and compute the residuals and resolution in the same function. - like have only one overall

def resolution_plot_pT(
    particles_file,
    tracks_file,
    particle_tree="particles",
    track_tree="tracksummary",
    pt_bins=np.linspace(0.1, 0.5, 30),
    eta_bins=np.linspace(-0.14, 0.14, 30),
    #pt_bins=np.linspace(0, 10, 50),
    output="pT_summary_resolution.png",
    output_rms="pT_resolution_plot_heatmap.png",
    output_mean="pT_mean_plot_heatmap.png",
    output_pT_rms="pT_resolution_plot_vs_pt.png",
    output_eta_rms="pT_resolution_plot_vs_eta.png",
    output_pT_mean="pT_resolution_mean_plot_vs_pt.png",
    output_eta_mean="pT_resolution_mean_plot_vs_eta.png",
    # output_dir = os.path.join("particle_tracking_btr", "resolution_plots"),
    # output = os.path.join(output_dir, "pT_resolution_plot_heatmap.png"),
    # output_pT = os.path.join(output_dir, "pT_resolution_plot_vs_pt.png"),
    # output_eta = os.path.join(output_dir, "pT_resolution_plot_vs_eta.png"),
):

    # ---------- Create output directory ----------

    # os.makedirs(output_dir, exist_ok=True)

    # output = os.path.join(output_dir, output)
    # output_pT = os.path.join(output_dir, output_pT)
    # output_eta = os.path.join(output_dir, output_eta)

    output_dir = os.path.join("z_btr_files", "resolution_plots")
    os.makedirs(output_dir, exist_ok=True)

    # ---------- READ PARTICLES ----------

    f_particles = uproot.open(particles_file)
    t_particles = f_particles[particle_tree]
    particles = t_particles.arrays(["particle_id", "pt", "eta"], library="ak")

    print("nr of events:", len(particles["particle_id"]))

    # n_events is the nr of events in the dataset
    n_events = len(particles["particle_id"])

    # ---------- READ TRACKS ----------

    f_tracks = uproot.open(tracks_file)
    t_tracks = f_tracks[track_tree]
    tracks = t_tracks.arrays(["majorityParticleId", "t_pT", "t_eta"], library="ak")

    # ---------- Bins ----------

    N_pt  = len(pt_bins) - 1
    N_eta = len(eta_bins) - 1

    # 2D list of lists for residuals
    residual_pool = [[[] for _ in range(N_eta)] for _ in range(N_pt)]

    # 1D list for residuals pT
    residual_pool_pT = [[] for _ in range(N_pt)]

    # 1D list for residuals eta
    residual_pool_eta = [[] for _ in range(N_eta)]

     # ---------- LOOP THROUGH EVENT ----------


    for event in range(n_events):

        # true particles in this event
        particle_ids = particles["particle_id"][event].to_list()
        particle_pts = particles["pt"][event].to_list()
        particle_etas = particles["eta"][event].to_list()

        #track data
        track_truth = tracks["majorityParticleId"][event]
        track_pts = tracks["t_pT"][event].to_list()

        # ---------- LOOP OVER TRACKS ----------

        #looping through tracks and matching to particles
        for ids, pt_r in zip(track_truth, track_pts):

            # skip broken truth info
            # if len(ids) < 3:
            #     continue

            # majority_id = ids[2]   # majorityParticleId
            majority_id = ids   # majorityParticleId

            # print(f"Track majority ID: {majority_id}, pt_r: {pt_r}")

            # skip fake tracks
            if majority_id == 0:
                continue

            if majority_id not in particle_ids:
                continue

            # find the index of the matched particle
            index = particle_ids.index(majority_id)

            pt_g  = particle_pts[index]
            eta_g = particle_etas[index]

            # print(f"Matched particle ID: {majority_id}, pt_g: {pt_g}, eta_g: {eta_g}")

            if pt_g <= 0.01:
                continue

            residual = (pt_g - pt_r) / pt_g

            bin_index_pT  = np.digitize(pt_g, pt_bins) - 1
            bin_index_eta = np.digitize(eta_g, eta_bins) - 1


            if 0 <= bin_index_pT < N_pt and 0 <= bin_index_eta < N_eta:
                residual_pool[bin_index_pT][bin_index_eta].append(residual)
                # print(f"Filled bin pt:{bin_index_pT}, eta:{bin_index_eta}, residual:{residual}")

            if 0 <= bin_index_pT < N_pt:
                residual_pool_pT[bin_index_pT].append(residual)
            
            
            if 0 <= bin_index_eta < N_eta:
                residual_pool_eta[bin_index_eta].append(residual)

    # ---------- Compute resoultion ----------

    # resolution is the standard deviation of the residuals in each bin

    #creates a 2D array of zeros.

    resolution = np.zeros((N_pt, N_eta))
    means = np.zeros((N_pt, N_eta))

    for i in range(N_pt):
        for j in range(N_eta):
            if len(residual_pool[i][j]) > 0:
                resolution[i][j] = np.std(residual_pool[i][j])
                means[i][j] = np.mean(residual_pool[i][j])
            else:
                resolution[i][j] = np.nan
                means[i][j] = np.nan

    #de ex: residual_pool[3][5] = [0.01, -0.02, 0.005, 0.008, ...]

    resolution_pT = np.zeros(N_pt)
    means_pT = np.zeros(N_pt)
    for i in range(N_pt):
        if len(residual_pool_pT[i]) > 0:
            resolution_pT[i] = np.std(residual_pool_pT[i])
            means_pT[i] = np.mean(residual_pool_pT[i])
        else:
            resolution_pT[i] = np.nan
            means_pT[i] = np.nan
    
    resolution_eta = np.zeros(N_eta)
    means_eta = np.zeros(N_eta)
    for j in range(N_eta):
        if len(residual_pool_eta[j]) > 0:
            resolution_eta[j] = np.std(residual_pool_eta[j])
            means_eta[j] = np.mean(residual_pool_eta[j])
        else:
            resolution_eta[j] = np.nan
            means_eta[j] = np.nan

    # ---------- Debug ----------

    print("Residual pool for bin (3,5):", residual_pool[3][5])
    print("Resolution for bin (3,5):", resolution[3, 5])

    # ---------- Plot resolution heatmap ----------

    plt.figure(figsize=(10, 6))
    plt.imshow(resolution.T, origin="lower", aspect="auto", extent=[pt_bins[0], pt_bins[-1], eta_bins[0], eta_bins[-1]])
    plt.xlabel("pT_generated [GeV]")
    plt.ylabel("eta_generated")
    plt.colorbar(label="pT Resolution")
    plt.title("pT Resolution vs Generated pT and eta")
    plt.grid()
    plt.legend()
    plt.savefig(output_rms)
    # print(f"Saved: {output}")
    plt.close()

    # ---------- Plot mean heatmap ----------

    plt.figure(figsize=(10, 6))
    plt.imshow(means.T, origin="lower", aspect="auto", extent=[pt_bins[0], pt_bins[-1], eta_bins[0], eta_bins[-1]])
    plt.xlabel("pT_generated [GeV]")
    plt.ylabel("eta_generated")
    plt.colorbar(label="pT Mean")
    plt.title("pT Mean vs Generated pT and eta")
    plt.grid()
    plt.legend()
    plt.savefig(output_mean)
    # print(f"Saved: {output}")
    plt.close()

    # ---------- Plot resolution vs pT ----------

    plt.figure(figsize=(10, 6))
    plt.plot(pt_bins[:-1] + np.diff(pt_bins)/2, resolution_pT, marker='o')
    plt.xlabel("pT_generated [GeV]")
    plt.ylabel("pT Resolution")
    plt.title("pT Resolution vs Generated pT")
    plt.grid()
    plt.legend()
    plt.savefig(output_pT_rms)
    # print(f"Saved: {output_pT}")
    plt.close()

    # ---------- Plot mean vs pT ----------

    plt.figure(figsize=(10, 6))
    plt.plot(pt_bins[:-1] + np.diff(pt_bins)/2, means_pT, marker='o')
    plt.xlabel("pT_generated [GeV]")
    plt.ylabel("pT Mean")
    plt.title("pT Mean vs Generated pT")
    plt.grid()
    plt.legend()
    plt.savefig(output_pT_mean)
    # print(f"Saved: {output_pT_mean}")
    plt.close()

    # ---------- Plot resolution vs eta ----------

    plt.figure(figsize=(10, 6))
    plt.plot(eta_bins[:-1] + np.diff(eta_bins)/2, resolution_eta, marker='o')
    plt.xlabel("eta_generated")
    plt.ylabel("pT Resolution")
    plt.title("pT Resolution vs Generated eta")
    plt.grid()
    plt.legend()
    plt.savefig(output_eta_rms)
    # print(f"Saved: {output_eta}")
    plt.close()

    # ---------- Plot mean vs eta ----------

    plt.figure(figsize=(10, 6))
    plt.plot(eta_bins[:-1] + np.diff(eta_bins)/2, means_eta, marker='o')
    plt.xlabel("eta_generated")
    plt.ylabel("pT Mean")
    plt.title("pT Mean vs Generated eta")
    plt.grid()
    plt.legend()
    plt.savefig(output_eta_mean)
    # print(f"Saved: {output_eta_mean}")
    plt.close()

    # ---------- Combined summary plot ----------

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))

    # ---------- Resolution heatmap ----------

    im0 = axes[0,0].imshow(
        resolution.T,
        origin="lower",
        aspect="auto",
        extent=[pt_bins[0], pt_bins[-1], eta_bins[0], eta_bins[-1]]
    )

    axes[0,0].set_xlabel("pT_generated [GeV]")
    axes[0,0].set_ylabel("eta_generated")
    axes[0,0].set_title("pT Resolution")
    fig.colorbar(im0, ax=axes[0,0])

    # ---------- Mean heatmap ----------

    im1 = axes[0,1].imshow(
        means.T,
        origin="lower",
        aspect="auto",
        extent=[pt_bins[0], pt_bins[-1], eta_bins[0], eta_bins[-1]]
    )

    axes[0,1].set_xlabel("pT_generated [GeV]")
    axes[0,1].set_ylabel("eta_generated")
    axes[0,1].set_title("pT Mean")
    fig.colorbar(im1, ax=axes[0,1])

    # ---------- Resolution vs pT ----------

    axes[1,0].plot(pt_bins[:-1] + np.diff(pt_bins)/2, resolution_pT, marker="o")
    axes[1,0].set_xlabel("pT_generated [GeV]")
    axes[1,0].set_ylabel("pT Resolution")
    axes[1,0].set_title("pT Resolution vs pT")
    axes[1,0].grid()

    # ---------- Mean vs pT ----------

    axes[1,1].plot(pt_bins[:-1] + np.diff(pt_bins)/2, means_pT, marker="o")
    axes[1,1].set_xlabel("pT_generated [GeV]")
    axes[1,1].set_ylabel("pT Mean")
    axes[1,1].set_title("Mean vs pT")
    axes[1,1].grid()

    # ---------- Resolution vs eta ----------

    axes[2,0].plot(eta_bins[:-1] + np.diff(eta_bins)/2, resolution_eta, marker="o")
    axes[2,0].set_xlabel("eta_generated")
    axes[2,0].set_ylabel("pT Resolution")
    axes[2,0].set_title("Resolution vs eta")
    axes[2,0].grid()

    # ---------- Mean vs eta ----------

    axes[2,1].plot(eta_bins[:-1] + np.diff(eta_bins)/2, means_eta, marker="o")
    axes[2,1].set_xlabel("eta_generated")
    axes[2,1].set_ylabel("pT Mean")
    axes[2,1].set_title("Mean vs eta")
    axes[2,1].grid()

    plt.tight_layout()

    # plt.savefig("particle_tracking_btr/resolution_plots/pT_resolution_summary.png")
    plt.savefig(os.path.join(output_dir, output))

    plt.close()

def resolution_plot_p(
    particles_file,
    tracks_file,
    particle_tree="particles",
    track_tree="tracksummary",
    pt_bins=np.linspace(0.01, 10, 30),
    eta_bins=np.linspace(-2, 2, 30),
    #pt_bins=np.linspace(0, 10, 50),
    output="p_summary_resolution.png",
    output_rms="p_resolution_plot.png",
    output_mean="p_mean_plot.png",
    output_pT="p_resolution_plot_vs_pt.png",
    output_eta="p_resolution_plot_vs_eta.png",
    output_pT_mean="p_resolution_mean_plot_vs_pt.png",
    output_eta_mean="p_resolution_mean_plot_vs_eta.png",
):

    output_dir = os.path.join("particle_tracking_btr", "resolution_plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # ---------- READ PARTICLES ----------

    f_particles = uproot.open(particles_file)
    t_particles = f_particles[particle_tree]
    particles = t_particles.arrays(["particle", "pt", "p", "eta"], library="ak")

    print("nr of events:", len(particles["particle"]))

    # n_events is the nr of events in the dataset
    n_events = len(particles["particle"])

    # ---------- READ TRACKS ----------

    f_tracks = uproot.open(tracks_file)
    t_tracks = f_tracks[track_tree]
    tracks = t_tracks.arrays(["majorityParticleId", "t_pT", "t_p", "t_eta"], library="ak")

    # ---------- Bins ----------

    N_pt  = len(pt_bins) - 1
    N_eta = len(eta_bins) - 1

    # 2D list of lists for residuals
    residual_pool = [[[] for _ in range(N_eta)] for _ in range(N_pt)]

    # 1D list for residuals pT
    residual_pool_pT = [[] for _ in range(N_pt)]

    # 1D list for residuals eta
    residual_pool_eta = [[] for _ in range(N_eta)]

     # ---------- LOOP THROUGH EVENT ----------


    for event in range(n_events):

        # true particles in this event
        particle_ids = particles["particle"][event].to_list()
        particle_pts = particles["pt"][event].to_list()
        particle_etas = particles["eta"][event].to_list()
        particle_p = particles["p"][event].to_list()

        #track data
        track_truth = tracks["majorityParticleId"][event]
        track_p = tracks["t_p"][event].to_list()

        # ---------- LOOP OVER TRACKS ----------

        #looping through tracks and matching to particles
        for ids, p_r, in zip(track_truth, track_p):

            # skip broken truth info
            if len(ids) < 3:
                continue

            majority_id = ids[2]   # majorityParticleId

            # skip fake tracks
            if majority_id == 0:
                continue

            if majority_id not in particle_ids:
                continue

            # find the index of the matched particle
            index = particle_ids.index(majority_id)

            pt_g  = particle_pts[index]
            p_g = particle_p[index]
            eta_g = particle_etas[index]

            if pt_g <= 0.01:
                continue

            residual = (p_g - p_r) / p_g

            bin_index_pT  = np.digitize(pt_g, pt_bins) - 1
            bin_index_eta = np.digitize(eta_g, eta_bins) - 1

            if 0 <= bin_index_pT < N_pt and 0 <= bin_index_eta < N_eta:
                residual_pool[bin_index_pT][bin_index_eta].append(residual)

            if 0 <= bin_index_pT < N_pt:
                residual_pool_pT[bin_index_pT].append(residual)
            
            
            if 0 <= bin_index_eta < N_eta:
                residual_pool_eta[bin_index_eta].append(residual)

    # ---------- Compute resoultion ----------

    # resolution is the standard deviation of the residuals in each bin

    #creates a 2D array of zeros.

    resolution = np.zeros((N_pt, N_eta))
    means = np.zeros((N_pt, N_eta))

    for i in range(N_pt):
        for j in range(N_eta):
            if len(residual_pool[i][j]) > 0:
                resolution[i][j] = np.std(residual_pool[i][j])
                means[i][j] = np.mean(residual_pool[i][j])
            else:
                resolution[i][j] = np.nan
                means[i][j] = np.nan

    #de ex: residual_pool[3][5] = [0.01, -0.02, 0.005, 0.008, ...]

    resolution_pT = np.zeros(N_pt)
    means_pT = np.zeros(N_pt)
    for i in range(N_pt):
        if len(residual_pool_pT[i]) > 0:
            resolution_pT[i] = np.std(residual_pool_pT[i])
            means_pT[i] = np.mean(residual_pool_pT[i])
        else:
            resolution_pT[i] = np.nan
            means_pT[i] = np.nan
    
    resolution_eta = np.zeros(N_eta)
    means_eta = np.zeros(N_eta)
    for j in range(N_eta):
        if len(residual_pool_eta[j]) > 0:
            resolution_eta[j] = np.std(residual_pool_eta[j])
            means_eta[j] = np.mean(residual_pool_eta[j])
        else:
            resolution_eta[j] = np.nan
            means_eta[j] = np.nan

    # ---------- Debug ----------

    print("Residual pool for bin (3,5):", residual_pool[3][5])
    print("Resolution for bin (3,5):", resolution[3, 5])

    # ---------- Plot resolution ----------

    plt.figure(figsize=(10, 6))
    plt.imshow(resolution.T, origin="lower", aspect="auto", extent=[pt_bins[0], pt_bins[-1], eta_bins[0], eta_bins[-1]])
    plt.xlabel("pT_generated [GeV]")
    plt.ylabel("eta_generated")
    plt.colorbar(label="p Resolution")
    plt.title("Momentum (p) Resolution vs Generated pT and eta")
    plt.grid()
    plt.legend()
    plt.savefig(output_rms)
    print(f"Saved: {output_rms}")
    plt.close()

    # ---------- Plot mean heatmap ----------

    plt.figure(figsize=(10, 6))
    plt.imshow(means.T, origin="lower", aspect="auto", extent=[pt_bins[0], pt_bins[-1], eta_bins[0], eta_bins[-1]])
    plt.xlabel("pT_generated [GeV]")
    plt.ylabel("eta_generated")
    plt.colorbar(label="p Mean")
    plt.title("Momentum (p) Mean vs Generated pT and eta")
    plt.grid()
    plt.legend()
    plt.savefig(output_mean)
    # print(f"Saved: {output_mean}")
    plt.close()


    # ---------- Plot resolution vs pT ----------

    plt.figure(figsize=(10, 6))
    plt.plot(pt_bins[:-1] + np.diff(pt_bins)/2, resolution_pT, marker='o')
    plt.xlabel("pT_generated [GeV]")
    plt.ylabel("p Resolution")
    plt.title("p Resolution vs Generated pT")
    plt.grid()
    plt.legend()
    plt.savefig(output_pT)
    # print(f"Saved: {output_pT}")
    plt.close()

        # ---------- Plot mean vs pT ----------

    plt.figure(figsize=(10, 6))
    plt.plot(pt_bins[:-1] + np.diff(pt_bins)/2, means_pT, marker='o')
    plt.xlabel("pT_generated [GeV]")
    plt.ylabel("pT Mean")
    plt.title("pT Mean vs Generated pT")
    plt.grid()
    plt.legend()
    plt.savefig(output_pT_mean)
    # print(f"Saved: {output_pT_mean}")
    plt.close()

    # ---------- Plot resolution vs eta ----------

    plt.figure(figsize=(10, 6))
    plt.plot(eta_bins[:-1] + np.diff(eta_bins)/2, resolution_eta, marker='o')
    plt.xlabel("eta_generated")
    plt.ylabel("p Resolution")
    plt.title("p Resolution vs Generated eta")
    plt.grid()
    plt.legend()
    plt.savefig(output_eta)
    # print(f"Saved: {output_eta}")
    plt.close()

        # ---------- Plot mean vs eta ----------

    plt.figure(figsize=(10, 6))
    plt.plot(eta_bins[:-1] + np.diff(eta_bins)/2, means_eta, marker='o')
    plt.xlabel("eta_generated")
    plt.ylabel("pT Mean")
    plt.title("pT Mean vs Generated eta")
    plt.grid()
    plt.legend()
    plt.savefig(output_eta_mean)
    # print(f"Saved: {output_eta_mean}")
    plt.close()

    # ---------- Combined summary plot ----------

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))

    # ---------- Resolution heatmap ----------

    im0 = axes[0,0].imshow(
        resolution.T,
        origin="lower",
        aspect="auto",
        extent=[pt_bins[0], pt_bins[-1], eta_bins[0], eta_bins[-1]]
    )

    axes[0,0].set_xlabel("pT_generated [GeV]")
    axes[0,0].set_ylabel("eta_generated")
    axes[0,0].set_title("p Resolution")
    fig.colorbar(im0, ax=axes[0,0])

    # ---------- Mean heatmap ----------

    im1 = axes[0,1].imshow(
        means.T,
        origin="lower",
        aspect="auto",
        extent=[pt_bins[0], pt_bins[-1], eta_bins[0], eta_bins[-1]]
    )

    axes[0,1].set_xlabel("pT_generated [GeV]")
    axes[0,1].set_ylabel("eta_generated")
    axes[0,1].set_title("p Mean")
    fig.colorbar(im1, ax=axes[0,1])

    # ---------- Resolution vs pT ----------

    axes[1,0].plot(pt_bins[:-1] + np.diff(pt_bins)/2, resolution_pT, marker="o")
    axes[1,0].set_xlabel("pT_generated [GeV]")
    axes[1,0].set_ylabel("p Resolution")
    axes[1,0].set_title("Momentum Resolution vs pT")
    axes[1,0].grid()

    # ---------- Mean vs pT ----------

    axes[1,1].plot(pt_bins[:-1] + np.diff(pt_bins)/2, means_pT, marker="o")
    axes[1,1].set_xlabel("pT_generated [GeV]")
    axes[1,1].set_ylabel("p Mean")
    axes[1,1].set_title("Momentum Mean vs pT")
    axes[1,1].grid()

    # ---------- Resolution vs eta ----------

    axes[2,0].plot(eta_bins[:-1] + np.diff(eta_bins)/2, resolution_eta, marker="o")
    axes[2,0].set_xlabel("eta_generated")
    axes[2,0].set_ylabel("p Resolution")
    axes[2,0].set_title("Momentum Resolution vs eta")
    axes[2,0].grid()

    # ---------- Mean vs eta ----------

    axes[2,1].plot(eta_bins[:-1] + np.diff(eta_bins)/2, means_eta, marker="o")
    axes[2,1].set_xlabel("eta_generated")
    axes[2,1].set_ylabel("p Mean")
    axes[2,1].set_title("Momentum Mean vs eta")
    axes[2,1].grid()

    plt.tight_layout()

    # plt.savefig("particle_tracking_btr/resolution_plots/pT_resolution_summary.png")
    plt.savefig(os.path.join(output_dir, output))

    plt.close()


def resolution_plot_theta(
    particles_file,
    tracks_file,
    particle_tree="particles",
    track_tree="tracksummary",
    pt_bins=np.linspace(0.01, 10, 30),
    eta_bins=np.linspace(-2, 2, 30),
    #pt_bins=np.linspace(0, 10, 50),
    output="theta_resolution_plot.png",
    output_pT="theta_resolution_plot_vs_pt.png",
    output_eta="theta_resolution_plot_vs_eta.png",
):

    # ---------- READ PARTICLES ----------

    f_particles = uproot.open(particles_file)
    t_particles = f_particles[particle_tree]
    particles = t_particles.arrays(["particle", "pt", "theta", "eta"], library="ak") 

    print("nr of events:", len(particles["particle"]))

    # n_events is the nr of events in the dataset
    n_events = len(particles["particle"])

    # ---------- READ TRACKS ----------

    f_tracks = uproot.open(tracks_file)
    t_tracks = f_tracks[track_tree]
    tracks = t_tracks.arrays(["majorityParticleId", "t_pT", "t_theta", "t_eta"], library="ak")

    # ---------- Bins ----------

    N_pt  = len(pt_bins) - 1
    N_eta = len(eta_bins) - 1

    # 2D list of lists for residuals
    residual_pool = [[[] for _ in range(N_eta)] for _ in range(N_pt)]

    # 1D list for residuals pT
    residual_pool_pT = [[] for _ in range(N_pt)]

    # 1D list for residuals eta
    residual_pool_eta = [[] for _ in range(N_eta)]

     # ---------- LOOP THROUGH EVENT ----------


    for event in range(n_events):

        # true particles in this event
        particle_ids = particles["particle"][event].to_list()
        particle_pts = particles["pt"][event].to_list()
        particle_theta = particles["theta"][event].to_list()
        particle_etas = particles["eta"][event].to_list()

        #track data
        track_truth = tracks["majorityParticleId"][event]
        track_theta = tracks["t_theta"][event].to_list()

        # ---------- LOOP OVER TRACKS ----------

        #looping through tracks and matching to particles
        for ids, theta_r in zip(track_truth, track_theta):

            # skip broken truth info
            if len(ids) < 3:
                continue

            majority_id = ids[2]   # majorityParticleId

            # skip fake tracks
            if majority_id == 0:
                continue

            if majority_id not in particle_ids:
                continue

            # find the index of the matched particle
            index = particle_ids.index(majority_id)

            pt_g  = particle_pts[index]
            theta_g = particle_theta[index]
            eta_g = particle_etas[index]

            if pt_g <= 0.01:
                continue

            residual = (theta_g - theta_r) / theta_g

            bin_index_pT  = np.digitize(pt_g, pt_bins) - 1
            bin_index_eta = np.digitize(eta_g, eta_bins) - 1

            if 0 <= bin_index_pT < N_pt and 0 <= bin_index_eta < N_eta:
                residual_pool[bin_index_pT][bin_index_eta].append(residual)

            if 0 <= bin_index_pT < N_pt:
                residual_pool_pT[bin_index_pT].append(residual)
            
            
            if 0 <= bin_index_eta < N_eta:
                residual_pool_eta[bin_index_eta].append(residual)

    # ---------- Compute resoultion ----------

    # resolution is the standard deviation of the residuals in each bin

    #creates a 2D array of zeros.

    resolution = np.zeros((N_pt, N_eta))

    for i in range(N_pt):
        for j in range(N_eta):

            if len(residual_pool[i][j]) > 0:
                resolution[i][j] = np.std(residual_pool[i][j])
            else:
                resolution[i][j] = np.nan

    resolution_pT = np.zeros(N_pt)
    for i in range(N_pt):
        if len(residual_pool_pT[i]) > 0:
            resolution_pT[i] = np.std(residual_pool_pT[i])
        else:
            resolution_pT[i] = np.nan
    
    resolution_eta = np.zeros(N_eta)
    for j in range(N_eta):
        if len(residual_pool_eta[j]) > 0:
            resolution_eta[j] = np.std(residual_pool_eta[j])
        else:
            resolution_eta[j] = np.nan

    #de ex: residual_pool[3][5] = [0.01, -0.02, 0.005, 0.008, ...]

    # ---------- Debug ----------

    print("Residual pool for bin (3,5):", residual_pool[3][5])
    print("Resolution for bin (3,5):", resolution[3, 5])

    # ---------- Plot resolution ----------

    plt.figure(figsize=(10, 6))
    plt.imshow(resolution.T, origin="lower", aspect="auto", extent=[pt_bins[0], pt_bins[-1], eta_bins[0], eta_bins[-1]])
    plt.xlabel("pT_generated [GeV]")
    plt.ylabel("eta_generated")
    plt.colorbar(label="Theta Resolution")
    plt.title("Theta (θ) Resolution vs Generated pT and eta")
    plt.grid()
    plt.legend()
    plt.savefig(output)
    print(f"Saved: {output}")
    plt.close()

    # ---------- Plot resolution vs pT ----------

    plt.figure(figsize=(10, 6))
    plt.plot(pt_bins[:-1] + np.diff(pt_bins)/2, resolution_pT, marker='o')
    plt.xlabel("pT_generated [GeV]")
    plt.ylabel("Theta (θ) Resolution")
    plt.title("Theta (θ) Resolution vs Generated pT")
    plt.grid()
    plt.legend()
    plt.savefig(output_pT)
    # print(f"Saved: {output_pT}")
    plt.close()

    # ---------- Plot resolution vs eta ----------

    plt.figure(figsize=(10, 6))
    plt.plot(eta_bins[:-1] + np.diff(eta_bins)/2, resolution_eta, marker='o')
    plt.xlabel("eta_generated")
    plt.ylabel("Theta (θ) Resolution")
    plt.title("Theta (θ) Resolution vs Generated eta")
    plt.grid()
    plt.legend()
    plt.savefig(output_eta)
    # print(f"Saved: {output_eta}")
    plt.close()

def resolution_plot_phi(
    particles_file,
    tracks_file,
    particle_tree="particles",
    track_tree="tracksummary",
    pt_bins=np.linspace(0.01, 10, 30),
    eta_bins=np.linspace(-2, 2, 30),
    #pt_bins=np.linspace(0, 10, 50),
    output="phi_resolution_plot.png",
    output_pT="phi_resolution_plot_vs_pt.png",
    output_eta="phi_resolution_plot_vs_eta.png",
):

    # ---------- READ PARTICLES ----------

    f_particles = uproot.open(particles_file)
    t_particles = f_particles[particle_tree]
    particles = t_particles.arrays(["particle", "pt", "phi", "eta"], library="ak") 

    print("nr of events:", len(particles["particle"]))

    # n_events is the nr of events in the dataset
    n_events = len(particles["particle"])

    # ---------- READ TRACKS ----------

    f_tracks = uproot.open(tracks_file)
    t_tracks = f_tracks[track_tree]
    tracks = t_tracks.arrays(["majorityParticleId", "t_pT", "t_phi", "t_eta"], library="ak")

    # ---------- Bins ----------

    N_pt  = len(pt_bins) - 1
    N_eta = len(eta_bins) - 1

    # 2D list of lists for residuals
    residual_pool = [[[] for _ in range(N_eta)] for _ in range(N_pt)]

    # 1D list for residuals pT
    residual_pool_pT = [[] for _ in range(N_pt)]

    # 1D list for residuals eta
    residual_pool_eta = [[] for _ in range(N_eta)]

     # ---------- LOOP THROUGH EVENT ----------


    for event in range(n_events):

        # true particles in this event
        particle_ids = particles["particle"][event].to_list()
        particle_pts = particles["pt"][event].to_list()
        particle_phi = particles["phi"][event].to_list()
        particle_etas = particles["eta"][event].to_list()

        #track data
        track_truth = tracks["majorityParticleId"][event]
        track_pts = tracks["t_pT"][event].to_list()
        track_phi = tracks["t_phi"][event].to_list()
        track_etas = tracks["t_eta"][event].to_list()

        # ---------- LOOP OVER TRACKS ----------

        #looping through tracks and matching to particles
        for ids, phi_r, pt_r, eta_r in zip(track_truth, track_phi, track_pts, track_etas):

            # skip broken truth info
            if len(ids) < 3:
                continue

            majority_id = ids[2]   # majorityParticleId

            # skip fake tracks
            if majority_id == 0:
                continue

            if majority_id not in particle_ids:
                continue

            # find the index of the matched particle
            index = particle_ids.index(majority_id)

            pt_g  = particle_pts[index]
            phi_g = particle_phi[index]
            eta_g = particle_etas[index]

            if pt_g <= 0.01:
                continue

            residual = (phi_g - phi_r) / phi_g

            bin_index_pT  = np.digitize(pt_g, pt_bins) - 1
            bin_index_eta = np.digitize(eta_g, eta_bins) - 1

            if 0 <= bin_index_pT < N_pt and 0 <= bin_index_eta < N_eta:
                residual_pool[bin_index_pT][bin_index_eta].append(residual)

            if 0 <= bin_index_pT < N_pt:
                residual_pool_pT[bin_index_pT].append(residual)
            
            
            if 0 <= bin_index_eta < N_eta:
                residual_pool_eta[bin_index_eta].append(residual)

    # ---------- Compute resoultion ----------

    # resolution is the standard deviation of the residuals in each bin

    #creates a 2D array of zeros.

    resolution = np.zeros((N_pt, N_eta))

    for i in range(N_pt):
        for j in range(N_eta):

            if len(residual_pool[i][j]) > 0:
                resolution[i][j] = np.std(residual_pool[i][j])
            else:
                resolution[i][j] = np.nan

    resolution_pT = np.zeros(N_pt)
    for i in range(N_pt):
        if len(residual_pool_pT[i]) > 0:
            resolution_pT[i] = np.std(residual_pool_pT[i])
        else:
            resolution_pT[i] = np.nan
    
    resolution_eta = np.zeros(N_eta)
    for j in range(N_eta):
        if len(residual_pool_eta[j]) > 0:
            resolution_eta[j] = np.std(residual_pool_eta[j])
        else:
            resolution_eta[j] = np.nan

    #de ex: residual_pool[3][5] = [0.01, -0.02, 0.005, 0.008, ...]

    # ---------- Debug ----------

    print("Residual pool for bin (3,5):", residual_pool[3][5])
    print("Resolution for bin (3,5):", resolution[3, 5])

    # ---------- Plot resolution ----------

    plt.figure(figsize=(10, 6))
    plt.imshow(resolution.T, origin="lower", aspect="auto", extent=[pt_bins[0], pt_bins[-1], eta_bins[0], eta_bins[-1]])
    plt.xlabel("pT_generated [GeV]")
    plt.ylabel("eta_generated")
    plt.colorbar(label="Phi (ϕ) Resolution")
    plt.title("Phi (ϕ) Resolution vs Generated pT and eta")
    plt.grid()
    plt.legend()
    plt.savefig(output)
    print(f"Saved: {output}")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(pt_bins[:-1] + np.diff(pt_bins)/2, resolution_pT, marker='o')
    plt.xlabel("pT_generated [GeV]")
    plt.ylabel("Phi (ϕ) Resolution")
    plt.title("Phi (ϕ) Resolution vs Generated pT")
    plt.grid()
    plt.legend()
    plt.savefig(output_pT)
    # print(f"Saved: {output_pT}")
    plt.close()

    # ---------- Plot resolution vs eta ----------

    plt.figure(figsize=(10, 6))
    plt.plot(eta_bins[:-1] + np.diff(eta_bins)/2, resolution_eta, marker='o')
    plt.xlabel("eta_generated")
    plt.ylabel("Phi (ϕ) Resolution")
    plt.title("Phi (ϕ) Resolution vs Generated eta")
    plt.grid()
    plt.legend()
    plt.savefig(output_eta)
    # print(f"Saved: {output_eta}")
    plt.close()

def resolution_plot_dca(
    particles_file,
    tracks_file,
    particle_tree="particles",
    track_tree="tracksummary",
    pt_bins=np.linspace(0.01, 10, 30),
    eta_bins=np.linspace(-2, 2, 30),
    #pt_bins=np.linspace(0, 10, 50),
    output="dca_resolution_plot.png",
    output_pT="dca_resolution_plot_vs_pt.png",
    output_eta="dca_resolution_plot_vs_eta.png",
):

    # ---------- READ PARTICLES ----------

    f_particles = uproot.open(particles_file)
    t_particles = f_particles[particle_tree]
    particles = t_particles.arrays(["particle", "pt", "eta"], library="ak") 

    print("nr of events:", len(particles["particle"]))

    # n_events is the nr of events in the dataset
    n_events = len(particles["particle"])

    # ---------- READ TRACKS ----------

    f_tracks = uproot.open(tracks_file)
    t_tracks = f_tracks[track_tree]
    tracks = t_tracks.arrays(["majorityParticleId", "t_pT", "t_d0", "t_eta"], library="ak")

    # ---------- Bins ----------

    N_pt  = len(pt_bins) - 1
    N_eta = len(eta_bins) - 1

    # 2D list of lists for residuals
    residual_pool = [[[] for _ in range(N_eta)] for _ in range(N_pt)]

    # 1D list for residuals pT
    residual_pool_pT = [[] for _ in range(N_pt)]

    # 1D list for residuals eta
    residual_pool_eta = [[] for _ in range(N_eta)]

     # ---------- LOOP THROUGH EVENT ----------


    for event in range(n_events):

        # true particles in this event
        particle_ids = particles["particle"][event].to_list()
        particle_pts = particles["pt"][event].to_list()
        particle_etas = particles["eta"][event].to_list()

        #track data
        track_truth = tracks["majorityParticleId"][event]
        track_pts = tracks["t_pT"][event].to_list()
        track_dca = tracks["t_d0"][event].to_list()
        track_etas = tracks["t_eta"][event].to_list()

        # ---------- LOOP OVER TRACKS ----------

        #looping through tracks and matching to particles
        for ids, theta_r in zip(track_truth, track_dca):

            # skip broken truth info
            # if len(ids) < 3:
            #     continue

            majority_id = ids[2]   # majorityParticleId

            # skip fake tracks
            if majority_id == 0:
                continue

            if majority_id not in particle_ids:
                continue

            # find the index of the matched particle
            index = particle_ids.index(majority_id)

            pt_g  = particle_pts[index]
            eta_g = particle_etas[index]

            if pt_g <= 0.01:
                continue

            residual = theta_r  # for DCA, we can just use the value itself as the residual, since we want to see how well it is reconstructed (ideally should be close to 0)

            bin_index_pT  = np.digitize(pt_g, pt_bins) - 1
            bin_index_eta = np.digitize(eta_g, eta_bins) - 1

            if 0 <= bin_index_pT < N_pt and 0 <= bin_index_eta < N_eta:
                residual_pool[bin_index_pT][bin_index_eta].append(residual)

            if 0 <= bin_index_pT < N_pt:
                residual_pool_pT[bin_index_pT].append(residual)
            
            
            if 0 <= bin_index_eta < N_eta:
                residual_pool_eta[bin_index_eta].append(residual)

    # ---------- Compute resoultion ----------

    # resolution is the standard deviation of the residuals in each bin

    #creates a 2D array of zeros.

    resolution = np.zeros((N_pt, N_eta))

    for i in range(N_pt):
        for j in range(N_eta):

            if len(residual_pool[i][j]) > 0:
                resolution[i][j] = np.std(residual_pool[i][j])
            else:
                resolution[i][j] = np.nan

    #de ex: residual_pool[3][5] = [0.01, -0.02, 0.005, 0.008, ...]

    resolution_pT = np.zeros(N_pt)
    for i in range(N_pt):
        if len(residual_pool_pT[i]) > 0:
            resolution_pT[i] = np.std(residual_pool_pT[i])
        else:
            resolution_pT[i] = np.nan
    
    resolution_eta = np.zeros(N_eta)
    for j in range(N_eta):
        if len(residual_pool_eta[j]) > 0:
            resolution_eta[j] = np.std(residual_pool_eta[j])
        else:
            resolution_eta[j] = np.nan

    # ---------- Debug ----------

    print("Residual pool for bin (3,5):", residual_pool[3][5])
    print("Resolution for bin (3,5):", resolution[3, 5])

    # ---------- Plot resolution ----------

    plt.figure(figsize=(10, 6))
    plt.imshow(resolution.T, origin="lower", aspect="auto", extent=[pt_bins[0], pt_bins[-1], eta_bins[0], eta_bins[-1]])
    plt.xlabel("pT_generated [GeV]")
    plt.ylabel("eta_generated")
    plt.colorbar(label="Theta Resolution")
    plt.title("Theta (θ) Resolution vs Generated pT and eta")
    plt.grid()
    plt.legend()
    plt.savefig(output)
    print(f"Saved: {output}")
    plt.close()

    # ---------- Plot resolution vs pT ----------

    plt.figure(figsize=(10, 6))
    plt.plot(pt_bins[:-1] + np.diff(pt_bins)/2, resolution_pT, marker='o')
    plt.xlabel("pT_generated [GeV]")
    plt.ylabel("DCA (d0) Resolution")
    plt.title("DCA (d0) Resolution vs Generated pT")
    plt.grid()
    plt.legend()
    plt.savefig(output_pT)
    # print(f"Saved: {output_pT}")
    plt.close()

    # ---------- Plot resolution vs eta ----------

    plt.figure(figsize=(10, 6))
    plt.plot(eta_bins[:-1] + np.diff(eta_bins)/2, resolution_eta, marker='o')
    plt.xlabel("eta_generated")
    plt.ylabel("DCA (d0) Resolution")
    plt.title("DCA (d0) Resolution vs Generated eta")
    plt.grid()
    plt.legend()
    plt.savefig(output_eta)
    # print(f"Saved: {output_eta}")
    plt.close()

if __name__ == "__main__":

    resolution_plot_pT(
        particles_file="particles.root",
        tracks_file="tracksummary_ckf.root",
        # output_dir="particle_tracking_btr/resolution_plots",
    )

    # resolution_plot_p(
    #     particles_file="particles.root",
    #     tracks_file="tracksummary_ckf.root",
    # )

    # resolution_plot_theta(
    #     particles_file="particles.root",
    #     tracks_file="tracksummary_ckf.root",
    # )

    # resolution_plot_phi(
    #     particles_file="particles.root",
    #     tracks_file="tracksummary_ckf.root",
    # )

    # resolution_plot_dca(
    #     particles_file="particles.root",
    #     tracks_file="tracksummary_ckf.root",
    # )