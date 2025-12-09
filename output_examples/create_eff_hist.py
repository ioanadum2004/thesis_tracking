import ROOT
import sys
import os
import math

def create_combined_efficiency_and_pt_plot(output_filename):
    ROOT.gStyle.SetPalette(ROOT.kBird)

    file_path = "tracksummary_ckf.root"
    particles_file_path = "particles.root"
    
    if not os.path.exists(file_path):
        print(f"Error: Required file '{file_path}' was not found.")
        return

    root_file = ROOT.TFile.Open(file_path)
    if not root_file or root_file.IsZombie():
        print(f"Error: Could not open the file '{file_path}'.")
        return
    
    track_tree = root_file.Get("tracksummary")
    if not track_tree or not isinstance(track_tree, ROOT.TTree):
        print("Error: Could not find TTree 'tracksummary' in the file.")
        root_file.Close()
        return

    # Try to open particles.root for simulated tracks
    particles_file = None
    particles_tree = None
    if os.path.exists(particles_file_path):
        particles_file = ROOT.TFile.Open(particles_file_path)
        if particles_file and not particles_file.IsZombie():
            particles_tree = particles_file.Get("particles")
            if particles_tree and isinstance(particles_tree, ROOT.TTree):
                print(f"Found particles tree with {particles_tree.GetEntries()} entries")
            else:
                print("Warning: Could not find 'particles' tree in particles.root")
        else:
            print(f"Warning: Could not open {particles_file_path}")
    else:
        print(f"Warning: {particles_file_path} not found")


    canvas = ROOT.TCanvas("c1", "Tracking Performance Summary", 2400, 2400)
    canvas.Divide(4, 5)

    pt_bins = 100
    pt_max = 0.5 # Auto-range - let ROOT determine from data
    p_bins = 100
    p_max = 8 # Total momentum range
    eta_bins = 100
    eta_min = -0.15
    eta_max = 0.15

    # --- Load matchingdetails to get reconstructable particles ---
    matching_file_path = "performance_finding_ckf_matchingdetails.root"
    reconstructable_particles = set()
    matched_particles_from_matching = set()
    
    if os.path.exists(matching_file_path):
        matching_file = ROOT.TFile.Open(matching_file_path)
        matching_tree = matching_file.Get("matchingdetails")
        if matching_tree:
            for entry in matching_tree:
                event_nr = entry.event_nr
                pid = entry.particle_id
                key = (event_nr, pid)
                reconstructable_particles.add(key)
                if entry.matched:
                    matched_particles_from_matching.add(key)
            matching_file.Close()
            print(f"Loaded {len(reconstructable_particles)} reconstructable particles from matchingdetails")
            print(f"Of which {len(matched_particles_from_matching)} are matched ({len(matched_particles_from_matching)/len(reconstructable_particles)*100:.2f}%)")
    else:
        print(f"Warning: {matching_file_path} not found, using all particles")
    
    # --- First row: Particle-based efficiency (nMatchedParticles/nReconstructableParticles) ---
    # Build particle info map: (event_id, particle_id) -> (pt, eta, p)
    particle_map = {}
    for event_idx, event in enumerate(particles_tree):
        for i in range(len(event.particle_id)):
            pid = event.particle_id[i]
            key = (event_idx, pid)
            particle_map[key] = {
                'pt': event.pt[i],
                'eta': event.eta[i],
                'p': event.p[i]
            }
    
    canvas.cd(1)
    ROOT.gPad.SetLeftMargin(0.15)
    
    # Use reconstructable particles as denominator for efficiency
    h_den_pt = ROOT.TH1F("h_den_pt", "Reconstructable Particles vs. pT", pt_bins, 0, pt_max)
    h_num_pt = ROOT.TH1F("h_num_pt", "Matched Particles vs. pT", pt_bins, 0, pt_max)
    
    # Fill denominator with reconstructable particles
    for key in reconstructable_particles:
        if key in particle_map:
            h_den_pt.Fill(particle_map[key]['pt'])
    
    # Fill numerator with matched particles using their TRUTH pt
    for key in matched_particles_from_matching:
        if key in particle_map:
            h_num_pt.Fill(particle_map[key]['pt'])
    
    efficiency_plot_pt = ROOT.TEfficiency(h_num_pt, h_den_pt)
    efficiency_plot_pt.SetTitle("Particle Efficiency (nMatchedParticles/nTrueParticles);p_{T} [GeV];Efficiency")
    efficiency_plot_pt.Draw("AP")
 
#     h_den_pt = ROOT.TH1F("h_den_pt", "Total Tracks vs. pT", pt_bins, 0, pt_max)
#     h_num_pt = ROOT.TH1F("h_num_pt", "Good Tracks vs. pT", pt_bins, 0, pt_max)
#     track_tree.Draw("t_pT>>h_den_pt", "", "goff")
#     track_tree.Draw("t_pT>>h_num_pt", "trackClassification == 1", "goff")
#     efficiency_plot_pt = ROOT.TEfficiency(h_num_pt, h_den_pt)
#     efficiency_plot_pt.SetTitle("Track Purity vs. pT (particles.root missing);p_{T} [GeV];Purity")
#     efficiency_plot_pt.Draw("AP")

    canvas.cd(2)
    ROOT.gPad.SetLeftMargin(0.15)
    h_clone_pt = ROOT.TH1F("h_clone_pt", "Clone Tracks vs. pT", pt_bins, 0, pt_max)
    h_all_tracks_pt_for_clone = ROOT.TH1F("h_all_tracks_pt_for_clone", "All Tracks vs. pT", pt_bins, 0, pt_max)
    # Manually fill histograms for jagged arrays
    for event in track_tree:
        for i in range(len(event.trackClassification)):
            pt = event.t_pT[i]
            h_all_tracks_pt_for_clone.Fill(pt)
            if event.trackClassification[i] == 2:
                h_clone_pt.Fill(pt)
    clone_rate_plot_pt = ROOT.TEfficiency(h_clone_pt, h_all_tracks_pt_for_clone)
    clone_rate_plot_pt.SetTitle("Duplicate Rate vs. pT (nClones/nTracks);p_{T} [GeV];Duplicate Rate")
    clone_rate_plot_pt.Draw("AP")

    canvas.cd(3)
    ROOT.gPad.SetLeftMargin(0.15)
    h_fake_pt = ROOT.TH1F("h_fake_pt", "Fake Tracks vs. pT", pt_bins, 0, pt_max)
    # Use the same denominator as clone rate
    for event in track_tree:
        for i in range(len(event.trackClassification)):
            if event.trackClassification[i] == 0:  # Unknown/Fake
                h_fake_pt.Fill(event.t_pT[i])
    fake_rate_plot_pt = ROOT.TEfficiency(h_fake_pt, h_all_tracks_pt_for_clone)
    fake_rate_plot_pt.SetTitle("Fake Rate vs. pT (nFakes/nTracks);p_{T} [GeV];Fake Rate")
    fake_rate_plot_pt.Draw("AP")

    canvas.cd(4)
    ROOT.gPad.SetLeftMargin(0.15)
    pt_hist = ROOT.TH1F("pt_dist", "Reconstructed Track pT Distribution;p_{T} [GeV];Number of Tracks", pt_bins, 0, pt_max)
    track_tree.Draw("t_pT>>pt_dist")
    pt_hist.SetLineColor(4)
    pt_hist.SetLineWidth(2)
    pt_hist.Draw()

    # --- Second row: plots vs. eta ---
    canvas.cd(5)
    ROOT.gPad.SetLeftMargin(0.15)
    if particles_tree:
        h_den_eta = ROOT.TH1F("h_den_eta", "Reconstructable Particles vs. eta", eta_bins, eta_min, eta_max)
        h_num_eta = ROOT.TH1F("h_num_eta", "Matched Particles vs. eta", eta_bins, eta_min, eta_max)
        
        # Fill denominator with reconstructable particles
        for key in reconstructable_particles:
            if key in particle_map:
                h_den_eta.Fill(particle_map[key]['eta'])
        
        # Fill numerator with matched particles using their TRUTH eta
        for key in matched_particles_from_matching:
            if key in particle_map:
                h_num_eta.Fill(particle_map[key]['eta'])
        
        efficiency_plot_eta = ROOT.TEfficiency(h_num_eta, h_den_eta)
        efficiency_plot_eta.SetTitle("Particle Efficiency (nMatchedParticles/nReconstructableParticles);#eta;Efficiency")
        efficiency_plot_eta.Draw("AP")
    else:
        h_den_eta = ROOT.TH1F("h_den_eta", "Total Tracks vs. eta", eta_bins, eta_min, eta_max)
        h_num_eta = ROOT.TH1F("h_num_eta", "Good Tracks vs. eta", eta_bins, eta_min, eta_max)
        track_tree.Draw("t_eta>>h_den_eta", "", "goff")
        track_tree.Draw("t_eta>>h_num_eta", "trackClassification == 1", "goff")
        efficiency_plot_eta = ROOT.TEfficiency(h_num_eta, h_den_eta)
        efficiency_plot_eta.SetTitle("Track Purity vs. eta (particles.root missing);#eta;Purity")
        efficiency_plot_eta.Draw("AP")

    canvas.cd(6)
    ROOT.gPad.SetLeftMargin(0.15)
    h_clone_eta = ROOT.TH1F("h_clone_eta", "Clone Tracks vs. eta", eta_bins, eta_min, eta_max)
    h_all_tracks_eta_for_clone = ROOT.TH1F("h_all_tracks_eta_for_clone", "All Tracks vs. eta", eta_bins, eta_min, eta_max)
    for event in track_tree:
        for i in range(len(event.trackClassification)):
            eta = event.t_eta[i]
            h_all_tracks_eta_for_clone.Fill(eta)
            if event.trackClassification[i] == 2:
                h_clone_eta.Fill(eta)
    clone_rate_plot_eta = ROOT.TEfficiency(h_clone_eta, h_all_tracks_eta_for_clone)
    clone_rate_plot_eta.SetTitle("Duplicate Rate vs. eta (nClones/nTracks);#eta;Duplicate Rate")
    clone_rate_plot_eta.Draw("AP")

    canvas.cd(7)
    ROOT.gPad.SetLeftMargin(0.15)
    h_fake_eta = ROOT.TH1F("h_fake_eta", "Fake Tracks vs. eta", eta_bins, eta_min, eta_max)
    for event in track_tree:
        for i in range(len(event.trackClassification)):
            if event.trackClassification[i] == 0:
                h_fake_eta.Fill(event.t_eta[i])
    fake_rate_plot_eta = ROOT.TEfficiency(h_fake_eta, h_all_tracks_eta_for_clone)
    fake_rate_plot_eta.SetTitle("Fake Rate vs. eta (nFakes/nTracks);#eta;Fake Rate")
    fake_rate_plot_eta.Draw("AP")

    canvas.cd(8)
    ROOT.gPad.SetLeftMargin(0.15)
    eta_hist = ROOT.TH1F("eta_dist", "Reconstructed Track #eta Distribution;#eta;Number of Tracks", eta_bins, eta_min, eta_max)
    track_tree.Draw("t_eta>>eta_dist")
    eta_hist.SetLineColor(4)
    eta_hist.SetLineWidth(2)
    eta_hist.Draw()

    # --- Third row: plots vs. total momentum (p) ---
    canvas.cd(9)
    ROOT.gPad.SetLeftMargin(0.15)
    if particles_tree:
        h_den_p = ROOT.TH1F("h_den_p", "Reconstructable Particles vs. p", p_bins, 0, p_max)
        h_num_p = ROOT.TH1F("h_num_p", "Matched Particles vs. p", p_bins, 0, p_max)
        
        # Fill denominator with reconstructable particles
        for key in reconstructable_particles:
            if key in particle_map:
                h_den_p.Fill(particle_map[key]['p'])
        
        # Fill numerator with matched particles using their TRUTH p
        for key in matched_particles_from_matching:
            if key in particle_map:
                h_num_p.Fill(particle_map[key]['p'])
        
        efficiency_plot_p = ROOT.TEfficiency(h_num_p, h_den_p)
        efficiency_plot_p.SetTitle("Particle Efficiency (nMatchedParticles/nReconstructableParticles);p [GeV];Efficiency")
        efficiency_plot_p.Draw("AP")
    else:
        h_den_p = ROOT.TH1F("h_den_p", "Total Tracks vs. p", p_bins, 0, p_max)
        h_num_p = ROOT.TH1F("h_num_p", "Good Tracks vs. p", p_bins, 0, p_max)
        track_tree.Draw("t_p>>h_den_p", "", "goff")
        track_tree.Draw("t_p>>h_num_p", "trackClassification == 1", "goff")
        efficiency_plot_p = ROOT.TEfficiency(h_num_p, h_den_p)
        efficiency_plot_p.SetTitle("Track Purity vs. p (particles.root missing);p [GeV];Purity")
        efficiency_plot_p.Draw("AP")

    canvas.cd(10)
    ROOT.gPad.SetLeftMargin(0.15)
    h_clone_p = ROOT.TH1F("h_clone_p", "Clone Tracks vs. p", p_bins, 0, p_max)
    h_all_tracks_p_for_clone = ROOT.TH1F("h_all_tracks_p_for_clone", "All Tracks vs. p", p_bins, 0, p_max)
    for event in track_tree:
        for i in range(len(event.trackClassification)):
            p = event.t_p[i]
            h_all_tracks_p_for_clone.Fill(p)
            if event.trackClassification[i] == 2:
                h_clone_p.Fill(p)
    clone_rate_plot_p = ROOT.TEfficiency(h_clone_p, h_all_tracks_p_for_clone)
    clone_rate_plot_p.SetTitle("Duplicate Rate vs. p (nClones/nTracks);p [GeV];Duplicate Rate")
    clone_rate_plot_p.Draw("AP")

    canvas.cd(11)
    ROOT.gPad.SetLeftMargin(0.15)
    h_fake_p = ROOT.TH1F("h_fake_p", "Fake Tracks vs. p", p_bins, 0, p_max)
    for event in track_tree:
        for i in range(len(event.trackClassification)):
            if event.trackClassification[i] == 0:
                h_fake_p.Fill(event.t_p[i])
    fake_rate_plot_p = ROOT.TEfficiency(h_fake_p, h_all_tracks_p_for_clone)
    fake_rate_plot_p.SetTitle("Fake Rate vs. p (nFakes/nTracks);p [GeV];Fake Rate")
    fake_rate_plot_p.Draw("AP")

    canvas.cd(12)
    ROOT.gPad.SetLeftMargin(0.15)
    p_hist = ROOT.TH1F("p_dist", "Reconstructed Track p Distribution;p [GeV];Number of Tracks", p_bins, 0, p_max)
    track_tree.Draw("t_p>>p_dist")
    p_hist.SetLineColor(4)
    p_hist.SetLineWidth(2)
    p_hist.Draw()

    # --- Fourth row: Track-based purity (nMatchedTracks/nAllTracks) ---
    canvas.cd(13)
    ROOT.gPad.SetLeftMargin(0.15)
    h_all_tracks_pt = ROOT.TH1F("h_all_tracks_pt", "All Tracks vs. pT", pt_bins, 0, pt_max)
    h_matched_tracks_pt = ROOT.TH1F("h_matched_tracks_pt", "Matched Tracks vs. pT", pt_bins, 0, pt_max)
    track_tree.Draw("t_pT>>h_all_tracks_pt", "", "goff")
    track_tree.Draw("t_pT>>h_matched_tracks_pt", "trackClassification == 1", "goff")
    purity_plot_pt = ROOT.TEfficiency(h_matched_tracks_pt, h_all_tracks_pt)
    purity_plot_pt.SetTitle("Track Purity (nMatchedTracks/nAllTracks);p_{T} [GeV];Purity")
    purity_plot_pt.Draw("AP")

    canvas.cd(14)
    ROOT.gPad.SetLeftMargin(0.15)
    h_all_tracks_eta = ROOT.TH1F("h_all_tracks_eta", "All Tracks vs. eta", eta_bins, eta_min, eta_max)
    h_matched_tracks_eta = ROOT.TH1F("h_matched_tracks_eta", "Matched Tracks vs. eta", eta_bins, eta_min, eta_max)
    track_tree.Draw("t_eta>>h_all_tracks_eta", "", "goff")
    track_tree.Draw("t_eta>>h_matched_tracks_eta", "trackClassification == 1", "goff")
    purity_plot_eta = ROOT.TEfficiency(h_matched_tracks_eta, h_all_tracks_eta)
    purity_plot_eta.SetTitle("Track Purity (nMatchedTracks/nAllTracks);#eta;Purity")
    purity_plot_eta.Draw("AP")

    canvas.cd(15)
    ROOT.gPad.SetLeftMargin(0.15)
    h_all_tracks_p = ROOT.TH1F("h_all_tracks_p", "All Tracks vs. p", p_bins, 0, p_max)
    h_matched_tracks_p = ROOT.TH1F("h_matched_tracks_p", "Matched Tracks vs. p", p_bins, 0, p_max)
    track_tree.Draw("t_p>>h_all_tracks_p", "", "goff")
    track_tree.Draw("t_p>>h_matched_tracks_p", "trackClassification == 1", "goff")
    purity_plot_p = ROOT.TEfficiency(h_matched_tracks_p, h_all_tracks_p)
    purity_plot_p.SetTitle("Track Purity (nMatchedTracks/nAllTracks);p [GeV];Purity")
    purity_plot_p.Draw("AP")

    canvas.cd(16)
    ROOT.gPad.SetLeftMargin(0.15)
    # Duplication rate among successfully reconstructed tracks (clones/(matched+clones))
    h_clone_vs_matched_pt = ROOT.TH1F("h_clone_vs_matched_pt", "Clone Tracks vs. pT", pt_bins, 0, pt_max)
    h_matched_plus_clone_pt = ROOT.TH1F("h_matched_plus_clone_pt", "Matched+Clone Tracks vs. pT", pt_bins, 0, pt_max)
    for event in track_tree:
        for i in range(len(event.trackClassification)):
            pt = event.t_pT[i]
            if event.trackClassification[i] == 1:  # Matched
                h_matched_plus_clone_pt.Fill(pt)
            elif event.trackClassification[i] == 2:  # Clone
                h_clone_vs_matched_pt.Fill(pt)
                h_matched_plus_clone_pt.Fill(pt)  # Also add to denominator
    dup_among_good_pt = ROOT.TEfficiency(h_clone_vs_matched_pt, h_matched_plus_clone_pt)
    dup_among_good_pt.SetTitle("Duplication Among Good Tracks (nClones/(nMatched+nClones));p_{T} [GeV];Duplication Rate")
    dup_among_good_pt.Draw("AP")

    # --- Fifth row: Reconstructable tracks and Simulated tracks plots ---
    # Pad 17: Reconstructable Tracks vs. pT
    canvas.cd(17)
    ROOT.gPad.SetLeftMargin(0.15)
    h_reconstructable_pt = h_den_pt.Clone("h_reconstructable_pt")
    h_reconstructable_pt.SetTitle("Reconstructable Tracks vs. pT;p_{T} [GeV];Number of Reconstructable Tracks")
    h_reconstructable_pt.SetLineColor(2)  # Red color
    h_reconstructable_pt.SetLineWidth(2)
    h_reconstructable_pt.Draw()

    # Pad 18: Reconstructable Tracks vs. eta
    canvas.cd(18)
    ROOT.gPad.SetLeftMargin(0.15)
    h_reconstructable_eta = h_den_eta.Clone("h_reconstructable_eta")
    h_reconstructable_eta.SetTitle("Reconstructable Tracks vs. #eta;#eta;Number of Reconstructable Tracks")
    h_reconstructable_eta.SetLineColor(2)  # Red color
    h_reconstructable_eta.SetLineWidth(2)
    h_reconstructable_eta.Draw()

    # Pad 19: Reconstructable Tracks vs. total momentum (p)
    canvas.cd(19)
    ROOT.gPad.SetLeftMargin(0.15)
    h_reconstructable_p = h_den_p.Clone("h_reconstructable_p")
    h_reconstructable_p.SetTitle("Reconstructable Tracks vs. p;p [GeV];Number of Reconstructable Tracks")
    h_reconstructable_p.SetLineColor(2)  # Red color
    h_reconstructable_p.SetLineWidth(2)
    h_reconstructable_p.Draw()

    # Pad 20: Simulated Tracks vs. eta (from particles.root)
    canvas.cd(20)
    ROOT.gPad.SetLeftMargin(0.15)
    if particles_tree:
        h_simulated_eta = ROOT.TH1F("h_simulated_eta", "Simulated Tracks vs. #eta;#eta;Number of Simulated Tracks", eta_bins, eta_min, eta_max)
        particles_tree.Draw("eta>>h_simulated_eta", "", "goff")
        h_simulated_eta.SetLineColor(8)  # Green color
        h_simulated_eta.SetLineWidth(2)
        h_simulated_eta.Draw()
    else:
        # Create empty histogram with message if particles.root not available
        h_empty = ROOT.TH1F("h_empty", "Simulated Tracks (particles.root not found);#eta;Number of Tracks", eta_bins, eta_min, eta_max)
        h_empty.Draw()


    canvas.Update()
    canvas.SaveAs(f"{output_filename}.png")
    print(f"Combined plot saved as '{output_filename}.png'")
    root_file.Close()
    if particles_file:
        particles_file.Close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python create_combined_efficiency_and_pt_plot.py <output_filename_without_extension>")
        sys.exit(1)
    
    output_filename = sys.argv[1]
    create_combined_efficiency_and_pt_plot(output_filename)