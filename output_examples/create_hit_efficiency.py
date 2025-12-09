import ROOT
import sys
import os

def create_hit_efficiency_plot(output_filename):
    # Load ROOT files and trees
    particles_file = ROOT.TFile.Open("particles.root")
    matching_file = ROOT.TFile.Open("performance_finding_ckf_matchingdetails.root")
    particles_tree = particles_file.Get("particles")
    matching_tree = matching_file.Get("matchingdetails")

    # Define histogram binning variables
    pt_bins = 100
    pt_min = 0.0
    pt_max = 0.5

    particle_info = {}
    canvas = ROOT.TCanvas("canvas", "Measurement Efficiency Plots", 1200, 450)
    canvas.Divide(3,1)
    ROOT.gStyle.SetPalette(ROOT.kBird)
    ROOT.gStyle.SetOptStat(0)  # Disable statistics box
   
    for i in range(particles_tree.GetEntries()):
        particles_tree.GetEntry(i)
        event_id = i
        for j in range(len(particles_tree.particle_id)):
            pid = particles_tree.particle_id[j]
            key = (event_id, pid)
            particle_info[key] = {'pt': particles_tree.pt[j]}
    print(f"Loaded {len(particle_info)} particles across all events")
    
    # First pass: collect all matched tracks for each particle
    particle_tracks = {}  # key: (event, pid) -> list of (n_matched, n_truth) tuples
    for i in range(matching_tree.GetEntries()):
        matching_tree.GetEntry(i)
        event_nr = matching_tree.event_nr
        pid = matching_tree.particle_id
        is_matched = matching_tree.matched
        key = (event_nr, pid)
        if is_matched and key in particle_info:
            if key not in particle_tracks:
                particle_tracks[key] = []
            particle_tracks[key].append((
                matching_tree.nMatchedHitsOnTrack,
                matching_tree.nParticleTruthHits
            ))
    
    # Second pass: for each particle, take the track with highest n_matched (leading track)
    n_matched = 0
    for key, tracks in particle_tracks.items():
        # Sort by n_matched_measurements (descending), take the first (highest)
        best_track = max(tracks, key=lambda x: x[0])
        particle_info[key]['n_matched_measurements'] = best_track[0]
        particle_info[key]['n_truth_measurements'] = best_track[1]
        particle_info[key]['has_track'] = True
        n_matched += 1
    
    print(f"Found {n_matched} unique matched particles with measurement info (using leading track)")
    
    # Create TEfficiency plot
    # Hit efficiency = n_matched_measurements / n_truth_measurements for particles with tracks
    
    # Plot 1: Hit efficiency vs pT (for reconstructed particles only)
    canvas.cd(1)
    ROOT.gPad.SetLeftMargin(0.15)
    h_hit_eff_pt_num = ROOT.TH1F("h_hit_eff_pt_num", "Matched Measurements", pt_bins, pt_min, pt_max)
    h_hit_eff_pt_den = ROOT.TH1F("h_hit_eff_pt_den", "Truth Measurements", pt_bins, pt_min, pt_max)
    
    # Create TProfile histograms for average measurements per particle
    h_matched_profile = ROOT.TProfile("h_matched_profile", "Average Matched Measurements", pt_bins, pt_min, pt_max)
    h_truth_profile = ROOT.TProfile("h_truth_profile", "Average Truth Measurements", pt_bins, pt_min, pt_max)
    
    for pid, info in particle_info.items():
        if info.get('has_track'):  # Only for reconstructed particles
            pt = info['pt']
            # Fill TH1F with counts (for efficiency calculation)
            h_hit_eff_pt_den.Fill(pt, info['n_truth_measurements'])
            h_hit_eff_pt_num.Fill(pt, info['n_matched_measurements'])
            # Fill TProfile with counts (will automatically average per particle)
            h_matched_profile.Fill(pt, info['n_matched_measurements'])
            h_truth_profile.Fill(pt, info['n_truth_measurements'])
    
    hit_eff_pt = ROOT.TEfficiency(h_hit_eff_pt_num, h_hit_eff_pt_den)
    hit_eff_pt.SetTitle("Measurement Efficiency vs p_{T} (Reconstructed Particles);p_{T} [GeV];Measurement Efficiency")
    hit_eff_pt.Draw("AP")
    
    # Plot 2: Average matched measurements per particle
    canvas.cd(2)
    ROOT.gPad.SetLeftMargin(0.15)
    h_matched_profile.SetTitle("Average Matched Measurements per Particle vs p_{T};p_{T} [GeV];Avg. Matched Measurements")
    h_matched_profile.SetLineColor(ROOT.kBlue)
    h_matched_profile.SetLineWidth(2)
    h_matched_profile.SetMinimum(0)
    h_matched_profile.Draw()
    
    # Plot 3: Average truth measurements per particle
    canvas.cd(3)
    ROOT.gPad.SetLeftMargin(0.15)
    h_truth_profile.SetTitle("Average Truth Measurements per Particle vs p_{T};p_{T} [GeV];Avg. Truth Measurements")
    h_truth_profile.SetLineColor(ROOT.kRed)
    h_truth_profile.SetLineWidth(2)
    h_truth_profile.SetMinimum(0)
    h_truth_profile.Draw()
    
    # Print some statistics
    print("\nMeasurement Efficiency Statistics:")
    total_truth_meas = sum(info['n_truth_measurements'] for info in particle_info.values() if info.get('has_track'))
    total_matched_meas = sum(info['n_matched_measurements'] for info in particle_info.values() if info.get('has_track'))
    print(f"Total truth measurements (reconstructed particles): {total_truth_meas}")
    print(f"Total matched measurements (on tracks): {total_matched_meas}")
    if total_truth_meas > 0:
        print(f"Overall measurement efficiency: {100.0 * total_matched_meas / total_truth_meas:.2f}%")
    else:
        print("No reconstructed particles found!")
    
    canvas.Update()
    canvas.SaveAs(f"{output_filename}.png")
    print(f"\nPlot saved as '{output_filename}.png'")
    
    matching_file.Close()
    particles_file.Close()
    return canvas

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python create_hit_efficiency.py <output_filename_without_extension> [pdf]")
        sys.exit(1)

    output_filename = sys.argv[1]
    save_pdf = False
    if len(sys.argv) == 3 and sys.argv[2].lower() == "pdf":
        save_pdf = True

    canvas = create_hit_efficiency_plot(output_filename)

    if save_pdf:
        if canvas:
            canvas.SaveAs(f"{output_filename}.pdf")
            print(f"PDF plot also saved as '{output_filename}.pdf'")
        else:
            print("Warning: Could not find canvas to save as PDF.")
