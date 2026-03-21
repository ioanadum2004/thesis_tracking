import ROOT
import sys

matching_file = ROOT.TFile.Open("performance_finding_ckf_matchingdetails.root") #for eveyr (reconstructable particle id, all tracks that match it as majority id)
matching_tree = matching_file.Get("matchingdetails")

particles_file = ROOT.TFile.Open("particles.root")
particles_tree = particles_file.Get("particles")

# Build particle map: (event_id, particle_id) -> (pt, eta)
particle_map = {}
for event in particles_tree:
    eid = int(event.event_id)
    for i in range(len(event.particle_id)):
        particle_map[(eid, int(event.particle_id[i]))] = (float(event.pt[i]), float(event.eta[i]))



# Collect unique reconstructable/matched particles 
reconstructable = set()                #loop over the matching details (reconstructable particles)
matched = set() 
for entry in matching_tree:
    key = (int(entry.event_nr), int(entry.particle_id))
    reconstructable.add(key)
    if entry.matched:
        matched.add(key)

# Fill numerator/denominator from matchingdetails
h_den_pt  = ROOT.TH1F("h_den_pt",  "", 50, 0, 0.5)
h_num_pt  = ROOT.TH1F("h_num_pt",  "", 50, 0, 0.5)

h_den_eta = ROOT.TH1F("h_den_eta", "", 50, -0.15, 0.15)
h_num_eta = ROOT.TH1F("h_num_eta", "", 50, -0.15, 0.15)

for key in reconstructable:                   #reconstructbale only has ecent nr and p_id now we get the pt and eta form the particle_map
    if key in particle_map:
        pt, eta = particle_map[key]
        h_den_pt.Fill(pt)
        h_den_eta.Fill(eta)
for key in matched:
    if key in particle_map:
        pt, eta = particle_map[key]
        h_num_pt.Fill(pt)
        h_num_eta.Fill(eta)

n_matched = h_num_pt.Integral()
n_total   = h_den_pt.Integral()
print(f"Overall efficiency: {n_matched:.0f}/{n_total:.0f} = {n_matched/n_total:.4f} ({n_matched/n_total*100:.2f}%)")

eff_pt  = ROOT.TEfficiency(h_num_pt,  h_den_pt)
eff_pt.SetTitle("Efficiency vs p_{T};p_{T} [GeV];Efficiency")
eff_eta = ROOT.TEfficiency(h_num_eta, h_den_eta)
eff_eta.SetTitle("Efficiency vs #eta;#eta;Efficiency")

canvas = ROOT.TCanvas("c", "Efficiency", 1600, 600)
canvas.Divide(2, 1)
canvas.cd(1)
eff_pt.Draw("AP")
canvas.cd(2)
eff_eta.Draw("AP")
out = sys.argv[1] if len(sys.argv) > 1 else "efficiency"
canvas.SaveAs(out + ".png")
print(f"Saved {out}.png")
