import uproot
import numpy as np
import matplotlib.pyplot as plt

def eff_vs_mult():

    multiplicities = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
    efficiencies_baseline = []
    efficiencies_sf = []

    fake_baseline = []
    fake_sf = []

    for mult in multiplicities:

        print(f"\n{'='*55}")
        print(f"  Multiplicity = {mult}")
        print(f"{'='*55}")

        for label, results_list, fake_list in [("baseline", efficiencies_baseline, fake_baseline), 
                                    ("seedfilter", efficiencies_sf, fake_sf)]:
            path = f"z_btr_files/multiplicity_sweep/mult_{mult}/{label}/tracksummary_ckf.root"
            
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

            eff = (unu_count + doi_count)/total_count
            fake = zero_count/total_count
            results_list.append(eff)
            fake_list.append(fake)
            print("efficiency:", eff)

    particles_per_event = [m * 5 * 5 for m in multiplicities]

    plt.figure()
    plt.plot(particles_per_event, efficiencies_baseline, "o-", color="blue",  label="Baseline")
    plt.plot(particles_per_event, efficiencies_sf,       "o-", color="orange", label="Seed Filter")
    plt.plot(particles_per_event, fake_baseline, "o-", color="green",  label="Baseline Fake Rate")
    plt.plot(particles_per_event, fake_sf,       "o-", color="red", label="Seed Filter Fake Rate")
    plt.title("Track Efficiency vs Multiplicity")
    plt.xlabel("Particles per event")
    plt.ylabel("Track Efficiency")
    plt.ylim(0, 1.05)
    plt.grid()
    plt.legend()
    plt.savefig("z_btr_files/efficiency_vs_multiplicity.png", dpi=150)
    plt.close()

    efficiencies_sf_baseline = []
    efficiencies_sf_baseline_minus = []
    counter = 1

    # print("\n Baseline \n")
    for i, mult in enumerate(multiplicities):
        efficiencies_sf_baseline.append(efficiencies_sf[i] / efficiencies_baseline[i])
        efficiencies_sf_baseline_minus.append(efficiencies_sf[i] - efficiencies_baseline[i])
        # print(counter," - ", efficiencies_sf_baseline[mult-1])
        counter += 1
        # print(efficiencies_baseline[entry], "\n")

    # print("\n Seed Filter \n")
    # for entry in range(len(efficiencies_baseline)):
    #     # efficiencies_sf_baseline[entry] = efficiencies_sf[entry]/efficiencies_baseline[entry]
    #     print(efficiencies_sf[entry], "\n")

    plt.figure()
    plt.plot(particles_per_event, efficiencies_sf_baseline, "o-", color="red",  label="Division")
    plt.title("Track Efficiency Seed Filter/Baseline vs Multiplicity")
    plt.xlabel("Particles per event")
    plt.ylabel("Track Efficiency")
    plt.ylim(0, 2.05)
    plt.grid()
    plt.legend()
    plt.savefig("z_btr_files/efficiency_vs_multiplicity_division.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(particles_per_event, efficiencies_sf_baseline_minus, "o-", color="green",  label="Minus")
    plt.title("Track Efficiency Seed Filter - Baseline vs Multiplicity")
    plt.xlabel("Particles per event")
    plt.ylabel("Track Efficiency")
    # plt.ylim(0, 0.0002)
    plt.grid()
    plt.legend()
    plt.savefig("z_btr_files/efficiency_vs_multiplicity_minus.png", dpi=150)
    plt.close()

if __name__ == "__main__":

    eff_vs_mult()