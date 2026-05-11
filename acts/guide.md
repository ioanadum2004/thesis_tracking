# Acts Tracking Pipeline: Replication Guide

This guide provides step-by-step instructions on how to set up the environment, compile the ACTS framework, and run the tracking scripts.

---

## Environment Setup & Initialization

<details>

<summary><strong>1. First-Time Setup</strong></summary>

clone git plus builiding acts and everything

Compile the ACTS Python bindings using CMake according to: https://wiki.nikhef.nl/alice/How_to_start_using_the_ACTS_framework

```bash
cmake -B build -S . -DACTS_BUILD_EXAMPLES=ON -DACTS_BUILD_EXAMPLES_PYTHON_BINDINGS=ON -DPYBIND11_USE_FETCHCONTENT=ON
```

```bash
cmake --build build
```

</details>
<details>

<summary><strong>2. Daily Login Instructions</strong></summary>
*(Note: You must run these steps every time you log into the cluster before running your scripts.)*

First, log in to the Nikhef login node (use the `-X -Y` flags to enable X11 forwarding for graphical interfaces):
```bash
ssh -X -Y [username]@login.nikhef.nl
```

Once logged in, access one of the interactive nodes (you can use i1, i2, or i3). For example:

```bash
ssh -X -Y stbc-i3.nikhef.nl
```

You need to set up the correct software environments from CVMFS.

First, source the LCG release environment needed for the build dependencies:

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc13-opt/setup.sh
```

Your working directories are located in the /data partition. Navigate to the directory where your Git repositories are saved:

```bash
cd /data/alice/[username]/thesis_tracking/acts
```

Finally, source the Python environment setup script:
```bash
source build/python/setup.sh
```

</details>

---

## Running the Tracking Pipeline

Depending on the results you want to replicate, you must first check out the appropriate branch. 

### Configurations & Branches
Before running the pipelines below, ensure you are on the correct branch for the configuration you want to test:

*   **Configuration A (No Weights):** `git checkout conf_A_12_fixed_no_weights`
*   **Configuration B (With Weights):** `git checkout conf_b_12_fixed_weights`
*   **Configuration C (Main):** `git checkout main`

---

### Execution Scripts

<details>

<summary><strong>1. Baseline Pipeline</strong></summary>

Run the following commands to generate and plot the baseline tracking data.

```bash
# 1. Run simulation with no model (baseline)
python Examples/Scripts/Python/perfect_spacepoints_multigen_btr.py --model none

# 2. Plot efficiency for all particles
python z_btr_files/plot_scripts/eff_plot_multiple_species.py --mode all_particles --out-dir z_btr_files/efficiency_plots/final_baseline/all_particles 

# 3. Plot efficiency for each particle species
python z_btr_files/plot_scripts/eff_plot_multiple_species.py --mode each_particle --out-dir z_btr_files/efficiency_plots/final_baseline/each_particle
```

Note: You only need to train the model once per configuration. Use --without-weights for Configuration A, and --with-weights for Configuration B.

## MLP

1. Run baseline (required for baseline data generation)
```bash
python Examples/Scripts/Python/perfect_spacepoints_multigen_btr.py --model none
```

# 2. Train the MLP model (Run ONLY ONE of these depending on your branch)
```bash
python z_btr_files/mlp_model.py --without-weights  # Use for Configuration A
```
# OR
```bash
python z_btr_files/mlp_model.py --with-weights     # Use for Configuration B & C
```

# 3. Run simulation using the trained MLP filter
```bash
python Examples/Scripts/Python/perfect_spacepoints_multigen_btr.py --model mlp
```

# 4. Generate efficiency plots
```bash
python z_btr_files/plot_scripts/eff_plot_multiple_species.py --mode all_particles --out-dir z_btr_files/efficiency_plots/final_mlp/all_particles 
python z_btr_files/plot_scripts/eff_plot_multiple_species.py --mode each_particle --out-dir z_btr_files/efficiency_plots/final_mlp/each_particle 
```

# 5. Generate difference plots against baseline
```bash
python z_btr_files/plot_scripts/eff_diff_all_particles.py \
    --tracks tracksummary_ckf.root \
    --particles particles.root \
    --tracks-base tracksummary_ckf_baseline.root \
    --particles-base particles_baseline.root \
    --out-dir z_btr_files/efficiency_plots/difference_mlp
```

## LightGBM

# 1. Run baseline (required for baseline data generation)
```bash
python Examples/Scripts/Python/perfect_spacepoints_multigen_btr.py --model none
```

# 2. Train the LightGBM model
```bash
python z_btr_files/tree_model.py
```

# 3. Run simulation using the trained LightGBM filter
```bash
python Examples/Scripts/Python/perfect_spacepoints_multigen_btr.py --model tree
```

# 4. Generate efficiency plots
```bash
python z_btr_files/plot_scripts/eff_plot_multiple_species.py --mode all_particles --out-dir z_btr_files/efficiency_plots/final_bdt/all_particles 
python z_btr_files/plot_scripts/eff_plot_multiple_species.py --mode each_particle --out-dir z_btr_files/efficiency_plots/final_bdt/each_particle 
```

# 5. Generate difference plots against baseline
```bash
python z_btr_files/plot_scripts/eff_diff_all_particles.py \
    --tracks tracksummary_ckf.root \
    --particles particles.root \
    --tracks-base tracksummary_ckf_baseline.root \
    --particles-base particles_baseline.root \
    --out-dir z_btr_files/efficiency_plots/difference_bdt
```

</details>
