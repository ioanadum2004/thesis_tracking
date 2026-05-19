# Acts Tracking Pipeline: Replication Guide

This project develops and evaluates a machine learning seed filter integrated into the ACTS (A Common Tracking Software) particle tracking framework, targeting the low transverse momentum regime (pT ∈ [0.10, 0.50] GeV). The filter is applied between the seeding stage and the Combinatorial Kalman Filter (CKF), reducing the number of fake seed candidates before the computationally expensive track-finding stage.

Two architectures are compared: a PyTorch Multilayer Perceptron (MLP) and a LightGBM Boosted Decision Tree (BDT), both exported to ONNX for C++ inference via ONNX Runtime. Three configurations are evaluated, varying the feature set (12 vs 27 features), classification threshold (fixed vs pT-dependent), and sample weighting strategy. Results show that per-bin sample weighting is essential in the low-pT regime, where severe class imbalance causes unweighted models to collapse in the lowest momentum bin. Weighted configurations reduce the CKF runtime by 27-40% while preserving particle-level reconstruction efficiency.

Developed as a bachelor's thesis at Maastricht University in collaboration with Nikhef, the Dutch National Institute for Subatomic Physics.

This guide provides step-by-step instructions on how to set up the environment, compile the ACTS framework, and run the tracking scripts. Each script has a desctiption at the top including directions on how to run it.

---

## Environment Setup & Initialization

<details>

<summary><strong>1. First-Time Setup</strong></summary>

On the Nikhef Stoomboot cluster, navigate to your working directory under /data/alice/:

```bash
cd /data/alice/<username>
```

Activate the LCG software environment (required for C++20/GCC 13 compatibility). This must be run in every new shell session:
```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc13-opt/setup.sh
```

Clone this GitHub repository into your working directory:

```bash
git clone https://github.com/ioanadum2004/thesis_tracking.git <source>
cd <source>
```
Replace <source> with your desired directory name (e.g. acts).

Compile the ACTS Python bindings using CMake according to: https://wiki.nikhef.nl/alice/How_to_start_using_the_ACTS_framework

```bash
cmake -B build -S . -DACTS_BUILD_EXAMPLES=ON -DACTS_BUILD_EXAMPLES_PYTHON_BINDINGS=ON -DPYBIND11_USE_FETCHCONTENT=ON
```

```bash
cmake --build build
```

Source the ACTS Python setup script to make the C++ bindings importable. This must be run in every new shell session:
```bash
source build/python/setup.sh
```

Install Python dependencies into a local directory to avoid hitting disk quota on /user/:
```bash
pip install uproot numpy matplotlib lightgbm scikit-learn torch onnxruntime \
  skl2onnx onnxmltools pandas plotly \
  --target /data/alice/<username>/python_packages/
```

Add the package directory to your Python path:
```bash
export PYTHONPATH=/data/alice/<username>/python_packages/:$PYTHONPATH
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

## Repository Structure & Core Files

Understanding the inputs, scripts, and outputs is key to modifying this pipeline. Here is a breakdown of the critical files:

### Pipeline Scripts
*   **`Examples/Scripts/Python/perfect_spacepoints_multigen_btr.py`**: The main ACTS simulation and reconstruction script. It handles particle generation, simulation, seed finding, the ML filter application, and the Combinatorial Kalman Filter (CKF).
*   **`z_btr_files/mlp_model.py`**: Handles data loading, PyTorch Multilayer Perceptron training, per-bin sample weighting, and exports the trained model to ONNX format.
*   **`estimatedparameters.root`**: Contains the estimated track parameters (e.g., initial pT, direction) for each seed candidate directly after the seeding stage, before they are processed by the ML filter or CKF.
*   **`z_btr_files/tree_model.py`**: Trains the LightGBM Boosted Decision Tree (BDT) and handles the SKLearn-to-ONNX conversion.

### Generated Data Products
After running the pipeline, several `.root` files are generated. These are standard ROOT files readable via `uproot`:
*   **`hits.root`**: Contains the 3D spacepoints/measurements generated by the simulated detector.
*   **`particles.root`**: Contains the Monte Carlo (MC) truth information (exact kinematics: pT, eta, phi) for all generated particles.
*   **`tracksummary_ckf.root`**: The output of the CKF. Contains reconstructed track parameters and their matching probability to truth particles.

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

Run the following commands to generate and plot the baseline tracking data:

1. Run simulation with no model (baseline)
```bash
python Examples/Scripts/Python/perfect_spacepoints_multigen_btr.py --model none
```

2. Plot efficiency for all particles
```bash
python z_btr_files/plot_scripts/eff_plot_multiple_species.py --mode all_particles --out-dir z_btr_files/efficiency_plots/final_baseline/all_particles 
```

3. Plot efficiency for each particle species
```bash
python z_btr_files/plot_scripts/eff_plot_multiple_species.py --mode each_particle --out-dir z_btr_files/efficiency_plots/final_baseline/each_particle
```
</details>

<details>
<summary><strong>2. MLP</strong></summary>

Note: You only need to train the model once per configuration. Use --without-weights for Configuration A, and --with-weights for Configuration B.

1. Run baseline (required for baseline data generation)
```bash
python Examples/Scripts/Python/perfect_spacepoints_multigen_btr.py --model none
```

2. Train the MLP model (Run ONLY ONE of these depending on your branch)
```bash
python z_btr_files/mlp_model.py --without-weights  # Use for Configuration A
```
OR
```bash
python z_btr_files/mlp_model.py --with-weights     # Use for Configuration B & C
```

3. Run simulation using the trained MLP filter
```bash
python Examples/Scripts/Python/perfect_spacepoints_multigen_btr.py --model mlp
```

4. Generate efficiency plots
```bash
python z_btr_files/plot_scripts/eff_plot_multiple_species.py --mode all_particles --out-dir z_btr_files/efficiency_plots/final_mlp/all_particles 
```

5. Generate difference plots against baseline
```bash
python z_btr_files/plot_scripts/eff_diff_all_particles.py \
    --tracks tracksummary_ckf.root \
    --particles particles.root \
    --tracks-base tracksummary_ckf_baseline.root \
    --particles-base particles_baseline.root \
    --out-dir z_btr_files/efficiency_plots/difference_mlp
```
</details>

<details>
<summary><strong>3. LightGBM</strong></summary>

Note: You only need to train the model once per configuration. Use --without-weights for Configuration A, and --with-weights for Configuration B.

1. Run baseline (required for baseline data generation)
```bash
python Examples/Scripts/Python/perfect_spacepoints_multigen_btr.py --model none
```

2. Train the LightGBM model
```bash
python z_btr_files/tree_model.py
```

3. Run simulation using the trained LightGBM filter
```bash
python Examples/Scripts/Python/perfect_spacepoints_multigen_btr.py --model tree
```

4. Generate efficiency plots
```bash
python z_btr_files/plot_scripts/eff_plot_multiple_species.py --mode all_particles --out-dir z_btr_files/efficiency_plots/final_bdt/all_particles 
```

5. Generate difference plots against baseline
```bash
python z_btr_files/plot_scripts/eff_diff_all_particles.py \
    --tracks tracksummary_ckf.root \
    --particles particles.root \
    --tracks-base tracksummary_ckf_baseline.root \
    --particles-base particles_baseline.root \
    --out-dir z_btr_files/efficiency_plots/difference_bdt
```
</details>

## Event Visualization & Animations

To gain an intuitive understanding of the tracking environment, you can generate both interactive 3D event displays of the detector collisions and simplified, truth-matched animations of specific particle trajectories.

<details>
<summary><strong>1. Interactive 3D Collision Displays</strong></summary>

This script parses the raw spacepoints and truth tracks to build an interactive, rotatable 3D display of the entire collision event using Plotly. It is highly useful for visually auditing dense track environments and checking for geometric edge cases.

*   **Inputs required:** `hits.root` (detector spacepoints) and `particles.root` (MC truth).
*   **Execution:**
    ```bash
    python thesis_animation.py \
        --hits hits.root \
        --particles particles.root \
        --outdir ./event_visualizations
```
