# Stage 0: ACTS Simulation

Generates simulated collision events using the ACTS framework for testing the tracking pipeline.

## Scripts

- **`generate_gnn_training_data.py`** - Generate ACTS simulation data (CSV files) using ACTS framework with particle gun (muons, 0.1-0.5 GeV pT)

## Configuration

- **Config file:** `../acorn_configs/simulation_(0)/acts_simulation.yaml`

## Usage

All scripts should be run from this stage directory:

```bash
cd /data/alice/bkuipers/low_pt_gnn_pipeline/simulation_\(0\)
python generate_gnn_training_data.py
```

## Output

- **Data directory:** `../data/csv/` - CSV spacepoint files from ACTS simulation
