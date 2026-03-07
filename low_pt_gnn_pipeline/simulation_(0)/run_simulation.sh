#!/bin/bash
# Suppress ACTS debug output by setting log level environment variable
export ACTS_LOG_LEVEL=INFO

# Run the simulation
python event_generator_for_gnn_training_data.py
