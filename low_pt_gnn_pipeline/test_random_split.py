#!/usr/bin/env python3
"""
Test how random_split behaves with and without seeds.
"""

import torch
from torch.utils.data import random_split
import numpy as np

# Simulate events with z-positions
# Let's say events 0-4999 have mean z = +10mm, events 5000-9999 have mean z = -10mm
# (simulating a subtle ordering)
n_events = 10000
event_ids = list(range(n_events))
z_values = []
for i in range(n_events):
    if i < 5000:
        z_values.append(10.0 + np.random.normal(0, 5))  # Positive z
    else:
        z_values.append(-10.0 + np.random.normal(0, 5))  # Negative z

z_values = np.array(z_values)

print("="*80)
print("TESTING random_split BEHAVIOR")
print("="*80)

# Test 1: Without seed (use current RNG state)
print("\n1. random_split WITHOUT explicit seed:")
# Don't set seed - use whatever state PyTorch RNG is in
dataset = list(range(n_events))
trainset, valset, testset = random_split(dataset, [9000, 500, 500])

train_z = np.mean([z_values[i] for i in trainset.indices])
val_z = np.mean([z_values[i] for i in valset.indices])
test_z = np.mean([z_values[i] for i in testset.indices])

print(f"  Train mean z: {train_z:.2f} mm")
print(f"  Val mean z:   {val_z:.2f} mm")
print(f"  Test mean z:  {test_z:.2f} mm")
print(f"  Val-Test diff: {abs(val_z - test_z):.2f} mm")

# Test 2: With seed
print("\n2. random_split WITH seed (42):")
torch.manual_seed(42)
dataset = list(range(n_events))
trainset, valset, testset = random_split(dataset, [9000, 500, 500])

train_z = np.mean([z_values[i] for i in trainset.indices])
val_z = np.mean([z_values[i] for i in valset.indices])
test_z = np.mean([z_values[i] for i in testset.indices])

print(f"  Train mean z: {train_z:.2f} mm")
print(f"  Val mean z:   {val_z:.2f} mm")
print(f"  Test mean z:  {test_z:.2f} mm")
print(f"  Val-Test diff: {abs(val_z - test_z):.2f} mm")

# Test 3: Check if indices are shuffled
print("\n3. Checking if indices are shuffled:")
print(f"  First 10 train indices: {trainset.indices[:10]}")
print(f"  First 10 val indices:   {valset.indices[:10]}")
print(f"  First 10 test indices:  {testset.indices[:10]}")

# Test 4: Sequential split (no shuffle)
print("\n4. Sequential split (for comparison):")
train_seq = list(range(9000))
val_seq = list(range(9000, 9500))
test_seq = list(range(9500, 10000))

train_z_seq = np.mean([z_values[i] for i in train_seq])
val_z_seq = np.mean([z_values[i] for i in val_seq])
test_z_seq = np.mean([z_values[i] for i in test_seq])

print(f"  Train mean z: {train_z_seq:.2f} mm")
print(f"  Val mean z:   {val_z_seq:.2f} mm")
print(f"  Test mean z:  {test_z_seq:.2f} mm")
print(f"  Val-Test diff: {abs(val_z_seq - test_z_seq):.2f} mm")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
print("If random_split WITHOUT seed shows similar bias to sequential split,")
print("then random_split is NOT properly shuffling when no seed is set!")
