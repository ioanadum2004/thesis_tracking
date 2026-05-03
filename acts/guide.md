# Acts Tracking Pipeline: Replication Guide

This guide provides step-by-step instructions on how to set up the environment, compile the ACTS framework, and run the tracking scripts.

---

## Environment Setup & Initialization

<details>
<summary><strong>1. Connect to the Cluster</strong></summary>

First, log in to the Nikhef login node (use the `-X -Y` flags to enable X11 forwarding for graphical interfaces):
```bash
ssh -X -Y [username]@login.nikhef.nl
```

Once logged in, access one of the interactive nodes (you can use i1, i2, or i3). For example:

```bash
ssh -X -Y stbc-i3.nikhef.nl
```

You need to set up the correct software environments from CVMFS.

First, load the ALICE-specific software

```bash
source /cvmfs/alice.cern.ch/etc/login.sh
```

Next, source the LCG release environment needed for the build dependencies:

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc13-opt/setup.sh
```

Your working directories are located in the /data partition. Navigate to the directory where your Git repositories are saved:

```bash
cd /data/alice/[username]/thesis_tracking/acts
```

Compile the ACTS Python bindings using CMake according to: https://wiki.nikhef.nl/alice/How_to_start_using_the_ACTS_framework

```bash
cmake -B build -S . -DACTS_BUILD_EXAMPLES=ON -DACTS_BUILD_EXAMPLES_PYTHON_BINDINGS=ON -DPYBIND11_USE_FETCHCONTENT=ON
```

```bash
cmake --build build
```