# Acts Tracking Pipeline: Replication Guide

This guide provides step-by-step instructions on how to set up the environment, compile the ACTS framework, and run the tracking scripts.

---

## Environment Setup & Initialization

<details>
<summary><strong>1. Connect to the Cluster</strong></summary>

First, log in to the Nikhef login node (use the `-X -Y` flags to enable X11 forwarding for graphical interfaces):
```bash
ssh -X -Y [username]@login.nikhef.nl

Once logged in, access one of the interactive nodes (you can use i1, i2, or i3). For example:

```bash
ssh -X -Y stbc-i1.nikhef.nl