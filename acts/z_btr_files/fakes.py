import uproot
import numpy as np
import pandas as pd

branches = [
    "event_nr", "track_nr",
    "trackClassification",
    "t_pT", "t_eta",
    "eQOP_fit", "eTHETA_fit",
    "nMeasurements", "nHoles", "nOutliers",
    "chi2Sum", "NDF",
]

with uproot.open("tracksummary_ckf.root") as f:
    tree = f["tracksummary"]
    data = {branch: tree[branch].array(library="np") for branch in branches}

df = pd.DataFrame(data)

print(type(df["trackClassification"].iloc[0]))
print(df["trackClassification"].iloc[0])

fakes = df[df["trackClassification"] == 0]
print(fakes[["t_pT", "eQOP_fit", "eTHETA_fit"]].notna().sum())
print(fakes[["t_pT", "eQOP_fit", "eTHETA_fit"]].head(5))