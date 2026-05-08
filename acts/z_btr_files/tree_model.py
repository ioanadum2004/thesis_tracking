"""
tree_model.py

Trains a LightGBM seed filter for the ACTS tracking pipeline.

Reads estimated track parameters from `estimatedparams.root` and spacepoint
coordinates from per-event CSV files produced by the CsvSeedWriter, constructs
a labelled dataset of real and fake seeds, trains a LightGBM classifier with
per-bin sample weighting to handle class imbalance, and exports the trained
model to ONNX format for C++ inference within the ACTS pipeline.

Usage
-----
    python tree_model.py

Requirements
------------
- estimatedparams.root must be present at the path specified by `root_path`
- Per-event CSV files (event000000000-seed.csv, ...) must be present at CSV_DIR
- Output artifacts are saved to the directory specified by `output_dir`

Output
------
    tree_seed_filter_model.txt   : LightGBM model in native format
    tree_seed_filter_model.onnx  : LightGBM model in ONNX format for C++ inference
    tree_seed_filter_scaler.pkl  : fitted StandardScaler (Python)
    tree_seed_filter_scaler.json : scaler mean and variance per feature (C++)
    dataset_B.csv                : full labelled dataset

Module structure
----------------
    load_and_label_data → split_and_scale → train_model
    → evaluate_model → evaluate_by_pt → save_artifacts

See guide.md for detailed documentation of each function.
"""

import uproot
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, recall_score, precision_score
import sys
sys.path.insert(0, "/data/alice/idumitra/thesis_tracking/python_packages")
import lightgbm as lgb
import joblib
from lightgbm import Booster
#from skl2onnx import convert_sklearn
#from skl2onnx.common.data_types import FloatTensorType
import json
from onnxmltools.convert import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType
import matplotlib.pyplot as plt

# creating_dataset.py
# ├── load_and_label_data(root_path)     → returns df
# ├── split_and_scale(df)                → returns X/y splits + fitted scaler  
# ├── train_model(X_train, y_train)      → returns trained model
# ├── evaluate_model(model, scaler, ...)  → prints metrics, threshold scan
# ├── save_artifacts(model, scaler, path) → saves .txt and .pkl
# └── main()    

#where you read info from the root file and where you save the csv dataset
output_dir = Path("/data/alice/idumitra/thesis_tracking/acts/z_btr_files/models")

#coloanele din dataset, gen informatiile despre fiecare seed
ROOT_BRANCHES = [
    "pt", "eta", "phi", "theta", "qop",
    "loc0", "loc1",
    "err_loc0", "err_loc1",
    "err_phi", "err_theta", "err_qop",
]

FEATURE_COLS = [
   "pt", "eta", "phi", "theta", "qop",
   "loc0", "loc1",
   "err_loc0", "err_loc1",
   "err_phi", "err_theta", "err_qop",
]

# FEATURE_COLS = [
#     "pt", "eta", "phi", "theta", "qop",
#     "loc0", "loc1",
#     "err_loc0", "err_loc1",
#     "err_phi", "err_theta", "err_qop",
#     # new features from CSV
#     "bX", "bY", "bZ", "mX", "mY", "mZ", "tX", "tY", "tZ",
#     # engineered features
#     "pull_loc0", "pull_loc1",
#     "dist_bm", "dist_mt", "dist_bt", "dist_ratio",
# ]

CSV_DIR = Path("/data/alice/idumitra/thesis_tracking/acts")  # adjust path

# 7    err_loc0           0
# 8    err_loc1           0
# 9     err_phi           0
# 10  err_theta           0

# ------------------------- Load data -------------------------

# for 27 features
def load_and_label_data(root_path, output_dir):
    with uproot.open(root_path) as f:
        tree = f["estimatedparams"]
        # branches = ROOT_BRANCHES + ["truthMatched", "event_nr"]
        branches = [
            "pt", "eta", "phi", "theta", "qop",
            "loc0", "loc1",
            "err_loc0", "err_loc1",
            "err_phi", "err_theta", "err_qop",
            "truthMatched", "event_nr"
        ]
        df = tree.arrays(branches, library="pd")
    
    # Rename label column
    df = df.rename(columns={"truthMatched": "label"})
    df["label"] = df["label"].astype(int)

    # Add a seed index per event - so at the end you have the seeds counted, 0,1,2,... for each event separately. This is just for analysis, not used as a feature.
    df["seed_id"] = df.groupby("event_nr").cumcount()

    # df["is_low_pt"] = (df["pt"] < 0.15).astype(float)
    # df["is_mid_pt"] = ((df["pt"] >= 0.15) & (df["pt"] < 0.25)).astype(float)

    # ---- NEW THINGS ---

    # --- engineered features from estimatedparams - pull is the residual (difference between measured and true value) divided by the uncertainty, so it tells you how many "sigma" away the measurement is from the truth. A pull close to 0 means the measurement is consistent with the truth within uncertainties, while a large pull (positive or negative) indicates a significant deviation that might be worth investigating.
    df["pull_loc0"] = df["loc0"] / (df["err_loc0"] + 1e-9)
    df["pull_loc1"] = df["loc1"] / (df["err_loc1"] + 1e-9)

    # --- load and join CSV files ---
    csv_frames = []

    for event_id in df["event_nr"].unique():
        csv_path = CSV_DIR / f"event{int(event_id):09d}-seed.csv" # this looks like event000000123-seed.csv
        if not csv_path.exists():
            print(f"Warning: missing CSV for event {event_id}")
            continue
        csv_df = pd.read_csv(csv_path)
        csv_df["event_nr"] = event_id
        # seed_id in CSV should match cumcount order - verify this assumption!
        csv_df["seed_id"] = range(len(csv_df))
        csv_frames.append(csv_df[[
            "event_nr", "seed_id",
            "bX","bY","bZ","mX","mY","mZ","tX","tY","tZ",
        ]])

    if csv_frames:
        df_csv = pd.concat(csv_frames, ignore_index=True)
        df = df.merge(df_csv, on=["event_nr", "seed_id"], how="left") # left means we keep all rows from df, and add columns from df_csv where we have a match on event_nr and seed_id. If no match, we get NaN for the new columns, which we can handle later.

        # engineered spacepoint features
        df["dist_bm"] = np.sqrt((df.mX-df.bX)**2 + (df.mY-df.bY)**2 + (df.mZ-df.bZ)**2)
        df["dist_mt"] = np.sqrt((df.tX-df.mX)**2 + (df.tY-df.mY)**2 + (df.tZ-df.mZ)**2)
        df["dist_bt"] = np.sqrt((df.tX-df.bX)**2 + (df.tY-df.bY)**2 + (df.tZ-df.bZ)**2)
        df["dist_ratio"] = df["dist_bm"] / (df["dist_mt"] + 1e-6)
    else:
        print("Warning: no CSV files found, training without spacepoint features")
        # remove spacepoint features from FEATURE_COLS if no CSV
    
    # drop rows where join failed
    df = df.dropna(subset=FEATURE_COLS)

    # ---- NEW THINGS end ---
    
    # Quick sanity check
    print(f"Total seeds:  {len(df)}")
    print(f"Real seeds:   {df['label'].sum()} ({100*df['label'].mean():.1f}%)")
    print(f"Fake seeds:   {(1-df['label']).sum()} ({100*(1-df['label']).mean():.1f}%)")
    print(f"\nEvents: {df['event_nr'].nunique()}")
    print(f"\nFirst few rows:")
    # print(df[["event_nr","seed_id","pt","eta","phi","theta","qop","loc0","loc1","label"]].head(10))
    print(df.head(10))

    df.to_csv(output_dir / "dataset_B.csv", index=False)
    print(f"\nSaved dataset_B.csv with {len(df)} rows and {len(df.columns)} columns")

    return df

# ------------------------- Load data -------------------------

# 12 features

def load_and_label_data_old(root_path, output_dir):
    with uproot.open(root_path) as f:
        tree = f["estimatedparams"]
        branches = [
            "pt", "eta", "phi", "theta", "qop",
            "loc0", "loc1",
            "err_loc0", "err_loc1",
            "err_phi", "err_theta", "err_qop",
            "truthMatched", "event_nr"
        ]
        df = tree.arrays(branches, library="pd")
    
    # Rename label column
    df = df.rename(columns={"truthMatched": "label"})
    df["label"] = df["label"].astype(int)

    # Add a seed index per event - so at the end you have the seeds counted, 0,1,2,... for each event separately. This is just for analysis, not used as a feature.
    df["seed_id"] = df.groupby("event_nr").cumcount()
    
    # Quick sanity check
    print(f"Total seeds:  {len(df)}")
    print(f"Real seeds:   {df['label'].sum()} ({100*df['label'].mean():.1f}%)")
    print(f"Fake seeds:   {(1-df['label']).sum()} ({100*(1-df['label']).mean():.1f}%)")
    print(f"\nEvents: {df['event_nr'].nunique()}")
    print(f"\nFirst few rows:")
    print(df[["event_nr","seed_id","pt","eta","phi","theta","qop","loc0","loc1","label"]].head(10))

    df.to_csv(output_dir / "dataset_B.csv", index=False)
    print(f"\nSaved dataset_B.csv with {len(df)} rows and {len(df.columns)} columns")

    return df

# ------------------------- Split & scale -------------------------

def split_and_scale(df):
    # Step 1: split at the EVENT level, not seed level
    all_events = df["event_nr"].unique()
    train_events, temp_events = train_test_split(all_events, test_size=0.3, random_state=42) # 70% train, 30% temp
    val_events,   test_events = train_test_split(temp_events, test_size=0.5, random_state=42) # split temp into 50% val, 50% test → overall 70% train, 15% val, 15% test
    #random_state is just a seed for the random number generator, so that you get the same split every time you run the code. You can choose any integer, or omit it for a different random split each time.

    # Step 2: select rows belonging to each split
    train_df = df[df["event_nr"].isin(train_events)]
    val_df   = df[df["event_nr"].isin(val_events)]
    test_df  = df[df["event_nr"].isin(test_events)]

    # Step 3: separate features and labels
    X_train = train_df[FEATURE_COLS]
    y_train = train_df["label"]
    pt_train = train_df["pt"].values

    X_val   = val_df[FEATURE_COLS]
    y_val   = val_df["label"]
    pt_val  = val_df["pt"].values

    X_test  = test_df[FEATURE_COLS]
    y_test  = test_df["label"]

    # Step 4: scale/normalise — fit ONLY on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # learns mean/std here
    X_val_scaled   = scaler.transform(X_val)         # applies same mean/std
    X_test_scaled  = scaler.transform(X_test)        # applies same mean/std
    #learn the scaling parameters from training data only, then apply those same parameters to 
    # val and test so everything is on the same scale as what the model was trained on.

    print(f"Train: {len(X_train)} seeds, {y_train.mean()*100:.1f}% real")
    print(f"Val:   {len(X_val)} seeds,   {y_val.mean()*100:.1f}% real")
    print(f"Test:  {len(X_test)} seeds,  {y_test.mean()*100:.1f}% real")

    # you scale because some features might have very different ranges (e.g. pt could be 0-100 GeV, while loc0 and loc1 are small numbers around 0.01). Scaling helps the model learn better by putting all features on a similar scale. It can also speed up training and improve convergence.

    # return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler
    # return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler, val_df, pt_train 
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler, val_df, pt_train, pt_val

def train_model_old(X_train_scaled, y_train):
    model = lgb.LGBMClassifier(
        n_estimators=50, # number of trees to build - each one learns something slightly different to correct the previous ones
        max_depth=4, # how deep each tree can grow - prevents overfitting - max 4 yes/no questions per tree
        learning_rate=0.1,
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]) # this is the ratio of fake to real seeds in the training data, 
        # tells the model "when you misclassify a fake, penalise that mistake more heavily". Essentially it artificially balances the importance of the two classes during training.
    )
    # The scale_pos_weight line tells LightGBM to penalise missing a fake seed more, compensating for the class imbalance. 

    model.fit(X_train_scaled, y_train)
    
    return model

def train_model(X_train_scaled, y_train, X_val_scaled, y_val, y_train_pt=None, y_val_pt=None, use_bin_weights=False):
    model = lgb.LGBMClassifier(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
    )

    if use_bin_weights and y_train_pt is not None:
        pt_bins = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
        sample_weights = np.ones(len(y_train))
        val_sample_weights = np.ones(len(y_val))
        
        for i in range(len(pt_bins) - 1):
            # Training weights
            mask = (y_train_pt >= pt_bins[i]) & (y_train_pt < pt_bins[i+1])
            y_bin = y_train[mask]
            
            n_real = (y_bin == 1).sum()
            n_fake = (y_bin == 0).sum()
            
            if n_real > 0 and n_fake > 0:
                # same logic as scale_pos_weight but per bin
                #bin_weight = n_fake / n_real
                #sample_weights[mask] = bin_weight
                real_mask = mask & (y_train.values == 1)
                sample_weights[real_mask] = n_fake / n_real
            
            # Validation weights
            mask_val = (y_val_pt >= pt_bins[i]) & (y_val_pt < pt_bins[i+1])
            y_bin_val = y_val[mask_val]
            n_real_v = (y_bin_val == 1).sum()
            n_fake_v = (y_bin_val == 0).sum()
            if n_real_v > 0 and n_fake_v > 0:
                real_mask_val = mask_val & (y_val.values == 1)
                val_sample_weights[real_mask_val] = n_fake_v / n_real_v
        
        # model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        model.fit(
            X_train_scaled, y_train, 
            sample_weight=sample_weights,
            eval_set=[(X_train_scaled, y_train), (X_val_scaled, y_val)],
            eval_sample_weight=[sample_weights, val_sample_weights],
            eval_metric="binary_logloss"
        )
    else:
        # pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
        # model.set_params(scale_pos_weight=pos_weight)
        # model.fit(X_train_scaled, y_train)
        pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
        model.set_params(scale_pos_weight=pos_weight)
        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_train_scaled, y_train), (X_val_scaled, y_val)],
            eval_metric="binary_logloss"
        )

    return model

def evaluate_model(model, X_val_scaled, y_val, X_test_scaled, y_test):
    # Evaluate on val
    # y_val_pred  = model.predict(X_val_scaled) #make predictions on the validation data. It returns a hard decision for each seed: 0 (fake) or 1 (real).
    y_val_proba = model.predict_proba(X_val_scaled)[:, 1] #instead of a hard decision, this returns a probability for each seed.
    y_val_pred = (y_val_proba >= 0.4).astype(int) #instead of a hard decision, this applies a threshold to the predicted probabilities.

    print(f"Val AUC:  {roc_auc_score(y_val, y_val_proba):.3f}") #this compares your predicted probabilities against the true labels and computes the AUC score
    print(classification_report(y_val, y_val_pred, target_names=["fake","real"]))

    # Try a higher threshold - only reject if very confident it's fake
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred_thresh = (y_val_proba >= threshold).astype(int)
        real_recall = recall_score(y_val, y_pred_thresh)
        fake_recall = recall_score(y_val, y_pred_thresh, pos_label=0)
        print(f"Threshold {threshold:.1f} → real recall: {real_recall:.2f}, fake recall: {fake_recall:.2f}")

    importance = pd.DataFrame({
        "feature": FEATURE_COLS,
        # "importance": model.feature_importances_
        "importance": model.booster_.feature_importance(importance_type='gain')
    }).sort_values("importance", ascending=False)

    print(importance)

# def save_artifacts(model, scaler, path):
#     path = Path(path)
#     model.booster_.save_model(str(path / "tree_seed_filter_model.txt")) # LightGBM native format
#     joblib.dump(scaler, path / "tree_seed_filter_scaler.pkl")
#     print("Model and scaler saved.")

def save_artifacts(model, scaler, path):
    path = Path(path)
    model.booster_.save_model(str(path / "tree_seed_filter_model.txt"))
    joblib.dump(scaler, path / "tree_seed_filter_scaler.pkl")
    
    # onnx for cpp
    n_features = len(scaler.mean_) 
    initial_type = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_lightgbm(model, initial_types=initial_type, zipmap=False)
   #onnx_model = convert_sklearn(model, initial_types=initial_type, zipmap=False)
    #onnx_model = convert_sklearn(model, initial_types=initial_type)

    import onnx
    from onnx import helper

    graph = onnx_model.graph
    indices_init = helper.make_tensor("gather_indices", onnx.TensorProto.INT64, [1], [1])
    graph.initializer.append(indices_init)
    gather_node = helper.make_node(
        "Gather",
        inputs=["probabilities", "gather_indices"],
        outputs=["output"],
        axis=1
    )
    graph.node.append(gather_node)
    del graph.output[:]
    graph.output.append(
        helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [None, 1])
    )
    
    with open(path / "tree_seed_filter_model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    # inspect output names
    #import onnxruntime as rt
    #sess = rt.InferenceSession(str(path / "tree_seed_filter_model.onnx"))
    #print("ONNX inputs:")
    #for i in sess.get_inputs():
    #    print(f"  {i.name} {i.shape}")
    #print("ONNX outputs:")
    #for o in sess.get_outputs():
    #    print(f"  {o.name} {o.shape}")

        # save scaler params as json
    scaler_params = {
        "scaler_means": scaler.mean_.tolist(),
        "scaler_stds": scaler.scale_.tolist()
    }
    with open(path / "scaler_params.json", "w") as f:
        json.dump(scaler_params, f)
    
    print("Saved: .txt, .pkl, .onnx, scaler .json")

def evaluate_by_pt_old(model, X_val_scaled, val_df, threshold=0.4):
    """
    Evaluates real and fake recall binned by unscaled pT.
    """
    # 1. Get probabilities using LightGBM (replaces PyTorch tensors/sigmoid)
    proba = model.predict_proba(X_val_scaled)[:, 1]
    
    # 2. Prepare the evaluation dataframe using the unscaled validation data
    df_eval = val_df.reset_index(drop=True).copy()
    df_eval["proba"] = proba
    df_eval["predicted"] = (proba >= threshold).astype(int)
    
    # 3. Define pT bins relevant to your range
    # Note: Ensure your actual pT values don't exceed 0.5 or fall below 0.1, 
    # otherwise pd.cut will assign them as NaN. Adjust bins if necessary.
    pt_bins = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    pt_labels = [f"{pt_bins[i]:.2f}-{pt_bins[i+1]:.2f}" for i in range(len(pt_bins)-1)]
    df_eval["pt_bin"] = pd.cut(df_eval["pt"], bins=pt_bins, labels=pt_labels)
    
    print(f"\nPerformance by pT bin (threshold={threshold}):")
    print(f"{'pT bin':<15} {'real recall':>12} {'fake recall':>12} {'n_real':>8} {'n_fake':>8}")
    print("-" * 60)
    
    for pt_bin in pt_labels:
        mask = df_eval["pt_bin"] == pt_bin
        subset = df_eval[mask]
        if len(subset) == 0:
            continue
        
        real_mask = subset["label"] == 1
        fake_mask = subset["label"] == 0
        
        if real_mask.sum() > 0:
            real_recall = (subset[real_mask]["predicted"] == 1).mean()
        else:
            real_recall = float("nan")
            
        if fake_mask.sum() > 0:
            fake_recall = (subset[fake_mask]["predicted"] == 0).mean()
        else:
            fake_recall = float("nan")
        
        print(f"{pt_bin:<15} {real_recall:>12.3f} {fake_recall:>12.3f} "
              f"{real_mask.sum():>8} {fake_mask.sum():>8}")

def evaluate_by_pt(model, X_val_scaled, val_df, threshold=0.4, dynamic=False):
    """
    Evaluates real and fake recall binned by unscaled pT.
    
    Args:
        dynamic: if True, uses pT-dependent thresholds:
                 pT < 0.15 → 0.20, pT < 0.20 → 0.30, pT >= 0.20 → 0.40
        threshold: fixed threshold to use when dynamic=False (default 0.4)
    """
    # 1. Get probabilities using LightGBM
    proba = model.predict_proba(X_val_scaled)[:, 1]
    
    # 2. Prepare the evaluation dataframe
    df_eval = val_df.reset_index(drop=True).copy()
    df_eval["proba"] = proba

    # 3. Apply threshold — dynamic or fixed
    if dynamic:
        def get_threshold(pt_value):
            if pt_value < 0.15:
                return 0.20
            elif pt_value < 0.20:
                return 0.30
            else:
                return 0.40

        df_eval["threshold_used"] = df_eval["pt"].apply(get_threshold)
        df_eval["predicted"] = (df_eval["proba"] >= df_eval["threshold_used"]).astype(int)
        threshold_label = "dynamic"
    else:
        df_eval["predicted"] = (proba >= threshold).astype(int)
        threshold_label = threshold

    # 4. Define pT bins
    pt_bins = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    pt_labels = [f"{pt_bins[i]:.2f}-{pt_bins[i+1]:.2f}" for i in range(len(pt_bins)-1)]
    df_eval["pt_bin"] = pd.cut(df_eval["pt"], bins=pt_bins, labels=pt_labels)
    
    print(f"\nPerformance by pT bin (threshold={threshold_label}):")
    print(f"{'pT bin':<15} {'real recall':>12} {'fake recall':>12} {'n_real':>8} {'n_fake':>8}")
    print("-" * 60)
    
    for pt_bin in pt_labels:
        mask = df_eval["pt_bin"] == pt_bin
        subset = df_eval[mask]
        if len(subset) == 0:
            continue
        
        real_mask = subset["label"] == 1
        fake_mask = subset["label"] == 0
        
        real_recall = (subset[real_mask]["predicted"] == 1).mean() if real_mask.sum() > 0 else float("nan")
        fake_recall = (subset[fake_mask]["predicted"] == 0).mean() if fake_mask.sum() > 0 else float("nan")
        
        print(f"{pt_bin:<15} {real_recall:>12.3f} {fake_recall:>12.3f} "
              f"{real_mask.sum():>8} {fake_mask.sum():>8}")
    
    print("-" * 60)

def show_class_balance(val_df, pt_bins=None):
    """Shows real:fake ratio per pT bin."""
    if pt_bins is None:
        pt_bins = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    
    pt_labels = [f"{pt_bins[i]:.2f}-{pt_bins[i+1]:.2f}" for i in range(len(pt_bins)-1)]
    df = val_df.copy()
    df["pt_bin"] = pd.cut(df["pt"], bins=pt_bins, labels=pt_labels)
    
    print(f"{'pT bin':<15} {'n_real':>8} {'n_fake':>8} {'ratio (r:f)':>12} {'% fake':>10}")
    print("-" * 55)
    
    for pt_bin in pt_labels:
        subset = df[df["pt_bin"] == pt_bin]
        if len(subset) == 0:
            continue
        n_real = (subset["label"] == 1).sum()
        n_fake = (subset["label"] == 0).sum()
        ratio = n_real / n_fake if n_fake > 0 else float("inf")
        pct_fake = 100 * n_fake / len(subset)
        print(f"{pt_bin:<15} {n_real:>8} {n_fake:>8} {ratio:>12.2f} {pct_fake:>9.1f}%")
    
    print("-" * 55)
    total_real = (df["label"] == 1).sum()
    total_fake = (df["label"] == 0).sum()
    print(f"{'TOTAL':<15} {total_real:>8} {total_fake:>8} "
          f"{total_real/total_fake:>12.2f} {100*total_fake/len(df):>9.1f}%")

def plot_lgbm_loss(model, output_dir, filename="tree_loss_curve.png"):
    # LightGBM stores the evaluation history in model.evals_result_
    results = model.evals_result_
    
    # Extract log loss (binary_logloss)
    train_loss = results['training']['binary_logloss']
    val_loss = results['valid_1']['binary_logloss']
    
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss, label='Training Loss', color='#90CAF9')
    plt.plot(val_loss, label='Validation Loss', color='#FFCC80')
    plt.xlabel('Tree Iteration (n_estimators)')
    plt.ylabel('Binary Logloss')
    plt.title('LightGBM Training and Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plot_path = Path(output_dir) / filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nLoss plot saved to: {plot_path}")
    plt.close()
    
if __name__ == "__main__":
    root_path = "/data/alice/idumitra/thesis_tracking/acts/estimatedparams.root"
    df = load_and_label_data_old(root_path, output_dir) # 12 features

    # Show class balance for entire dataset
    print("\n-- Class balance per pT bin (full dataset) --")
    show_class_balance(df)
    
    X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler, val_df, pt_train, pt_val = split_and_scale(df)

    # model = train_model_old(X_train_scaled, y_train)

    # with bin weights
    #model = train_model(X_train_scaled, y_train, y_train_pt=pt_train, use_bin_weights=True)
    model = train_model(X_train_scaled, y_train, X_val_scaled, y_val, y_train_pt=pt_train, y_val_pt=pt_val, use_bin_weights=True)

    # without
    # model = train_model(X_train_scaled, y_train)

    plot_lgbm_loss(model, output_dir, filename="tree_loss_curve_weighted.png")

    evaluate_model(model, X_val_scaled, y_val, X_test_scaled, y_test)
    print("\n-- Fixed threshold --")
    evaluate_by_pt(model, X_val_scaled, val_df, threshold=0.4, dynamic=False)
    
    print("\n-- Dynamic threshold --")
    evaluate_by_pt(model, X_val_scaled, val_df, dynamic=True)
    save_artifacts(model, scaler, output_dir)
