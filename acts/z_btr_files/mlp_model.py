import uproot
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, recall_score, precision_score
import sys
sys.path.insert(0, "/data/alice/idumitra/thesis_tracking/python_packages")
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import json
import matplotlib.pyplot as plt

# Define the architecture of the MLP
# feedforward network with 2 hidden layers, ReLU activations, and an output layer with 1 neuron (for binary classification)
# relu is a common activation function that helps the network learn complex patterns, and the final output layer will produce a single value (logit) that can be converted to a probability with a sigmoid later.
class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super(SimpleMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32), # input layer: 12 features → 32 neurons
            nn.ReLU(),                # activation function
            nn.Linear(32, 16),        # hidden layer: 32 → 16 neurons
            nn.ReLU(),                # activation function
            nn.Linear(16, 1)          # output layer: 16 → 1 neuron (logit for binary classification)
            # a sigmoid means the output will be between 0 and 1, for probabilities, add it later 

            # nn.Linear(input_dim, 64),
            # nn.ReLU(),
            # nn.Linear(64, 32),
            # nn.ReLU(),
            # nn.Linear(32, 1)
        )

        # each layer kinda "learns" to transform the data into a more useful representation for the next layer, and the final output is a single number that can be interpreted as the likelihood of being a "real seed" after applying sigmoid.

    def forward(self, x):
        return self.network(x)
    
class MLPWithSigmoid(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.base(x))

# creating_dataset.py
# ├── load_and_label_data(root_path)     → returns df
# ├── split_and_scale(df)                → returns X/y splits + fitted scaler  
# ├── train_model(X_train, y_train)      → returns trained model
# ├── evaluate_model(model, scaler, ...)  → prints metrics, threshold scan
# ├── save_artifacts(model, scaler, path) → saves .txt and .pkl
# └── main()    

#where you read info from the root file and where you save the csv dataset
output_dir = Path("/data/alice/idumitra/thesis_tracking/acts/z_btr_files")

#coloanele din dataset, gen informatiile despre fiecare seed - they come from TrackParametersContainer
ROOT_BRANCHES = [
    "pt", "eta", "phi", "theta", "qop",
    "loc0", "loc1",
    "err_loc0", "err_loc1",
    "err_phi", "err_theta", "err_qop",
]

# this one for old dataset

#FEATURE_COLS = [
#    "pt", "eta", "phi", "theta", "qop",
#    "loc0", "loc1",
#    "err_loc0", "err_loc1",
#    "err_phi", "err_theta", "err_qop",
#]

# this one for new dataset

FEATURE_COLS = [
     "pt", "eta", "phi", "theta", "qop",
     "loc0", "loc1",
     "err_loc0", "err_loc1",
     "err_phi", "err_theta", "err_qop",
     # new features from CSV
     "bX", "bY", "bZ", "mX", "mY", "mZ", "tX", "tY", "tZ",
     # engineered features
     "pull_loc0", "pull_loc1",
     "dist_bm", "dist_mt", "dist_bt", "dist_ratio",
 ]

CSV_DIR = Path("/data/alice/idumitra/thesis_tracking/acts")  # adjust path

# 7    err_loc0           0
# 8    err_loc1           0
# 9     err_phi           0
# 10  err_theta           0

# ------------------------- Load data -------------------------

# dataset that has the original seed features from estimatedparams.root, plus the joined features from the CSV files (quality, vertexZ, spacepoint coords), plus some engineered features like pull and distances. This is the one we will train on.
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

# old verion without coordinates
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

def compute_sample_weights(df, pt_bins=None):
    """
    Computes per-sample weights based on real:fake ratio within each pT bin.
    Within each bin, fake seeds are upweighted to match the real seed count.
    """
    if pt_bins is None:
        pt_bins = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    
    pt_labels = [f"{pt_bins[i]:.2f}-{pt_bins[i+1]:.2f}" for i in range(len(pt_bins)-1)]
    df = df.copy()
    df["pt_bin"] = pd.cut(df["pt"], bins=pt_bins, labels=pt_labels)
    weights = np.ones(len(df))
    
    for pt_bin in pt_labels:
        bin_mask = df["pt_bin"] == pt_bin
        subset = df[bin_mask]
        if len(subset) == 0:
            continue
        
        n_real = (subset["label"] == 1).sum()
        n_fake = (subset["label"] == 0).sum()
        
        if n_fake == 0 or n_real == 0:
            continue
        
        # upweight fakes within this bin to match real count
        fake_weight = n_real / n_fake
        
        fake_mask = bin_mask & (df["label"] == 0)
        weights[fake_mask.values] = fake_weight
        
        print(f"Bin {pt_bin}: n_real={n_real}, n_fake={n_fake}, fake_weight={fake_weight:.2f}")
    
    return weights

# ------------------------- Split & scale -------------------------

def split_and_scale(df):
    # Step 1: split at the EVENT level, not seed level
    all_events = df["event_nr"].unique()
    train_events, temp_events = train_test_split(all_events, test_size=0.3, random_state=42) # 70% train, 30% temp
    val_events,   test_events = train_test_split(temp_events, test_size=0.5, random_state=42) # split temp into 50% val, 50% test → overall 70% train, 15% val, 15% test
    #random_state is just a seed for the random number generator, so that you get the same split every time you run the code. You can choose any integer, or omit it for a different random split each time.

    # Step 2: select rows belonging to each split
    # adica toate seed-urile dintr-un event merg in acelasi split, nu ai un seed in train si altul in val/test din acelasi event
    train_df = df[df["event_nr"].isin(train_events)]
    val_df   = df[df["event_nr"].isin(val_events)]
    test_df  = df[df["event_nr"].isin(test_events)]

    # Step 3: separate features and labels
    X_train = train_df[FEATURE_COLS]
    y_train = train_df["label"]

    X_val   = val_df[FEATURE_COLS]
    y_val   = val_df["label"]

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

    # return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler, val_df, test_df

def train_model_old(X_train_scaled, y_train):
    # 1. Convert numpy arrays/pandas series to PyTorch Tensors
    X_tensor = torch.tensor(X_train_scaled, dtype=torch.float32) 
    y_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

    # 2. Handle class imbalance - meaning if you have many more fake seeds than real seeds, the model might just learn to predict "fake" all the time. To prevent this, we can use a weighted loss function that gives more importance to the minority class (real seeds).
    num_fakes = len(y_train[y_train == 0])
    num_reals = len(y_train[y_train == 1])
    pos_weight_val = num_fakes / num_reals # this means the loss will penalize mistakes on real seeds more than mistakes on fake seeds, encouraging the model to pay more attention to correctly identifying real seeds.
    # you can also do it the other way around, so you prioritize the fake seeds
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32)

    # 3. Initialize model, loss function, and optimizer
    input_dim = X_train_scaled.shape[1]
    model = SimpleMLP(input_dim)
    
    # BCEWithLogitsLoss combines a Sigmoid layer and Binary Cross Entropy Loss securely - it rewards confident correct predictions and heavily punishes confident wrong predictions.
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # measures how wrong the model's predictions are
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam is an optimization algorithm that adjusts the model's weights based on the computed gradients to minimize the loss function. It's popular for training neural networks because it adapts the learning rate for each parameter, which can lead to faster convergence.

    # 4. The Training Loop - where the model learns from the data by adjusting its weights to minimize the loss function
    epochs = 1000
    model.train() # Set model to training mode
    print(f"Training MLP for {epochs} epochs...")

    train_losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()           # clear old gradients
        outputs = model(X_tensor)       # forward pass, computes predictions
        loss = criterion(outputs, y_tensor) # calculate loss
        loss.backward()                 # backpropagation, computes gradients
        optimizer.step()                # update weights w adam
        
        train_losses.append(loss.item())

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    return model, train_losses

def train_model(X_train_scaled, y_train, train_df=None):
    X_tensor = torch.tensor(X_train_scaled, dtype=torch.float32) 
    y_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

    num_fakes = len(y_train[y_train == 0])
    num_reals = len(y_train[y_train == 1])
    pos_weight_val = num_fakes / num_reals
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32)

    input_dim = X_train_scaled.shape[1]
    model = SimpleMLP(input_dim)
    
    # if train_df is provided, compute per-bin sample weights
    # otherwise fall back to the global pos_weight approach
    if train_df is not None:
        # build a lightweight df just for weight computation
        weight_df = train_df[["pt"]].copy().reset_index(drop=True)
        weight_df["label"] = y_train.values
        sample_weights = compute_sample_weights(weight_df)
        weight_tensor = torch.tensor(sample_weights, dtype=torch.float32).unsqueeze(1)
        # use unweighted loss — we apply weights manually per sample
        criterion = nn.BCEWithLogitsLoss(reduction='none')
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        weight_tensor = None

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 1000
    model.train()
    print(f"Training MLP for {epochs} epochs...")

    train_losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        
        if weight_tensor is not None:
            # compute loss per sample, then weight and average manually
            loss = (criterion(outputs, y_tensor) * weight_tensor).mean()
        else:
            loss = criterion(outputs, y_tensor)
            
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    return model, train_losses

def evaluate_model(model, X_val_scaled, y_val):
    model.eval() # Set model to evaluation mode
    
    # 1. Run inference without tracking gradients (saves memory/time)
    with torch.no_grad():
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
        logits = model(X_val_tensor)
        
        # 2. Apply Sigmoid to convert raw outputs (logits) into probabilities (0 to 1)
        y_val_proba = torch.sigmoid(logits).numpy().flatten()
    
    # 3. the rest of the evaluation logic remains the same
    y_val_pred = (y_val_proba >= 0.4).astype(int) # threshold = 0.4

    print(f"\nVal AUC:  {roc_auc_score(y_val, y_val_proba):.3f}")
    print(classification_report(y_val, y_val_pred, target_names=["fake","real"]))

    print("Threshold Scan:")
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred_thresh = (y_val_proba >= threshold).astype(int)
        real_recall = recall_score(y_val, y_pred_thresh)
        fake_recall = recall_score(y_val, y_pred_thresh, pos_label=0)
        print(f"Threshold {threshold:.1f} → real recall: {real_recall:.2f}, fake recall: {fake_recall:.2f}")

    #  the tradeoff between fake rejection and real seed efficiency

def save_artifacts(model, scaler, path):
    path = Path(path)
    model.eval()
    
    # To export to ONNX, PyTorch needs a "dummy input" to trace the network's shape
    # The shape must match (batch_size, number_of_features)
    # num_features = scaler.n_features_in_
    # dummy_input = torch.randn(1, num_features, dtype=torch.float32)
    
    # onnx_path = path / "mlp_seed_filter_model.onnx"

    export_model = MLPWithSigmoid(model)  # wrap it so cpp gets the probabilities
    dummy_input = torch.randn(1, scaler.n_features_in_, dtype=torch.float32)
    torch.onnx.export(
        export_model,
        dummy_input,
        path / "mlp_seed_filter_model.onnx",
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    artifacts = {
        "scaler_means": list(scaler.mean_),
        "scaler_stds":  list(scaler.scale_),
        "feature_cols": FEATURE_COLS
    }
    with open(path / "scaler_params.json", "w") as f:
        json.dump(artifacts, f, indent=2)
    print(f"Scaler params saved to: {path / 'scaler_params.json'}")
    
    # Save the scaler exactly as before
    joblib.dump(scaler, path / "mlp_seed_filter_scaler.pkl")
    print(f"\nModel saved to: {path / 'mlp_seed_filter_model.onnx'}")
    print(f"Scaler saved to: {path / 'mlp_seed_filter_scaler.pkl'}")

# plot loss

def plot_loss(train_losses, output_dir):
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('BCE With Logits Loss')
    plt.title('Model Training Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot to your directory
    plot_path = Path(output_dir) / "training_loss.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nLoss plot saved to: {plot_path}")
    plt.close() # Close the figure to free up memory

def wrong_evaluate_by_pt(model, X_val_scaled, val_df, threshold=0.4, dynamic=False):
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
    
def evaluate_by_pt(model, X_val_scaled, y_val, df_val, threshold=0.4):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
        proba = torch.sigmoid(model(X_tensor)).numpy().flatten()
    
    df_eval = df_val.reset_index(drop=True)
    df_eval["proba"] = proba

    # --- dynamic thresholding based on pT ---

    print("\n-- Dynamic threshold --")
    
    def get_threshold(pt_value):
        if pt_value < 0.15:
            return 0.20
        elif pt_value < 0.20:
            return 0.30
        else:
            return 0.40

    df_eval["threshold_used"] = df_eval["pt"].apply(get_threshold)
    df_eval["predicted"] = (df_eval["proba"] >= df_eval["threshold_used"]).astype(int)
    
    # ---------------
    # if u want to use a fixed threshold instead of dynamic, just uncomment this line and comment out the dynamic thresholding above
    
   # print("\n-- Fixed threshold (threshold={threshold}): --")

    #df_eval["predicted"] = (proba >= threshold).astype(int)
    
    # define pT bins relevant to your range
    pt_bins = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    pt_labels = [f"{pt_bins[i]:.2f}-{pt_bins[i+1]:.2f}" for i in range(len(pt_bins)-1)]
    df_eval["pt_bin"] = pd.cut(df_eval["pt"], bins=pt_bins, labels=pt_labels)
    
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
        
        # if pt_bin == "0.10-0.15":
        #     real_scores = subset[real_mask]["proba"]
        #     fake_scores = subset[fake_mask]["proba"]
        #     print(f"\n  Real score distribution: mean={real_scores.mean():.3f}, "
        #         f"median={real_scores.median():.3f}, std={real_scores.std():.3f}")
        #     print(f"  Fake score distribution: mean={fake_scores.mean():.3f}, "
        #         f"median={fake_scores.median():.3f}, std={fake_scores.std():.3f}")
    
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

if __name__ == "__main__":
    root_path = "/data/alice/idumitra/thesis_tracking/acts/estimatedparams.root"
    df = load_and_label_data(root_path, output_dir) # all 29 features, including coordinates and engineered ones
    X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler, val_df, test_df = split_and_scale(df)

    # reconstruct train_df so we have unscaled pT for weight computation
    train_df = df.loc[y_train.index] # train df is just the original df rows corresponding to the training indices, so we can access the unscaled features like pT for computing weights

    # ---- Model 1: no per-bin weights (global pos_weight only) ----
    print("\n========== NO BIN WEIGHTS ==========")
    # model_no_weights, losses_no_weights = train_model(X_train_scaled, y_train)  # no train_df passed
    # plot_loss(losses_no_weights, output_dir)
    # evaluate_model(model_no_weights, X_val_scaled, y_val)
    
    # # print("\n-- Fixed threshold (0.4) --")
    # evaluate_by_pt(model_no_weights, X_val_scaled, y_val, val_df, threshold=0.4)

    # ---- Model 2: with per-bin weights ----
    print("\n========== WITH BIN WEIGHTS ==========")
    model_weights, losses_weights = train_model(X_train_scaled, y_train, train_df=train_df)  # train_df passed
    plot_loss(losses_weights, output_dir)
    evaluate_model(model_weights, X_val_scaled, y_val)
    
    # print("\n-- Dynamic --")
    evaluate_by_pt(model_weights, X_val_scaled, y_val, val_df, threshold=0.4)

    # save whichever model you decide is better
    save_artifacts(model_weights, scaler, output_dir)
