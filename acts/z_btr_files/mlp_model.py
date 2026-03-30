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

# Define the architecture of your Multi-Layer Perceptron
class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super(SimpleMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1) 
            # Note: No Sigmoid here! We apply it later for better numerical stability during training.
        )

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

#coloanele din dataset, gen informatiile despre fiecare seed
FEATURE_COLS = [
    "pt", "eta", "phi", "theta", "qop",
    "loc0", "loc1",
    "err_loc0", "err_loc1",
    "err_phi", "err_theta", "err_qop",
]

# ------------------------- Load data -------------------------

def load_and_label_data(root_path, output_dir):
    with uproot.open(root_path) as f:
        tree = f["estimatedparams"]
        branches = FEATURE_COLS + ["truthMatched", "event_nr"]
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

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler

def train_model(X_train_scaled, y_train):
    # 1. Convert numpy arrays/pandas series to PyTorch Tensors
    X_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

    # 2. Handle class imbalance (Equivalent to scale_pos_weight in LightGBM)
    num_fakes = len(y_train[y_train == 0])
    num_reals = len(y_train[y_train == 1])
    pos_weight_val = num_fakes / num_reals
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32)

    # 3. Initialize model, loss function, and optimizer
    input_dim = X_train_scaled.shape[1]
    model = SimpleMLP(input_dim)
    
    # BCEWithLogitsLoss combines a Sigmoid layer and Binary Cross Entropy Loss securely
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. The Training Loop (unlike LightGBM, we must write the loop explicitly)
    epochs = 300
    model.train() # Set model to training mode
    print(f"Training MLP for {epochs} epochs...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()           # Clear old gradients
        outputs = model(X_tensor)       # Forward pass
        loss = criterion(outputs, y_tensor) # Calculate loss
        loss.backward()                 # Backpropagation
        optimizer.step()                # Update weights
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    return model

def evaluate_model(model, X_val_scaled, y_val, X_test_scaled, y_test):
    model.eval() # Set model to evaluation mode
    
    # 1. Run inference without tracking gradients (saves memory/time)
    with torch.no_grad():
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
        logits = model(X_val_tensor)
        
        # 2. Apply Sigmoid to convert raw outputs (logits) into probabilities (0 to 1)
        y_val_proba = torch.sigmoid(logits).numpy().flatten()
    
    # 3. The rest of your evaluation logic remains exactly the same!
    y_val_pred = (y_val_proba >= 0.4).astype(int)

    print(f"\nVal AUC:  {roc_auc_score(y_val, y_val_proba):.3f}")
    print(classification_report(y_val, y_val_pred, target_names=["fake","real"]))

    print("Threshold Scan:")
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred_thresh = (y_val_proba >= threshold).astype(int)
        real_recall = recall_score(y_val, y_pred_thresh)
        fake_recall = recall_score(y_val, y_pred_thresh, pos_label=0)
        print(f"Threshold {threshold:.1f} → real recall: {real_recall:.2f}, fake recall: {fake_recall:.2f}")

    # Note: Neural Networks do not have a built-in "feature_importances_" attribute like tree models do.
    # If you need feature importance for an MLP, you would need to implement SHAP values or permutation importance.

def save_artifacts(model, scaler, path):
    path = Path(path)
    model.eval()
    
    # To export to ONNX, PyTorch needs a "dummy input" to trace the network's shape
    # The shape must match (batch_size, number_of_features)
    # num_features = scaler.n_features_in_
    # dummy_input = torch.randn(1, num_features, dtype=torch.float32)
    
    # onnx_path = path / "mlp_seed_filter_model.onnx"

    export_model = MLPWithSigmoid(model)  # wrap it
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

if __name__ == "__main__":
    root_path = "/data/alice/idumitra/thesis_tracking/acts/estimatedparams.root"
    df = load_and_label_data(root_path, output_dir)
    X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler = split_and_scale(df)
    model = train_model(X_train_scaled, y_train)
    evaluate_model(model, X_val_scaled, y_val, X_test_scaled, y_test)
    save_artifacts(model, scaler, output_dir)