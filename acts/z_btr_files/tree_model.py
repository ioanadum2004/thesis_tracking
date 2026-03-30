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
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print(importance)

def save_artifacts(model, scaler, path):
    path = Path(path)
    model.booster_.save_model(str(path / "tree_seed_filter_model.txt")) # LightGBM native format
    joblib.dump(scaler, path / "tree_seed_filter_scaler.pkl")
    print("Model and scaler saved.")

if __name__ == "__main__":
    root_path = "/data/alice/idumitra/thesis_tracking/acts/estimatedparams.root"
    df = load_and_label_data(root_path, output_dir)
    X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler = split_and_scale(df)
    model = train_model(X_train_scaled, y_train)
    evaluate_model(model, X_val_scaled, y_val, X_test_scaled, y_test)
    save_artifacts(model, scaler, output_dir)