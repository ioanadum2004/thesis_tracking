import acts.examples
import acts
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from tree_model import FEATURE_COLS

class MLSeedFilter(acts.examples.IAlgorithm):
    def __init__(self, model_path, scaler_path, threshold, level):
        acts.examples.IAlgorithm.__init__(self, "MLSeedFilter", level)
        self.model  = lgb.Booster(model_file=str(model_path))
        self.scaler = joblib.load(scaler_path)
        self.threshold = threshold

        # Declare data handles so the sequencer knows what this algorithm reads/writes
        self.inputSeeds = acts.examples.ReadDataHandle(
            self, "InputSeeds"
        )
        self.outputSeeds = acts.examples.WriteDataHandle(
            self, "OutputSeeds"
        )
        self.inputSeeds.initialize("estimatedparameters")
        self.outputSeeds.initialize("estimatedparameters_filtered")

    def execute(self, context):
        # 1. Read seeds from the whiteboard - The whiteboard is accessed via context.eventStore
        # 2. Extract features into a DataFrame
        # 3. Scale + predict probabilities
        # 4. Keep only seeds above threshold
        # 5. Write filtered seeds back to whiteboard

        try:
            seeds = context.eventStore.get("estimatedparameters")

            print(f"[ML] Seeds in: {len(seeds)}")

            if len(seeds) == 0:
                context.eventStore.add("estimatedparameters_filtered", [])
                return acts.examples.ProcessCode.SUCCESS

            rows = []
            for seed in seeds:
                # extract first
                params = seed.parameters()
                cov    = seed.covariance()
                loc0, loc1, phi, theta, qop = params[0], params[1], params[2], params[3], params[4]
                eta = -np.log(np.tan(theta / 2))
                err_loc0  = np.sqrt(cov[0, 0])
                err_loc1  = np.sqrt(cov[1, 1])
                err_phi   = np.sqrt(cov[2, 2])
                err_theta = np.sqrt(cov[3, 3])
                err_qop   = np.sqrt(cov[4, 4])
                # then append
                rows.append({
                    "pt": np.sin(theta) / abs(qop),  # convert qop 
                    "eta":      eta,
                    "phi":      phi,
                    "theta":    theta,
                    "qop":      qop,
                    "loc0":     loc0,
                    "loc1":     loc1,
                    "err_loc0": err_loc0,
                    "err_loc1": err_loc1,
                    "err_phi":  err_phi,
                    "err_theta":err_theta,
                    "err_qop":  err_qop,
                })
            df = pd.DataFrame(rows, columns=FEATURE_COLS)

            # Scale + predict
            # Scale features
            X = self.scaler.transform(df)

            # Predict probabilities (LightGBM Booster)
            y_proba = self.model.predict(X)

            # Apply threshold
            mask = y_proba >= self.threshold

            filtered_seeds = [seed for seed, keep in zip(seeds, mask) if keep]

            print(f"[ML] Seeds out: {len(filtered_seeds)}")

            context.eventStore.add("estimatedparameters_filtered", filtered_seeds)

            return acts.examples.ProcessCode.SUCCESS
        
        except Exception as e:
            print(f"[ML] ERROR in event {context.eventNumber}: {e}")
            # Fall back to passing all seeds through unfiltered
            context.eventStore.add("estimatedparameters_filtered", seeds)
            return acts.examples.ProcessCode.SUCCESS