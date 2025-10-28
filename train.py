# train_offline.py
# Simple offline training script. Run once locally to generate artifacts.
# Usage:
#   python train_offline.py --data data/manufacturing_dataset.csv

import argparse
import pandas as pd
import backend as be

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="C:/Users/suraj/OneDrive/Desktop/capstone_1/data/manufacturing_dataset_1000_samples (1).csv",
                        help="Path to CSV with target column 'Parts_Per_Hour'")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    info = be.train_model_from_df(df)
    be.clear_artifacts_cache()
    print("Training completed.")
    print("Metrics:", info["metrics"])
    print("Artifacts saved to:", info["artifacts"])

if __name__ == "__main__":
    main()