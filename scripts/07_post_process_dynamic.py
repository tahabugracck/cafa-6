import pandas as pd
import numpy as np
import os
import sys

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_SUBMISSION = os.path.join(BASE_DIR, "output", "submission_specialist.tsv")
IA_FILE = os.path.join(BASE_DIR, "data", "Train", "IA.tsv")
OUTPUT_FILE = os.path.join(BASE_DIR, "output", "submission_final.tsv")

# Configuration
TOP_K = 70  # Keep top K predictions per protein

def apply_dynamic_threshold():
    print("Starting Dynamic Thresholding (IA Weighted)...")
    
    # 1. Validation
    if not os.path.exists(INPUT_SUBMISSION):
        print(f"Error: Input file {INPUT_SUBMISSION} not found.")
        return
    if not os.path.exists(IA_FILE):
        print(f"Error: IA file {IA_FILE} not found.")
        return

    # 2. Load IA (Information Accretion) Weights
    print("Loading Information Accretion weights...")
    try:
        ia_df = pd.read_csv(IA_FILE, sep="\t", names=["Term", "IA"])
    except Exception as e:
        print(f"Error reading IA file: {e}")
        return
    
    # 3. Load Submission
    print("Loading submission file...")
    df = pd.read_csv(INPUT_SUBMISSION, sep="\t", header=None, names=["ID", "Term", "Score"], dtype={"Score": float})
    
    # 4. Merge Data
    merged = pd.merge(df, ia_df, on="Term", how="left")
    # Fill missing IA values with default (1.0)
    merged["IA"] = merged["IA"].fillna(1.0)
    
    print(f"Original rows: {len(merged):,}")
    
    # 5. Apply Dynamic Logic
    # Strategy:
    # - High IA (Rare/Valuable): Keep even if confidence is low.
    # - Low IA (Common): Keep only if confidence is high.
    
    mask_valuable = (merged["IA"] >= 2.0) & (merged["Score"] >= 0.005)
    mask_common   = (merged["IA"] < 0.5) & (merged["Score"] >= 0.15)
    mask_normal   = (merged["IA"] >= 0.5) & (merged["IA"] < 2.0) & (merged["Score"] >= 0.01)
    
    final_df = merged[mask_valuable | mask_common | mask_normal]
    
    print(f"Filtered rows: {len(final_df):,}")
    
    # 6. Top-K Filtering
    # To reduce file size and remove noise, keep only top predictions per protein
    print(f"Applying Top-{TOP_K} filter per protein...")
    final_df = final_df[["ID", "Term", "Score"]]
    final_df = final_df.sort_values(["ID", "Score"], ascending=[True, False])
    final_df = final_df.groupby("ID").head(TOP_K)
    
    # 7. Save Final Output
    print(f"Saving final submission to: {OUTPUT_FILE}")
    final_df.to_csv(OUTPUT_FILE, sep="\t", header=False, index=False)
    
    print("Post-processing completed successfully.")

if __name__ == "__main__":
    apply_dynamic_threshold()