import torch
import pandas as pd
import numpy as np
import sys
import os
import networkx
import obonet
from tqdm import tqdm

# Add project root to sys.path to import src modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.model import CAFA6Model

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2048

# Paths
INPUT_DIR = os.path.join(BASE_DIR, "input")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

TEST_EMB_FILE = os.path.join(INPUT_DIR, "test_embeddings_650M.npy")
TEST_ID_FILE = os.path.join(INPUT_DIR, "test_ids_650M.npy")
TAXON_FILE = os.path.join(DATA_DIR, "Test", "testsuperset-taxon-list.tsv")
TRAIN_TAXON_FILE = os.path.join(DATA_DIR, "Train", "train_taxonomy.tsv")
OBO_FILE = os.path.join(DATA_DIR, "Train", "go-basic.obo")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "submission_specialist.tsv")

# Aspects and class counts used during training
ASPECTS = {
    "F": 2000,
    "C": 1500,
    "P": 3000
}

def generate_ensemble():
    print("Starting Ensemble Inference...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Test Data
    if not os.path.exists(TEST_EMB_FILE) or not os.path.exists(TEST_ID_FILE):
        raise FileNotFoundError("Test embeddings not found. Please run embedding extraction script first.")

    test_embs = np.load(TEST_EMB_FILE)
    test_ids = np.load(TEST_ID_FILE)
    
    # 2. Prepare Taxonomy Maps
    # Load Test Taxonomy
    try:
        test_tax_df = pd.read_csv(TAXON_FILE, sep="\t", header=None, names=["ID", "TaxonID"])
        test_tax_df.set_index("ID", inplace=True)
    except Exception as e:
        print(f"Warning: Could not load test taxonomy ({e}). Defaulting to 0.")
        test_tax_df = pd.DataFrame()

    # Load Train Taxonomy to map IDs correctly
    if os.path.exists(TRAIN_TAXON_FILE):
        train_tax = pd.read_csv(TRAIN_TAXON_FILE, sep="\t", header=None, names=["ID", "TaxonID"])
        unique_taxons = train_tax['TaxonID'].unique()
        taxon_map = {tid: i+1 for i, tid in enumerate(unique_taxons)}
        num_taxons = len(unique_taxons) + 1
    else:
        raise FileNotFoundError("Train taxonomy file missing. Cannot map taxon IDs.")

    all_predictions = []

    # 3. Inference Loop for Each Specialist Model
    for aspect, num_classes in ASPECTS.items():
        print(f"Processing aspect: {aspect}...")
        
        model_path = os.path.join(MODEL_DIR, f"best_model_{aspect}.pth")
        map_path = os.path.join(MODEL_DIR, f"label_map_{aspect}.npy")
        
        if not os.path.exists(model_path):
            print(f"Skipping {aspect}: Model file not found.")
            continue

        # Load Label Map and Model
        label_map = np.load(map_path, allow_pickle=True).item()
        idx_to_term = {v: k for k, v in label_map.items()}
        
        model = CAFA6Model(num_classes=num_classes, num_taxons=num_taxons).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        
        # Batch Prediction
        for i in tqdm(range(0, len(test_ids), BATCH_SIZE), desc=f"Inference {aspect}"):
            batch_embs = torch.tensor(test_embs[i:i+BATCH_SIZE], dtype=torch.float32).to(DEVICE)
            batch_ids = test_ids[i:i+BATCH_SIZE]
            
            # Map Taxon IDs
            batch_taxons = []
            for pid in batch_ids:
                if pid in test_tax_df.index:
                    val = test_tax_df.loc[pid, "TaxonID"]
                    tid = val.iloc[0] if isinstance(val, pd.Series) else val
                    batch_taxons.append(taxon_map.get(tid, 0))
                else:
                    batch_taxons.append(0)
            
            batch_taxons = torch.tensor(batch_taxons, dtype=torch.long).to(DEVICE)
            
            with torch.no_grad():
                logits = model(batch_embs, batch_taxons)
                probs = torch.sigmoid(logits).cpu().numpy()
            
            # Filter and Store
            for j, pid in enumerate(batch_ids):
                p_probs = probs[j]
                # Lower threshold for 'P' due to higher complexity
                threshold = 0.005 if aspect == 'P' else 0.01
                
                indices = np.where(p_probs > threshold)[0]
                for idx in indices:
                    all_predictions.append(f"{pid}\t{idx_to_term[idx]}\t{p_probs[idx]:.3f}")

    # Write Temporary Raw Predictions
    temp_file = os.path.join(OUTPUT_DIR, "temp_submission.tsv")
    with open(temp_file, "w") as f:
        f.write("\n".join(all_predictions))
        
    # 4. Post-Processing: Ontology Propagation
    print("Applying Ontology Propagation (Hierarchy Rules)...")
    if not os.path.exists(OBO_FILE):
         print("Error: OBO file not found. Skipping propagation.")
         return

    graph = obonet.read_obo(OBO_FILE)
    df = pd.read_csv(temp_file, sep="\t", header=None, names=["ProteinID", "Term", "Score"])
    
    final_rows = []
    
    # Propagate scores from children to parents
    for protein_id, group in tqdm(df.groupby("ProteinID"), desc="Propagating"):
        scores = dict(zip(group["Term"], group["Score"]))
        terms = list(scores.keys())
        
        for term in terms:
            if term not in graph: continue
            current = scores[term]
            
            try:
                # Get ancestors (parents)
                ancestors = networkx.descendants(graph, term)
            except: 
                continue
            
            for anc in ancestors:
                if anc in scores:
                    scores[anc] = max(scores[anc], current)
                else:
                    scores[anc] = current
        
        # Final filter to keep file size manageable
        for term, score in scores.items():
            if score >= 0.01:
                final_rows.append(f"{protein_id}\t{term}\t{score:.3f}")
                
    # Save Final File
    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(final_rows))
        
    print(f"Inference Completed. Output saved to: {OUTPUT_FILE}")
    
    # Cleanup
    if os.path.exists(temp_file):
        os.remove(temp_file)

if __name__ == "__main__":
    generate_ensemble()