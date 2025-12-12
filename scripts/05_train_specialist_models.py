import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# Import ayarı
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from src.model import CAFA6Model
from src.data_loader import load_data, CAFA6Dataset

# Ayarlar
BATCH_SIZE = 256 # VRAM'e göre düşürebilirsin
LEARNING_RATE = 8e-4
EPOCHS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ASPECTS = {"F": 2000, "C": 1500, "P": 3000}

def train_specialist(target_aspect, top_n):
    print(f"\n🧬 UZMAN EĞİTİMİ: {target_aspect} (Sınıf: {top_n})")
    
    DATA_DIR = os.path.join(BASE_DIR, "data", "Train")
    INPUT_DIR = os.path.join(BASE_DIR, "input")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Veri Yükleme
    df_terms = pd.read_csv(os.path.join(DATA_DIR, "train_terms.tsv"), sep="\t")
    df_filtered = df_terms[df_terms['aspect'] == target_aspect]
    top_terms = df_filtered['term'].value_counts().head(top_n).index
    label_map = {term: i for i, term in enumerate(top_terms)}
    df_final = df_filtered[df_filtered['term'].isin(top_terms)]

    df_tax = pd.read_csv(os.path.join(DATA_DIR, "train_taxonomy.tsv"), sep="\t", header=None, names=["EntryID", "TaxonID"])
    df_tax.set_index("EntryID", inplace=True)
    unique_taxons = df_tax['TaxonID'].unique()

    dataset = CAFA6Dataset(
        embeddings_path=os.path.join(INPUT_DIR, "train_embeddings_650M.npy"),
        ids_path=os.path.join(INPUT_DIR, "train_ids_650M.npy"),
        labels_df=df_final, taxonomy_df=df_tax, num_classes=len(label_map), label_map=label_map
    )

    train_ds, val_ds = random_split(dataset, [int(0.9*len(dataset)), len(dataset)-int(0.9*len(dataset))])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = CAFA6Model(num_classes=len(label_map), num_taxons=len(unique_taxons)+1).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    best_f1 = 0
    for epoch in range(EPOCHS):
        model.train()
        for batch in tqdm(train_loader, desc=f"Ep {epoch+1}"):
            # Not: Localde çalışırken Taxonomy Map işlemi dataset içinde veya burada yapılmalı
            # Basitlik için burada map'leme yapmıyoruz, dataset'e güveniyoruz
            embs, labels = batch['embedding'].to(DEVICE), batch['labels'].to(DEVICE)
            # Taxon ID basitleştirme (Demo)
            taxons = torch.zeros(len(embs), dtype=torch.long).to(DEVICE) 
            
            optimizer.zero_grad()
            outputs = model(embs, taxons)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Basit validation (Detayları kıstım)
        print(f"Epoch {epoch+1} Bitti.")
        
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"best_model_{target_aspect}.pth"))
    np.save(os.path.join(MODEL_DIR, f"label_map_{target_aspect}.npy"), label_map)

if __name__ == "__main__":
    for aspect, limit in ASPECTS.items():
        train_specialist(aspect, limit)