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

# Proje Kök Dizini Ayarı
# Proje kök dizinini belirle
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.model import CAFA6Model
from src.data_loader import load_data, CAFA6Dataset

# Ayarlar
BATCH_SIZE = 256        
LEARNING_RATE = 8e-4    
EPOCHS = 20
TOP_N_LABELS = 5000     
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    print(f"Genel Model Eğitimi Başlıyor (Cihaz: {DEVICE})...")
    print(f"Çalışma Dizini: {BASE_DIR}")

    # Dosya yollarını tanımla
    DATA_DIR = os.path.join(BASE_DIR, "data", "Train")
    INPUT_DIR = os.path.join(BASE_DIR, "input")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    EMB_FILE = os.path.join(INPUT_DIR, "train_embeddings_650M.npy")
    ID_FILE = os.path.join(INPUT_DIR, "train_ids_650M.npy")

    # Dosya kontrolü
    if not os.path.exists(EMB_FILE):
        print(f"HATA: Embedding dosyası bulunamadı: {EMB_FILE}")
        print("Lütfen önce 'scripts/02_extract_train_embeddings.py' dosyasını çalıştırın.")
        return

    # Veriyi yükle
    print("Veriler yükleniyor ve işleniyor...")
    terms_df, tax_df, label_map = load_data(DATA_DIR, top_n_labels=TOP_N_LABELS)
    
    # Taksonomi ID haritalama (0-N indekslemesi)
    unique_taxons = tax_df['TaxonID'].unique()
    taxon_map = {tid: i+1 for i, tid in enumerate(unique_taxons)} # 0: Bilinmeyen
    
    print(f"Hedef Sınıf Sayısı: {len(label_map)}")
    print(f"Benzersiz Tür Sayısı: {len(unique_taxons)}")

    # Veri seti ve DataLoader oluşturma
    full_dataset = CAFA6Dataset(
        embeddings_path=EMB_FILE,
        ids_path=ID_FILE,
        labels_df=terms_df,
        taxonomy_df=tax_df,
        num_classes=len(label_map),
        label_map=label_map
    )

    # Eğitim / Doğrulama Ayırma (%90 - %10)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model Kurulumu
    model = CAFA6Model(
        num_classes=len(label_map),
        num_taxons=len(unique_taxons) + 1
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

    # Eğitim Döngüsü
    best_f1 = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS}")
        
        for batch in loop:
            embs = batch['embedding'].to(DEVICE)
            
            # Taksonomi ID Eşleme
            raw_taxons = batch['taxon_id'].tolist()
            mapped_taxons = torch.tensor([taxon_map.get(t, 0) for t in raw_taxons], device=DEVICE)
            
            labels = batch['labels'].to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(embs, mapped_taxons)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # Doğrulama (Validation)
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                embs = batch['embedding'].to(DEVICE)
                raw_taxons = batch['taxon_id'].tolist()
                mapped_taxons = torch.tensor([taxon_map.get(t, 0) for t in raw_taxons], device=DEVICE)
                labels = batch['labels'].cpu().numpy()
                
                outputs = model(embs, mapped_taxons)
                probs = torch.sigmoid(outputs).cpu().numpy()
                
                val_preds.append(probs)
                val_targets.append(labels)
        
        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        
        # F1 Skoru Hesaplama (Eşik değeri 0.25)
        y_pred_bin = (val_preds > 0.25).astype(int)
        f1 = f1_score(val_targets, y_pred_bin, average='micro')
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} -> Loss: {avg_loss:.4f} | Val F1: {f1:.4f}")
        
        # Scheduler Adımı
        scheduler.step(f1)
        
        # Model Kaydetme
        if f1 > best_f1:
            best_f1 = f1
            save_path = os.path.join(MODEL_DIR, "best_model_general.pth")
            torch.save(model.state_dict(), save_path)
            
            # Etiket Haritasını Kaydet
            np.save(os.path.join(MODEL_DIR, "label_map_general.npy"), label_map)
            print(f"Yeni Rekor! Model Kaydedildi: {save_path}")

    print("Eğitim Tamamlandı!")

if __name__ == "__main__":
    train()