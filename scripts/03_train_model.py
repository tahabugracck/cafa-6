import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score

# Proje yolunu ekle (src klasörünü bulabilmek için)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.data_loader import load_data, CAFA6Dataset
from src.model import CAFA6Model

# --- AYARLAR ---
EPOCHS = 20           # Kaç tur dönecek?
BATCH_SIZE = 128      # Bir seferde kaç protein alacak? (i9 işlemcide 32-64 yapabilirsin, GPU varsa 256)
LEARNING_RATE = 1e-3
TOP_N_LABELS = 1500   # En sık geçen 1500 etiketi tahmin edeceğiz
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    print(f"🔥 Eğitim Başlıyor... Cihaz: {DEVICE}")
    
    # 1. Veri Hazırlığı
    DATA_DIR = os.path.join(BASE_DIR, "data", "Train")
    EMB_DIR = os.path.join(BASE_DIR, "input")
    EMB_FILE = os.path.join(EMB_DIR, "train_embeddings_650M.npy")
    ID_FILE = os.path.join(EMB_DIR, "train_ids_650M.npy")

    # Ham veriyi işle
    terms_df, tax_df, label_map = load_data(DATA_DIR, top_n_labels=TOP_N_LABELS)
    
    # Taksonomi ID'lerini 0,1,2... formatına çevirmek için harita
    unique_taxons = tax_df['TaxonID'].unique()
    taxon_map = {tid: i+1 for i, tid in enumerate(unique_taxons)} # 0'ı unknown için ayırdık
    
    # Dataset oluştur
    full_dataset = CAFA6Dataset(
        embeddings_path=EMB_FILE,
        ids_path=ID_FILE,
        labels_df=terms_df,
        taxonomy_df=tax_df,
        num_classes=TOP_N_LABELS,
        label_map=label_map
    )
    
    # Dataset içinde taxon ID'leri integer'a çevirmemiz lazım (Hızlı hack)
    # Gerçek projede bunu Dataset class'ının içinde yapmak daha temizdir ama şimdilik burada yapalım.
    def taxon_transform(dataset_obj):
        new_taxons = []
        for tid in dataset_obj.taxonomy_df['TaxonID']:
            new_taxons.append(taxon_map.get(tid, 0))
        return new_taxons
    
    # Train / Val Ayrımı (%90 - %10)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"📊 Eğitim Verisi: {train_size}, Doğrulama: {val_size}")

    # 2. Model Kurulumu
    model = CAFA6Model(
        num_classes=TOP_N_LABELS,
        num_taxons=len(unique_taxons) + 1
    ).to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss() # Multi-label için standart kayıp fonksiyonu
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

    # 3. Eğitim Döngüsü
    best_f1 = 0.0
    
    # Model kaydetme klasörü
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        # Progress Bar
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in loop:
            embs = batch['embedding'].to(DEVICE)
            # Taxon ID'lerini map'leyerek gönder
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
            
        # 4. Doğrulama (Validation)
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
        
        # Basit F1 Hesaplama (Threshold 0.3)
        y_pred_bin = (val_preds > 0.3).astype(int)
        f1 = f1_score(val_targets, y_pred_bin, average='micro')
        
        print(f"Epoch {epoch+1} -> Loss: {total_loss/len(train_loader):.4f} | Val F1 Score: {f1:.4f}")
        
        # Scheduler güncelle
        scheduler.step(f1)
        
        # En iyi modeli kaydet
        if f1 > best_f1:
            best_f1 = f1
            save_path = os.path.join(BASE_DIR, "models", "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"🎉 Yeni Rekor! Model kaydedildi: {save_path}")

    print("🏁 Eğitim Tamamlandı!")

if __name__ == "__main__":
    train()