"""
CAFA 6 - Protein Function Prediction
Script: 02_extract_embeddings.py
Description: 
    Protein dizilerini (FASTA) okur ve ESM-2 modelini kullanarak 
    sayısal vektörlere (Embedding) dönüştürür.
    Çıktıları 'input' klasörüne kaydeder.
"""

import torch
import pandas as pd
import numpy as np
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os
import gc

# --- AYARLAR ---
# Model: ESM-2 650M (Dengeli ve güçlü)
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"

# Donanım Ayarları (RTX 4060 için optimize)
BATCH_SIZE = 8  # Hafıza hatası alırsan 4'e düşür
MAX_LEN = 1024  # ESM-2 maksimum uzunluğu

# Dosya Yolları (Otomatik Ayarlı)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "Train")
OUTPUT_DIR = os.path.join(BASE_DIR, "input")

# Çıktı klasörünü oluştur
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GPU Kontrolü
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Kullanılan Cihaz: {device}")
if device.type == 'cuda':
    print(f"   Kart: {torch.cuda.get_device_name(0)}")
    torch_dtype = torch.float16 # Hız için
else:
    print("⚠️ DİKKAT: GPU yok! Bu işlem CPU ile günlerce sürebilir!")
    torch_dtype = torch.float32

def load_fasta(file_path):
    """FASTA dosyasını okur ve listeye çevirir."""
    if not os.path.exists(file_path):
        print(f"❌ Dosya bulunamadı: {file_path}")
        return []
        
    print(f"📂 Dosya okunuyor: {os.path.basename(file_path)}")
    data = []
    for record in SeqIO.parse(file_path, "fasta"):
        # ID temizleme (sp|P12345|NAME -> P12345)
        if "|" in record.id:
            clean_id = record.id.split('|')[1]
        else:
            clean_id = record.id
        data.append({"id": clean_id, "seq": str(record.seq)})
    
    return pd.DataFrame(data)

def extract_embeddings(df, prefix):
    """Modeli çalıştırır ve embeddingleri kaydeder."""
    print(f"🚀 Model Yükleniyor: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch_dtype).to(device)
    model.eval()

    sequences = df['seq'].tolist()
    ids = df['id'].values
    embeddings = []
    
    print(f"📊 İşlenecek Protein Sayısı: {len(sequences)}")
    
    # Batch Processing
    for i in tqdm(range(0, len(sequences), BATCH_SIZE), desc="Embedding Çıkarılıyor"):
        batch_seqs = sequences[i:i + BATCH_SIZE]
        
        # Tokenize (Max 1024 uzunluk)
        inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # (Batch, Seq, Dim) -> (Batch, Dim) : Ortalama alıyoruz
            batch_emb = outputs.last_hidden_state.mean(dim=1)
            
        embeddings.append(batch_emb.to(torch.float32).cpu().numpy())
        
        del inputs, outputs, batch_emb
        torch.cuda.empty_cache()

    # Birleştir ve Kaydet
    final_emb = np.vstack(embeddings)
    
    emb_path = os.path.join(OUTPUT_DIR, f"{prefix}_embeddings_650M.npy")
    id_path = os.path.join(OUTPUT_DIR, f"{prefix}_ids_650M.npy")
    
    np.save(emb_path, final_emb)
    np.save(id_path, ids)
    
    print(f"✅ KAYDEDİLDİ:")
    print(f"   -> {emb_path}")
    print(f"   -> {id_path}")
    print(f"   -> Boyut: {final_emb.shape}")
    
    del model, tokenizer, final_emb
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # 1. Eğitim Verisi İçin
    train_fasta = os.path.join(DATA_DIR, "train_sequences.fasta")
    df_train = load_fasta(train_fasta)
    
    if len(df_train) > 0:
        # Test amaçlı şimdilik ilk 1000 tanesini yapabilirsin. 
        # Hepsini yapmak için aşağıdaki [:1000] kısmını sil!
        # df_train = df_train[:1000] 
        extract_embeddings(df_train, "train")
    
    print("\n🎉 İŞLEM TAMAMLANDI! Şimdi data_loader.py çalışacaktır.")