import torch
import pandas as pd
import numpy as np
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

# --- AYARLAR ---
# Scriptin çalıştığı yerden proje ana dizinini bul
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INPUT_DIR = os.path.join(PROJECT_ROOT, "input")

# Model Ayarları (Local CPU için optimize)
MODEL_NAME = "facebook/esm2_t6_8M_UR50D" # En küçük model (Local dostu)
BATCH_SIZE = 8 # CPU'da RAM şişmemesi için düşük tutuyoruz
DEVICE = "cpu" # Senin bilgisayarın için CPU modu

print(f"Çalışma Dizini: {PROJECT_ROOT}")
print(f"Kullanılan Cihaz: {DEVICE} (i9 işlemcin kullanılacak)")

# 1. Veriyi Yükle ve Temizle
def load_data():
    fasta_path = os.path.join(INPUT_DIR, "train_sequences.fasta")
    print(f"Veri okunuyor: {fasta_path}")
    
    data = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        # ID düzeltme (Local fix'imiz)
        clean_id = record.id.split('|')[1]
        data.append({"id": clean_id, "seq": str(record.seq)})
        
    return pd.DataFrame(data)

# 2. Embedding Fonksiyonu
def extract_embeddings(df):
    print(f"Model yükleniyor: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    
    sequences = df['seq'].tolist()
    embeddings = []
    
    print(f"Toplam {len(sequences)} protein işlenecek. Bu işlem biraz zaman alabilir...")
    
    # Batch işlemi
    for i in tqdm(range(0, len(sequences), BATCH_SIZE)):
        batch_seqs = sequences[i:i+BATCH_SIZE]
        
        # Tokenize (CPU'yu yormamak için max_length'i biraz kıstık, çoğu protein sığar)
        inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Sequence (Seq_len, 320) -> Mean Pooling -> (1, 320)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            
        embeddings.append(batch_embeddings.numpy()) # CPU olduğu için .cpu() gerekmez ama zararı yok
        
    return np.vstack(embeddings)

if __name__ == "__main__":
    # Veriyi oku
    df = load_data()
    print(f"Toplam Protein: {len(df)}")
    
    # Embedding çıkar
    emb_matrix = extract_embeddings(df)
    
    # Kaydet
    output_emb_path = os.path.join(INPUT_DIR, "train_embeddings.npy")
    output_id_path = os.path.join(INPUT_DIR, "train_ids.npy")
    
    print("Dosyalar kaydediliyor...")
    np.save(output_emb_path, emb_matrix)
    np.save(output_id_path, df['id'].values)
    
    print(f"BİTTİ! Dosyalar şurada: {INPUT_DIR}")
    print(f"Matris Boyutu: {emb_matrix.shape}")