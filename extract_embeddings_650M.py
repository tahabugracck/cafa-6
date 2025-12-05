import torch
import pandas as pd
import numpy as np
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os
import gc

# --- AYARLAR ---
# Büyük Model (650 Milyon Parametre)
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"

# RTX 4060 (8GB VRAM) için güvenli ayarlar
# Eğer "Out of Memory" hatası verirse bunu 4'e düşürün.
BATCH_SIZE = 8 

# Klasör Yolları
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input")

# --- GPU OPTİMİZASYONU ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Kullanılan Cihaz: {device}")

if device.type == 'cuda':
    print(f"   Ekran Kartı: {torch.cuda.get_device_name(0)}")
    # RTX serisi için FP16 (Yarım Hassasiyet) hızlandırması
    torch_dtype = torch.float16 
else:
    print("⚠️ UYARI: GPU bulunamadı! CPU ile bu işlem günler sürer.")
    torch_dtype = torch.float32

def load_fasta(filename):
    path = os.path.join(INPUT_DIR, filename)
    print(f"📂 Okunuyor: {filename}")
    
    if not os.path.exists(path):
        print(f"❌ HATA: Dosya bulunamadı -> {path}")
        return []

    data = []
    for record in SeqIO.parse(path, "fasta"):
        # ID temizleme
        if "|" in record.id:
            clean_id = record.id.split('|')[1]
        else:
            clean_id = record.id.split()[0]
        data.append({"id": clean_id, "seq": str(record.seq)})
    
    return pd.DataFrame(data)

def extract_embeddings(df, prefix):
    print(f"🚀 Model Yükleniyor: {MODEL_NAME}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch_dtype).to(device)
    model.eval()

    sequences = df['seq'].tolist()
    embeddings = []
    
    print(f"📊 Toplam {len(sequences)} protein işlenecek. (Batch Size: {BATCH_SIZE})")
    
    # Progress Bar ile döngü
    for i in tqdm(range(0, len(sequences), BATCH_SIZE)):
        batch_seqs = sequences[i:i + BATCH_SIZE]
        
        # Tokenize
        inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Embeddings (Batch, Seq_Len, 1280) -> Mean -> (Batch, 1280)
            batch_emb = outputs.last_hidden_state.mean(dim=1)
            
        # GPU'dan CPU'ya alıp listeye ekle (RAM tasarrufu)
        embeddings.append(batch_emb.to(torch.float32).cpu().numpy())
        
        # GPU belleğini temizle (Garanti olsun)
        del inputs, outputs, batch_emb
        torch.cuda.empty_cache()

    # Birleştir
    final_emb = np.vstack(embeddings)
    print(f"✅ Tamamlandı! Matris Boyutu: {final_emb.shape}")
    
    # Kaydet
    output_emb_file = os.path.join(INPUT_DIR, f"{prefix}_embeddings_650M.npy")
    output_id_file = os.path.join(INPUT_DIR, f"{prefix}_ids_650M.npy")
    
    np.save(output_emb_file, final_emb)
    np.save(output_id_file, df['id'].values)
    print(f"💾 Dosyalar kaydedildi: {output_emb_file}")
    
    # Bellek temizliği
    del model, tokenizer, final_emb
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # 1. Train Verisi
    #df_train = load_fasta("train_sequences.fasta")
    #if len(df_train) > 0:
    #    extract_embeddings(df_train, "train")
    
    #print("-" * 30)
    
    # 2. Test Verisi
    df_test = load_fasta("testsuperset.fasta")
    if len(df_test) > 0:
        extract_embeddings(df_test, "test")
        
    print("\n🎉 TÜM İŞLEMLER BİTTİ!")