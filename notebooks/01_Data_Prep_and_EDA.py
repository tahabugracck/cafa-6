import pandas as pd
from Bio import SeqIO
import os

# --- PATH AYARLARI (GÜNCELLENDİ) ---
# Bu script'in nerede olduğunu dinamik olarak buluyoruz
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # notebooks klasörünün yolu
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)              # Bir üst klasör (Proje ana dizini)
INPUT_DIR = os.path.join(PROJECT_ROOT, "input")         # input klasörünün tam yolu

# Dosya yollarını tam yol (absolute path) olarak oluşturuyoruz
TRAIN_FASTA = os.path.join(INPUT_DIR, "train_sequences.fasta")
TRAIN_TERMS = os.path.join(INPUT_DIR, "train_terms.tsv")

print(f"Çalışma Dizini: {PROJECT_ROOT}")
print(f"Okunacak Fasta Yolu: {TRAIN_FASTA}")

# --- BURADAN SONRASI AYNI ---

def load_fasta(file_path):
    """FASTA dosyasını okuyup Pandas DataFrame'e çevirir."""
    print(f"Loading {os.path.basename(file_path)}...")
    data = []
    for record in SeqIO.parse(file_path, "fasta"):
        # HATA DÜZELTME: Header'ı '|' karakterine göre bölüp 2. parçayı alıyoruz.
        # Örnek: "sp|A0A0C5B5G6|MOTSC_HUMAN" -> "A0A0C5B5G6"
        clean_id = record.id.split('|')[1]
        
        data.append({
            "id": clean_id,
            "seq": str(record.seq),
            "len": len(record.seq)
        })
    return pd.DataFrame(data)

# 1. Protein Dizilerini Oku
if os.path.exists(TRAIN_FASTA):
    df_train = load_fasta(TRAIN_FASTA)
    print(f"Toplam Protein Sayısı: {len(df_train)}")
    print(df_train.head(3))
else:
    print(f"HATA: Dosya bulunamadı -> {TRAIN_FASTA}")
    exit()

# 2. Etiketleri Oku
if os.path.exists(TRAIN_TERMS):
    print(f"\nLoading {os.path.basename(TRAIN_TERMS)}...")
    df_terms = pd.read_csv(TRAIN_TERMS, sep="\t")
    print(f"Toplam Etiket Kaydı: {len(df_terms)}")
    print(df_terms.head(3))
else:
    print(f"HATA: Dosya bulunamadı -> {TRAIN_TERMS}")
    exit()

# 3. İstatistiklere Bakalım
print("\n--- Özet Bilgiler ---")
print(f"Benzersiz Protein Sayısı (Terms dosyasında): {df_terms['EntryID'].nunique()}")
print("Ontoloji Dağılımı:")
print(df_terms['aspect'].value_counts())

# 4. Veriyi Birleştirme (Opsiyonel Kontrol)
ids_in_fasta = set(df_train['id'])
ids_in_terms = set(df_terms['EntryID'])
print(f"\nFasta dosyasında olup etiketi olmayan protein var mı? : {len(ids_in_fasta - ids_in_terms)}")