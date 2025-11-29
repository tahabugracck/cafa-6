import pandas as pd
import os
from tqdm import tqdm

# --- AYARLAR ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INPUT_PATH = os.path.join(PROJECT_ROOT, "output", "submissions", "submission_v2_propagation.tsv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "submissions", "submission.tsv") # Kaggle'ın istediği isim

MAX_TERMS_PER_PROTEIN = 1500  # Yarışma kuralı

def filter_and_save():
    print(f"Dosya okunuyor (Bu biraz sürebilir, 40M satır)...")
    # Sütun isimlerini vererek okuyoruz
    df = pd.read_csv(INPUT_PATH, sep="\t", names=["ProteinID", "GO_Term", "Score"])
    
    print(f"Toplam Satır: {len(df)}")
    
    # 1. Sıralama: Her protein grubu içinde Skora göre büyükten küçüğe sırala
    # (Burası RAM'i biraz zorlayabilir, o yüzden performant bir yöntem kullanıyoruz)
    print("Sıralanıyor ve Filtreleniyor...")
    
    # En yüksek puanlıları alacağız
    df_sorted = df.sort_values(['ProteinID', 'Score'], ascending=[True, False])
    
    # Her ProteinID grubu için ilk 1500 satırı al
    df_final = df_sorted.groupby('ProteinID').head(MAX_TERMS_PER_PROTEIN)
    
    print(f"Filtreleme Sonrası Satır Sayısı: {len(df_final)}")
    
    # 2. Kaydetme
    print(f"Kaydediliyor: {os.path.basename(OUTPUT_PATH)}")
    df_final.to_csv(OUTPUT_PATH, sep="\t", header=False, index=False)
    
    print("✅ İŞLEM TAMAM! Kaggle'a yüklemeye hazır.")

if __name__ == "__main__":
    if os.path.exists(INPUT_PATH):
        filter_and_save()
    else:
        print("HATA: v2 dosyası bulunamadı!")