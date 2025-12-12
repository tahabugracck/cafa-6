import pandas as pd
import numpy as np
import os
import sys

# Dosya Yolları
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_SUBMISSION = os.path.join(BASE_DIR, "output", "submission_specialist.tsv")
IA_FILE = os.path.join(BASE_DIR, "data", "Train", "IA.tsv")
OUTPUT_FILE = os.path.join(BASE_DIR, "output", "submission_final.tsv")

# Yapılandırma
TOP_K = 70  # Her protein için en iyi K tahmini sakla

def apply_dynamic_threshold():
    print("Dinamik Eşikleme Başlıyor (IA Ağırlıklı)...")
    
    # 1. Doğrulama
    if not os.path.exists(INPUT_SUBMISSION):
        print(f"Hata: Girdi dosyası {INPUT_SUBMISSION} bulunamadı.")
        return
    if not os.path.exists(IA_FILE):
        print(f"Hata: IA dosyası {IA_FILE} bulunamadı.")
        return

    # 2. IA (Bilgi Kazanımı) Ağırlıklarını Yükle
    print("Bilgi Kazanımı ağırlıkları yükleniyor...")
    try:
        ia_df = pd.read_csv(IA_FILE, sep="\t", names=["Term", "IA"])
    except Exception as e:
        print(f"IA dosyası okuma hatası: {e}")
        return
    
    # 3. Gönderim Dosyasını Yükle
    print("Submission dosyası yükleniyor...")
    df = pd.read_csv(INPUT_SUBMISSION, sep="\t", header=None, names=["ID", "Term", "Score"], dtype={"Score": float})
    
    # 4. Verileri Birleştir
    merged = pd.merge(df, ia_df, on="Term", how="left")
    # Eksik IA değerlerini varsayılan (1.0) ile doldur
    merged["IA"] = merged["IA"].fillna(1.0)
    
    print(f"Orijinal satır sayısı: {len(merged):,}")
    
    # 5. Dinamik Mantık Uygula
    # Strateji:
    # - Yüksek IA (Nadir/Değerli): Güven düşük olsa bile sakla.
    # - Düşük IA (Yaygın): Sadece güven yüksekse sakla.
    
    mask_valuable = (merged["IA"] >= 2.0) & (merged["Score"] >= 0.005)
    mask_common   = (merged["IA"] < 0.5) & (merged["Score"] >= 0.15)
    mask_normal   = (merged["IA"] >= 0.5) & (merged["IA"] < 2.0) & (merged["Score"] >= 0.01)
    
    final_df = merged[mask_valuable | mask_common | mask_normal]
    
    print(f"Filtrelenmiş satır sayısı: {len(final_df):,}")
    
    # 6. Top-K Filtreleme
    # Dosya boyutunu küçültmek ve gürültüyü kaldırmak için protein başına en iyi tahminleri sakla
    print(f"Protein başına Top-{TOP_K} filtresi uygulanıyor...")
    final_df = final_df[["ID", "Term", "Score"]]
    final_df = final_df.sort_values(["ID", "Score"], ascending=[True, False])
    final_df = final_df.groupby("ID").head(TOP_K)
    
    # 7. Final Dosyayı Kaydet
    print(f"Final dosya şuraya kaydediliyor: {OUTPUT_FILE}")
    final_df.to_csv(OUTPUT_FILE, sep="\t", header=False, index=False)
    
    print("Son işleme başarıyla tamamlandı.")

if __name__ == "__main__":
    apply_dynamic_threshold()