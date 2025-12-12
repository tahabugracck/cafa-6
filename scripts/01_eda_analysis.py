import os
import pandas as pd
import sys

# Proje kök dizinini belirle
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "Train")

def analyze_data():
    print("Veri Analizi Başlıyor...")
    terms_path = os.path.join(DATA_DIR, "train_terms.tsv")
    tax_path = os.path.join(DATA_DIR, "train_taxonomy.tsv")

    if os.path.exists(terms_path):
        df = pd.read_csv(terms_path, sep="\t")
        print(f"-> Toplam Etiket: {len(df)}")
        print(f"-> Benzersiz Protein: {df['EntryID'].nunique()}")
        print("-> Kategori Dağılımı:")
        print(df['aspect'].value_counts())
    else:
        print("train_terms.tsv bulunamadı.")

    if os.path.exists(tax_path):
        # Başlık yoksa sütun adlarını manuel olarak ata
        df_tax = pd.read_csv(tax_path, sep="\t", header=None, names=["ID", "TaxonID"])
        print(f"-> Toplam Tür Sayısı: {df_tax['TaxonID'].nunique()}")
        print("-> En Sık Görülen 5 Tür:")
        print(df_tax['TaxonID'].value_counts().head(5))

if __name__ == "__main__":
    analyze_data()