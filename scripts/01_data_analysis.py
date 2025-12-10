"""
CAFA 6 - Protein Function Prediction
Script: 01_data_analysis.py
Author: [Senin Adın/Takım Adın]
Description: 
    Veri setindeki etiketlerin (GO Terms), türlerin (Taxonomy) 
    ve ontoloji yapısının analizini yapar.
    Bu çıktıları kullanarak alan uzmanlarına (Biyolog/Tıpçı) sorular soracağız.
"""
import os
import pandas as pd
import obonet
import networkx as nx

# --- KONFİGÜRASYON (Dosya Yolları) ---
# Proje kök dizinine göre ayarlanmıştır.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "Train")

FILES = {
    "terms": os.path.join(DATA_DIR, "train_terms.tsv"),
    "taxonomy": os.path.join(DATA_DIR, "train_taxonomy.tsv"),
    "obo": os.path.join(DATA_DIR, "go-basic.obo")
}

def analyze_terms():
    """Etiketlerin (Labels) dağılımını analiz eder."""
    print(f"\n{'='*20} 1. ETİKET (TERM) ANALİZİ {'='*20}")
    
    if not os.path.exists(FILES["terms"]):
        print("HATA: train_terms.tsv bulunamadı!")
        return

    df = pd.read_csv(FILES["terms"], sep="\t")
    
    print(f"-> Toplam Etiket Satırı: {len(df):,}")
    print(f"-> Benzersiz Protein Sayısı: {df['EntryID'].nunique():,}")
    print(f"-> Benzersiz GO Terimi Sayısı: {df['term'].nunique():,}")
    
    # Aspect (MF, BP, CC) Dağılımı
    print("\n[Alt Ontoloji (Aspect) Dağılımı]")
    # BPO: Biological Process, CCO: Cellular Component, MFO: Molecular Function
    print(df['aspect'].value_counts())
    
    return df

def analyze_taxonomy():
    """Türlerin (Species) dağılımını analiz eder."""
    print(f"\n{'='*20} 2. TÜR (TAXONOMY) ANALİZİ {'='*20}")
    
    if not os.path.exists(FILES["taxonomy"]):
        print("HATA: train_taxonomy.tsv bulunamadı!")
        return

    # Dosyayı oku
    df = pd.read_csv(FILES["taxonomy"], sep="\t")
    
    # --- HATA AYIKLAMA KISMI ---
    print(f"Tespit Edilen Kolon İsimleri: {df.columns.tolist()}")
    
    # Eğer 'taxonomyID' yoksa, muhtemelen 2. sütun Taxon ID'dir.
    # Kolon ismini otomatik bulalım:
    if 'taxonomyID' in df.columns:
        col_name = 'taxonomyID'
    elif len(df.columns) >= 2:
        col_name = df.columns[1] # 2. sütunu al
        print(f"⚠️ 'taxonomyID' bulunamadı. '{col_name}' sütunu kullanılıyor.")
    else:
        print("❌ HATA: Dosyada yeterli sütun yok!")
        return

    print(f"-> Toplam Protein Sayısı: {len(df):,}")
    print(f"-> Farklı Tür Sayısı: {df[col_name].nunique()}")
    
    print("\n[En Çok Verisi Olan İlk 10 Tür]")
    top_10 = df[col_name].value_counts().head(10)
    print(top_10)

def analyze_ontology():
    """GO (Gene Ontology) Ağaç yapısını analiz eder."""
    print(f"\n{'='*20} 3. ONTOLOJİ (HİYERARŞİ) ANALİZİ {'='*20}")
    
    if not os.path.exists(FILES["obo"]):
        print("HATA: go-basic.obo bulunamadı! Lütfen indirin.")
        return

    print("-> OBO dosyası yükleniyor (biraz sürebilir)...")
    graph = obonet.read_obo(FILES["obo"])
    
    print(f"-> Graf üzerindeki Düğüm (Node) Sayısı: {len(graph):,}")
    print(f"-> Kenar (Edge/İlişki) Sayısı: {graph.number_of_edges():,}")
    
    # Örnek: 'is_a' ilişkisi kontrolü
    print("\n[Hiyerarşi Örneği]")
    if 'GO:0005575' in graph: # Cellular Component Root
        print(f"Root (GO:0005575): {graph.nodes['GO:0005575']['name']}")
        
    return graph

if __name__ == "__main__":
    analyze_terms()
    analyze_taxonomy()
    analyze_ontology()
    print(f"\n{'='*20} ANALİZ TAMAMLANDI {'='*20}")