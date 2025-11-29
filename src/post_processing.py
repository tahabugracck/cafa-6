import pandas as pd
import networkx as nx
import obonet
import os
from tqdm import tqdm

# --- PATH AYARLARI ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INPUT_DIR = os.path.join(PROJECT_ROOT, "input")
SUBMISSION_DIR = os.path.join(PROJECT_ROOT, "output", "submissions")

# Girdi ve Çıktı Dosyaları
OBO_PATH = os.path.join(INPUT_DIR, "go-basic.obo")
INPUT_SUB_PATH = os.path.join(SUBMISSION_DIR, "submission_v1_baseline.tsv")
OUTPUT_SUB_PATH = os.path.join(SUBMISSION_DIR, "submission_v2_propagation.tsv")

def propagate_scores():
    print("1. GO Ontolojisi (Graph) yükleniyor...")
    # OBO dosyasını graph olarak oku
    graph = obonet.read_obo(OBO_PATH)
    
    # NetworkX graph'ı, edge'lerin yönü Child -> Parent şeklindedir.
    # Bu yüzden 'is_a' ilişkilerini takip edeceğiz.
    
    print("2. Mevcut Submission dosyası okunuyor...")
    # Header yok, sütun isimlerini biz veriyoruz
    df = pd.read_csv(INPUT_SUB_PATH, sep="\t", names=["ProteinID", "GO_Term", "Score"])
    
    print(f"   Orijinal tahmin sayısı: {len(df)}")
    
    # İşlemleri hızlandırmak için veriyi Dictionary formatına çeviriyoruz
    # {ProteinID: {Term: Score, Term2: Score2...}}
    protein_scores = {}
    
    print("3. Veri hazırlanıyor...")
    for pid, term, score in tqdm(df.values, desc="Mapping"):
        if pid not in protein_scores:
            protein_scores[pid] = {}
        protein_scores[pid][term] = score

    new_predictions = []
    
    print("4. Hiyerarşi Yayılımı (Propagation) yapılıyor...")
    # Her protein için döngü
    for pid, term_dict in tqdm(protein_scores.items(), desc="Propagating"):
        
        # O proteinin tahmin edilen tüm terimleri için
        # O anki terimlerin kopyasını alalım (iterasyon sırasında sözlük değişmesin diye)
        current_terms = list(term_dict.keys())
        
        for term in current_terms:
            score = term_dict[term]
            
            # Eğer terim grafikte yoksa (nadir durum), atla
            if term not in graph:
                continue
                
            # Bu terimin tüm atalarını (parents/ancestors) bul
            # networkx.ancestors, o node'dan gidilebilecek tüm node'ları getirir
            ancestors = nx.ancestors(graph, term)
            
            for ancestor in ancestors:
                # Ebeveyn skoru, çocuğun skoru kadar olmalı (en az)
                # Eğer ebeveyn zaten listede varsa ve skoru yüksekse dokunma
                # Eğer yoksa veya düşükse, çocuğun skorunu ona ver.
                if ancestor in term_dict:
                    term_dict[ancestor] = max(term_dict[ancestor], score)
                else:
                    term_dict[ancestor] = score
        
        # Güncellenmiş sözlüğü listeye ekle
        for term, final_score in term_dict.items():
            new_predictions.append([pid, term, final_score])

    print("5. Yeni dosya kaydediliyor...")
    df_new = pd.DataFrame(new_predictions, columns=["ProteinID", "GO_Term", "Score"])
    
    # Skorları yuvarla (Dosya boyutu için)
    df_new['Score'] = df_new['Score'].round(3)
    
    # Tekrar edenleri temizle (Garanti olsun)
    df_new = df_new.drop_duplicates()
    
    print(f"   Yeni tahmin sayısı: {len(df_new)}")
    
    # Kaydet
    df_new.to_csv(OUTPUT_SUB_PATH, sep="\t", header=False, index=False)
    print(f"✅ BİTTİ! Dosya: {OUTPUT_SUB_PATH}")

if __name__ == "__main__":
    if os.path.exists(INPUT_SUB_PATH):
        propagate_scores()
    else:
        print(f"HATA: Girdi dosyası bulunamadı -> {INPUT_SUB_PATH}")
        print("Lütfen önce v1 submission dosyasını oluşturun.")