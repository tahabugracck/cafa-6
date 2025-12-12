import torch
import pandas as pd
import numpy as np
import sys
import os
import networkx
import obonet
from tqdm import tqdm

# Proje Kök Dizini Ayarı
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.model import CAFA6Model

# Yapılandırma
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2048

# Dosya Yolları
INPUT_DIR = os.path.join(BASE_DIR, "input")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

TEST_EMB_FILE = os.path.join(INPUT_DIR, "test_embeddings_650M.npy")
TEST_ID_FILE = os.path.join(INPUT_DIR, "test_ids_650M.npy")
TAXON_FILE = os.path.join(DATA_DIR, "Test", "testsuperset-taxon-list.tsv")
TRAIN_TAXON_FILE = os.path.join(DATA_DIR, "Train", "train_taxonomy.tsv")
OBO_FILE = os.path.join(DATA_DIR, "Train", "go-basic.obo")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "submission_specialist.tsv")

# Eğitim sırasında kullanılan aspect ve sınıf sayıları
ASPECTS = {
    "F": 2000,
    "C": 1500,
    "P": 3000
}

def generate_ensemble():
    print("Topluluk (Ensemble) Çıkarımı Başlıyor...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Test Verisini Yükle
    if not os.path.exists(TEST_EMB_FILE) or not os.path.exists(TEST_ID_FILE):
        raise FileNotFoundError("Test yerleştirmeleri bulunamadı. Lütfen önce embedding çıkarma betiğini çalıştırın.")

    test_embs = np.load(TEST_EMB_FILE)
    test_ids = np.load(TEST_ID_FILE)
    
    # 2. Taksonomi Haritalarını Hazırla
    # Test Taksonomisini Yükle
    try:
        test_tax_df = pd.read_csv(TAXON_FILE, sep="\t", header=None, names=["ID", "TaxonID"])
        test_tax_df.set_index("ID", inplace=True)
    except Exception as e:
        print(f"Uyarı: Test taksonomisi yüklenemedi ({e}). Varsayılan olarak 0 atanıyor.")
        test_tax_df = pd.DataFrame()

    # Takson ID'lerini doğru şekilde eşlemek için Eğitim Taksonomisini Yükle
    if os.path.exists(TRAIN_TAXON_FILE):
        train_tax = pd.read_csv(TRAIN_TAXON_FILE, sep="\t", header=None, names=["ID", "TaxonID"])
        unique_taxons = train_tax['TaxonID'].unique()
        taxon_map = {tid: i+1 for i, tid in enumerate(unique_taxons)}
        num_taxons = len(unique_taxons) + 1
    else:
        raise FileNotFoundError("Eğitim taksonomi dosyası eksik. Takson ID'leri eşlenemiyor.")

    all_predictions = []

    # 3. Her Uzman Model İçin Çıkarım Döngüsü
    for aspect, num_classes in ASPECTS.items():
        print(f"Aspect işleniyor: {aspect}...")
        
        model_path = os.path.join(MODEL_DIR, f"best_model_{aspect}.pth")
        map_path = os.path.join(MODEL_DIR, f"label_map_{aspect}.npy")
        
        if not os.path.exists(model_path):
            print(f"Atlanıyor {aspect}: Model dosyası bulunamadı.")
            continue

        # Etiket Haritasını ve Modeli Yükle
        label_map = np.load(map_path, allow_pickle=True).item()
        idx_to_term = {v: k for k, v in label_map.items()}
        
        model = CAFA6Model(num_classes=num_classes, num_taxons=num_taxons).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        
        # Toplu Tahmin (Batch Prediction)
        for i in tqdm(range(0, len(test_ids), BATCH_SIZE), desc=f"Çıkarım {aspect}"):
            batch_embs = torch.tensor(test_embs[i:i+BATCH_SIZE], dtype=torch.float32).to(DEVICE)
            batch_ids = test_ids[i:i+BATCH_SIZE]
            
            # Takson ID'lerini eşle
            batch_taxons = []
            for pid in batch_ids:
                if pid in test_tax_df.index:
                    val = test_tax_df.loc[pid, "TaxonID"]
                    tid = val.iloc[0] if isinstance(val, pd.Series) else val
                    batch_taxons.append(taxon_map.get(tid, 0))
                else:
                    batch_taxons.append(0)
            
            batch_taxons = torch.tensor(batch_taxons, dtype=torch.long).to(DEVICE)
            
            with torch.no_grad():
                logits = model(batch_embs, batch_taxons)
                probs = torch.sigmoid(logits).cpu().numpy()
            
            # Filtrele ve Sakla
            for j, pid in enumerate(batch_ids):
                p_probs = probs[j]
                # 'P' için daha düşük eşik değeri (daha yüksek karmaşıklık nedeniyle)
                threshold = 0.005 if aspect == 'P' else 0.01
                
                indices = np.where(p_probs > threshold)[0]
                for idx in indices:
                    all_predictions.append(f"{pid}\t{idx_to_term[idx]}\t{p_probs[idx]:.3f}")

    # Geçici Ham Tahminleri Yaz
    temp_file = os.path.join(OUTPUT_DIR, "temp_submission.tsv")
    with open(temp_file, "w") as f:
        f.write("\n".join(all_predictions))
        
    # 4. Son İşleme: Ontoloji Yayılımı
    print("Ontoloji Yayılımı Uygulanıyor (Hiyerarşi Kuralları)...")
    if not os.path.exists(OBO_FILE):
         print("Hata: OBO dosyası bulunamadı. Yayılım atlanıyor.")
         return

    graph = obonet.read_obo(OBO_FILE)
    df = pd.read_csv(temp_file, sep="\t", header=None, names=["ProteinID", "Term", "Score"])
    
    final_rows = []
    
    # Puanları alt sınıflardan üst sınıflara yay
    for protein_id, group in tqdm(df.groupby("ProteinID"), desc="Yayılıyor"):
        scores = dict(zip(group["Term"], group["Score"]))
        terms = list(scores.keys())
        
        for term in terms:
            if term not in graph: continue
            current = scores[term]
            
            try:
                # Ataları (ebeveynleri) al
                ancestors = networkx.descendants(graph, term)
            except: 
                continue
            
            for anc in ancestors:
                if anc in scores:
                    scores[anc] = max(scores[anc], current)
                else:
                    scores[anc] = current
        
        # Dosya boyutunu yönetilebilir tutmak için son filtreleme
        for term, score in scores.items():
            if score >= 0.01:
                final_rows.append(f"{protein_id}\t{term}\t{score:.3f}")
                
    # Son Dosyayı Kaydet
    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(final_rows))
        
    print(f"Çıkarım Tamamlandı. Çıktı şuraya kaydedildi: {OUTPUT_FILE}")
    
    # Temizlik
    if os.path.exists(temp_file):
        os.remove(temp_file)

if __name__ == "__main__":
    generate_ensemble()