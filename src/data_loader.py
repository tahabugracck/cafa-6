"""
CAFA 6 - Protein Function Prediction
Module: src/data_loader.py
Description: 
    Eğitim için gerekli verileri yükler. 
    Otomatik olarak proje ana dizinini bulur ve dosya yollarını ayarlar.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import sys

class CAFA6Dataset(Dataset):
    def __init__(self, embeddings_path, ids_path, labels_df, taxonomy_df, num_classes, label_map):
        """
        PyTorch Dataset Sınıfı.
        Model eğitimi sırasında verileri parça parça (batch) RAM'e yükler.
        """
        # 1. Embeddingleri Yükle
        print(f"📦 Embeddingler yükleniyor: {os.path.basename(embeddings_path)}...")
        try:
            self.embeddings = np.load(embeddings_path)
            self.ids = np.load(ids_path)
        except FileNotFoundError:
            print(f"❌ HATA: Embedding dosyası bulunamadı: {embeddings_path}")
            sys.exit(1)
            
        # ID'den Index'e hızlı erişim sözlüğü
        self.id_to_idx = {pid: i for i, pid in enumerate(self.ids)}
        
        self.labels_df = labels_df
        self.taxonomy_df = taxonomy_df
        self.num_classes = num_classes
        self.label_map = label_map
        
        # Hem Embedding'i olan hem de Etiketi olan proteinleri bul (Kesişim Kümesi)
        valid_ids = set(self.ids) & set(labels_df['EntryID'].values)
        self.valid_ids_list = list(valid_ids)
        
        print(f"✅ Eşleşen ve Eğitime Hazır Protein Sayısı: {len(self.valid_ids_list)}")

    def __len__(self):
        return len(self.valid_ids_list)

    def __getitem__(self, idx):
        # 1. Protein ID'sini al
        entry_id = self.valid_ids_list[idx]
        
        # 2. Embedding Vektörünü bul
        emb_idx = self.id_to_idx[entry_id]
        embedding = self.embeddings[emb_idx]
        
        # 3. Etiketleri (Labels) Bul ve Vektöre Çevir (Multi-Hot Encoding)
        # Örn: [0, 1, 0, 0, 1, ...]
        protein_terms = self.labels_df[self.labels_df['EntryID'] == entry_id]['term'].values
        
        label_vector = torch.zeros(self.num_classes, dtype=torch.float32)
        for term in protein_terms:
            if term in self.label_map:
                label_idx = self.label_map[term]
                label_vector[label_idx] = 1.0
        
        # 4. Taksonomi bilgisini al
        # Eğer protein listede yoksa varsayılan olarak 0 veriyoruz
        taxon_id = 0
        if entry_id in self.taxonomy_df.index:
            taxon_id = self.taxonomy_df.loc[entry_id, 'TaxonID']
            
        return {
            "embedding": torch.tensor(embedding, dtype=torch.float32),
            "labels": label_vector,
            "taxon_id": torch.tensor(taxon_id, dtype=torch.long),
            "entry_id": entry_id
        }

def load_data(data_dir, top_n_labels=1500):
    """
    Ham tsv dosyalarını okur ve temizler.
    """
    terms_path = os.path.join(data_dir, "train_terms.tsv")
    tax_path = os.path.join(data_dir, "train_taxonomy.tsv")
    
    # 1. Etiketleri Oku
    if not os.path.exists(terms_path):
        raise FileNotFoundError(f"train_terms.tsv bulunamadı: {terms_path}")
        
    print("📂 Etiketler okunuyor ve işleniyor...")
    df_terms = pd.read_csv(terms_path, sep="\t")
    
    # En sık geçen N etiketi seç (Burası modelin ne kadar geniş öğreneceğini belirler)
    top_terms = df_terms['term'].value_counts().head(top_n_labels).index
    label_map = {term: i for i, term in enumerate(top_terms)}
    
    # Sadece seçilen etiketleri içeren veriyi tut
    df_terms_filtered = df_terms[df_terms['term'].isin(top_terms)]
    
    # 2. Taksonomi Oku (Header Fix Uygulandı)
    print("🧬 Taksonomi bilgisi okunuyor...")
    if os.path.exists(tax_path):
        # Header yoksa, header=None deyip isimleri biz veriyoruz
        df_tax = pd.read_csv(tax_path, sep="\t", header=None, names=["EntryID", "TaxonID"])
        df_tax = df_tax.set_index("EntryID")
    else:
        print("⚠️ UYARI: train_taxonomy.tsv bulunamadı, taksonomi özelliği kullanılmayacak.")
        df_tax = pd.DataFrame()
    
    return df_terms_filtered, df_tax, label_map

# --- TEST BLOĞU ---
if __name__ == "__main__":
    # Proje kök dizinini bul (src klasörünün bir üstü)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Dosya Yolları
    DATA_DIR = os.path.join(BASE_DIR, "data", "Train")
    
    # Embeddinglerin olduğu klasör (Genelde 'input' veya kök dizindedir)
    # NOT: Embeddingleri nereye kaydettiysen burayı kontrol et!
    EMB_DIR = os.path.join(BASE_DIR, "input") 
    
    EMB_FILE = os.path.join(EMB_DIR, "train_embeddings_650M.npy")
    ID_FILE = os.path.join(EMB_DIR, "train_ids_650M.npy")
    
    print(f"📍 Çalışma Dizini: {BASE_DIR}")
    
    if os.path.exists(EMB_FILE):
        terms, tax, l_map = load_data(DATA_DIR, top_n_labels=100)
        
        dataset = CAFA6Dataset(
            embeddings_path=EMB_FILE,
            ids_path=ID_FILE,
            labels_df=terms,
            taxonomy_df=tax,
            num_classes=100,
            label_map=l_map
        )
        
        # İlk veriyi test et
        sample = dataset[0]
        print("\n🎉 BAŞARILI! Örnek Veri:")
        print(f"   Protein ID: {sample['entry_id']}")
        print(f"   Vektör Boyutu: {sample['embedding'].shape}")
        print(f"   Taxon ID: {sample['taxon_id']}")
        print(f"   Aktif Etiket Sayısı: {sample['labels'].sum().item()}")
    else:
        print("\n❌ HATA: Embedding dosyaları bulunamadı!")
        print(f"Aranan yol: {EMB_FILE}")
        print("Lütfen önce 'extract_embeddings_650M.py' kodunu çalıştırın ve 'input' klasörünü kontrol edin.")