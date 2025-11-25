# CAFA 6: Protein Function Prediction Project

Bu proje, Kaggle üzerinde düzenlenen **CAFA 6 (Critical Assessment of Functional Annotation)** yarışması için geliştirilmiş bir yapay zeka çözümüdür. Amaç, proteinlerin amino asit dizilerini (sequences) kullanarak biyolojik işlevlerini (Gene Ontology Terms) tahmin etmektir.

## 🎯 Proje Hedefi

Proteinler için üç ana ontolojide tahmin yapmak:

1.  **Molecular Function (MF):** Moleküler aktivite.
2.  **Biological Process (BP):** Dahil olunan biyolojik süreç.
3.  **Cellular Component (CC):** Hücresel konum.

## 📂 Klasör Yapısı

- **input/**: Ham veriler ve oluşturulan embeddingler (GitHub'da takip edilmez).
- **notebooks/**: Veri analizi ve model eğitim not defterleri.
- **src/**: Yardımcı Python scriptleri (Feature extraction vb.).
- **output/**: Model çıktıları ve submission dosyaları.

## 🚀 Kurulum ve Çalıştırma

1.  Gerekli kütüphaneleri yükleyin:

    ```bash
    pip install -r requirements.txt
    ```

2.  Kaggle'dan verileri indirin ve `input/` klasörüne atın.

3.  Veri analizi için:

    ```bash
    python notebooks/01_Data_Prep_and_EDA.py
    ```

4.  Embedding çıkarmak için (CPU/GPU):
    ```bash
    python src/feature_extractor.py
    ```

## 📊 Mevcut Durum (Baseline)

- **Model:** 3 Katmanlı MLP (Multi-Layer Perceptron).
- **Embeddings:** Facebook ESM-2 (t6_8M) modeli kullanıldı.
- **Skor:** İlk denemeler yapıldı, geliştirme aşamasında.

---

_Detaylı teknik dokümantasyon için `Documentation.txt` dosyasına bakınız._
