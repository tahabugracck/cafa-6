{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dc4fd51a",
   "metadata": {
    "vscode": {
     "languageId": "plaintext"
    }
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from Bio import SeqIO\n",
    "import os\n",
    "\n",
    "# Dosya yolları (Senin klasör yapına göre ayarlandı)\n",
    "INPUT_DIR = \"../input\"\n",
    "TRAIN_FASTA = os.path.join(INPUT_DIR, \"train_sequences.fasta\")\n",
    "TRAIN_TERMS = os.path.join(INPUT_DIR, \"train_terms.tsv\")\n",
    "\n",
    "def load_fasta(file_path):\n",
    "    \"\"\"FASTA dosyasını okuyup Pandas DataFrame'e çevirir.\"\"\"\n",
    "    print(f\"Loading {file_path}...\")\n",
    "    data = []\n",
    "    # SeqIO ile fasta okuma\n",
    "    for record in SeqIO.parse(file_path, \"fasta\"):\n",
    "        data.append({\n",
    "            \"id\": record.id,\n",
    "            \"seq\": str(record.seq),\n",
    "            \"len\": len(record.seq)\n",
    "        })\n",
    "    return pd.DataFrame(data)\n",
    "\n",
    "# 1. Protein Dizilerini Oku\n",
    "df_train = load_fasta(TRAIN_FASTA)\n",
    "print(f\"Toplam Protein Sayısı: {len(df_train)}\")\n",
    "print(df_train.head(3))\n",
    "\n",
    "# 2. Etiketleri Oku\n",
    "print(f\"\\nLoading {TRAIN_TERMS}...\")\n",
    "df_terms = pd.read_csv(TRAIN_TERMS, sep=\"\\t\")\n",
    "print(f\"Toplam Etiket Kaydı: {len(df_terms)}\")\n",
    "print(df_terms.head(3))\n",
    "\n",
    "# 3. İstatistiklere Bakalım\n",
    "print(\"\\n--- Özet Bilgiler ---\")\n",
    "print(f\"Benzersiz Protein Sayısı (Terms dosyasında): {df_terms['EntryID'].nunique()}\")\n",
    "print(\"Ontoloji Dağılımı:\")\n",
    "print(df_terms['aspect'].value_counts())\n",
    "\n",
    "# 4. Veriyi Birleştirme (Opsiyonel Kontrol)\n",
    "# Her proteinin en az bir etiketi var mı kontrol edelim\n",
    "ids_in_fasta = set(df_train['id'])\n",
    "ids_in_terms = set(df_terms['EntryID'])\n",
    "print(f\"\\nFasta dosyasında olup etiketi olmayan protein var mı? : {len(ids_in_fasta - ids_in_terms)}\")"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
