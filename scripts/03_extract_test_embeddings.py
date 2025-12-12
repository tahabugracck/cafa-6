import torch
import os
import numpy as np
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FASTA = os.path.join(BASE_DIR, "data", "Test", "testsuperset.fasta")
OUTPUT_DIR = os.path.join(BASE_DIR, "input")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
BATCH_SIZE = 8 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract():
    print(f"Test Embedding İşlemi Başlıyor... Cihaz: {DEVICE}")
    if not os.path.exists(INPUT_FASTA):
        print("Test fasta dosyası bulunamadı.")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    sequences, ids = [], []
    for record in SeqIO.parse(INPUT_FASTA, "fasta"):
        clean_id = record.id.split('|')[1] if "|" in record.id else record.id
        sequences.append(str(record.seq))
        ids.append(clean_id)

    embeddings = []
    for i in tqdm(range(0, len(sequences), BATCH_SIZE)):
        batch_seqs = sequences[i:i + BATCH_SIZE]
        inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy())

    np.save(os.path.join(OUTPUT_DIR, "test_embeddings_650M.npy"), np.vstack(embeddings))
    np.save(os.path.join(OUTPUT_DIR, "test_ids_650M.npy"), np.array(ids))
    print("Test Embedding İşlemi Tamamlandı.")

if __name__ == "__main__":
    extract()