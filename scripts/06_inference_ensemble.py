import torch
import pandas as pd
import numpy as np
import sys
import os
import obonet
import networkx
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from src.model import CAFA6Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ASPECTS = {"F": 2000, "C": 1500, "P": 3000}

def generate_ensemble():
    print("🚀 Ensemble Tahmin Başlıyor...")
    # ... (Colab'deki 'generate_specialist_submission' kodunun aynısı)
    # Sadece dosya yollarını os.path.join(BASE_DIR, ...) şeklinde ayarla.
    # Örnek: os.path.join(BASE_DIR, "input", "test_embeddings_650M.npy")

if __name__ == "__main__":
    generate_ensemble()