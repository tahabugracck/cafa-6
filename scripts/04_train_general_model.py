import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import re

# Proje kök dizinini yola ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import CAFA6Model
from src.data_loader import load_data, CAFA6Dataset

# Ayarlar
BATCH_SIZE = 256
LEARNING_RATE = 8e-4
EPOCHS = 20
TOP_N_LABELS = 5000  # Geniş kapsamlı öğrenme
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    print(f"🚀 Genel Model Eğitimi Başlıyor (Cihaz: {DEVICE})...")
    # ... (Buraya Colab'deki 03_train_model.py kodunun içeriğini yapıştır)
    # Colab'deki kodun aynısı, sadece dosya yollarını 'input/' ve 'data/' olarak göreceli ver.