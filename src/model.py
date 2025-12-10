import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Modelin daha derin öğrenmesini sağlayan ve 'Vanishing Gradient' 
    sorununu engelleyen blok.
    """
    def __init__(self, hidden_dim, dropout_rate=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        # Skip Connection (Girdiyi çıktıya ekle)
        return self.activation(x + self.block(x))

class CAFA6Model(nn.Module):
    def __init__(self, num_classes, num_taxons, input_dim=1280, hidden_dim=512, taxon_emb_dim=64):
        """
        Args:
            num_classes: Tahmin edilecek GO terimi sayısı (Örn: 1500)
            num_taxons: Toplam benzersiz tür sayısı
            input_dim: ESM-2 modelinden gelen vektör boyutu (1280)
        """
        super().__init__()
        
        # 1. Protein İşleme Kolu
        self.protein_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # 2. Taksonomi (Tür) İşleme Kolu (Tıpçı Stratejisi)
        # Tür ID'sini alıp ona özel bir vektör öğrenir.
        self.taxon_embedding = nn.Embedding(num_taxons + 1, taxon_emb_dim)
        
        # 3. Birleştirme (Concatenation)
        # Protein (512) + Taxon (64) = 576
        combined_dim = hidden_dim + taxon_emb_dim
        
        # 4. Derin Öğrenme Blokları (Residual)
        self.res_blocks = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim)
        )
        
        # 5. Çıktı Katmanı (Binary Classification for Multi-Label)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, embedding, taxon_id):
        # Protein özellikleri
        x_prot = self.protein_encoder(embedding)
        
        # Tür özellikleri
        x_tax = self.taxon_embedding(taxon_id)
        
        # Birleştir
        x = torch.cat([x_prot, x_tax], dim=1)
        
        # Derin ağdan geçir
        x = self.res_blocks(x)
        
        # Sınıflandırma
        logits = self.classifier(x)
        return logits