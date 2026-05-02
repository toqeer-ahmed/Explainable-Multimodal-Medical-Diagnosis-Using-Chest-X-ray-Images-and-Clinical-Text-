import torch
import torch.nn as nn
from .model_image import ImageEncoder
from .model_text import TextEncoder

class MultimodalFusion(nn.Module):
    def __init__(self, num_classes, freeze_bert=False):
        super(MultimodalFusion, self).__init__()
        
        # Initialize the unimodal encoders
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder(freeze_bert=freeze_bert)
        
        # --- UPGRADE: Cross-Modal Attention Fusion ---
        # Image is 1280 (EfficientNet-B0), Text is 768 (ClinicalBERT)
        self.hidden_dim = 512
        
        # Project both to same dimension
        self.v_proj = nn.Linear(self.image_encoder.feature_dim, self.hidden_dim)
        self.t_proj = nn.Linear(self.text_encoder.feature_dim, self.hidden_dim)
        
        # Multi-Head Attention (Text attends to Image)
        self.attention = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=8, batch_first=True)
        
        # Fusion Layers (Fully Connected)
        # Input size is hidden_dim after residual combination
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, images, input_ids, attention_mask):
        # 1. Extract features
        img_features = self.image_encoder(images)  # (batch_size, 1280)
        text_features = self.text_encoder(input_ids, attention_mask)  # (batch_size, 768)
        
        # 2. Project to shared hidden dimension
        v = self.v_proj(img_features).unsqueeze(1) # (batch_size, 1, hidden_dim)
        t = self.t_proj(text_features).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # 3. Cross-Modal Attention (Text query, Image key/value)
        attn_out, _ = self.attention(query=t, key=v, value=v)
        
        # 4. Residual Connection
        fused_features = (attn_out + t).squeeze(1) # (batch_size, hidden_dim)
        
        # 5. Final Classification
        logits = self.classifier(fused_features)  # (batch_size, num_classes)
        
        return logits

if __name__ == "__main__":
    # Test fusion model
    dummy_num_classes = 10
    model = MultimodalFusion(num_classes=dummy_num_classes)
    
    dummy_images = torch.randn(2, 3, 224, 224)
    dummy_input_ids = torch.randint(0, 1000, (2, 128))
    dummy_mask = torch.ones(2, 128)
    
    output = model(dummy_images, dummy_input_ids, dummy_mask)
    print(f"Fusion Model output shape: {output.shape} (Expected: 2, 10)")
