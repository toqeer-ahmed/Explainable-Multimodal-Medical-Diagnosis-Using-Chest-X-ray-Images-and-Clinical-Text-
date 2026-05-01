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
        
        # Calculate combined feature dimension
        self.combined_dim = self.image_encoder.feature_dim + self.text_encoder.feature_dim
        
        # Fusion Layers (Fully Connected)
        self.classifier = nn.Sequential(
            nn.Linear(self.combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, images, input_ids, attention_mask):
        # 1. Extract image features
        img_features = self.image_encoder(images)  # (batch_size, 1024)
        
        # 2. Extract text features
        text_features = self.text_encoder(input_ids, attention_mask)  # (batch_size, 768)
        
        # 3. Concatenate modalities
        fused_features = torch.cat((img_features, text_features), dim=1)  # (batch_size, 1792)
        
        # 4. Final Classification (Outputs raw logits, Sigmoid is applied in Loss function)
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
