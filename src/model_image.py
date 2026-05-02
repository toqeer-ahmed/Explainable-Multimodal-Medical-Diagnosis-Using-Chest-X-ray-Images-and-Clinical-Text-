import torch
import torch.nn as nn
from torchvision import models

class ImageEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(ImageEncoder, self).__init__()
        # --- UPGRADE: EfficientNet-B0 instead of DenseNet ---
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        efficientnet = models.efficientnet_b0(weights=weights)
        
        self.features = efficientnet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Output feature size of EfficientNet-B0 is 1280
        self.feature_dim = 1280

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        # Output shape: (batch_size, 1024)
        return x

if __name__ == "__main__":
    model = ImageEncoder()
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Image Encoder output shape: {output.shape}")
