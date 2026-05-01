import torch
import torch.nn as nn
from torchvision import models

class ImageEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(ImageEncoder, self).__init__()
        # Load DenseNet121, common for medical image tasks
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        densenet = models.densenet121(weights=weights)
        
        # We extract features, so we remove the final classifier layer
        self.features = densenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Output feature size of DenseNet121 is 1024
        self.feature_dim = 1024

    def forward(self, x):
        # x shape: (batch_size, 3, 224, 224)
        x = self.features(x)
        x = self.pool(x)
        # Flatten
        x = torch.flatten(x, 1)
        # Output shape: (batch_size, 1024)
        return x

if __name__ == "__main__":
    model = ImageEncoder()
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Image Encoder output shape: {output.shape}")
