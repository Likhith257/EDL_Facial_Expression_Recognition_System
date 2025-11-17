
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ChannelAttention(nn.Module):
    """Channel Attention Module (part of CBAM)"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Average pool
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        # Max pool
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        out = avg_out + max_out
        return self.sigmoid(out).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """Spatial Attention Module (part of CBAM)"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel-wise average and max
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out


class EfficientNetB3Emotion(nn.Module):
    """
    EfficientNet-B3 with CBAM attention for emotion recognition.
    Optimized for Apple Silicon M4 Pro.
    """
    def __init__(self, num_classes=7, pretrained=True, dropout=0.3):
        super(EfficientNetB3Emotion, self).__init__()
        
        # Load pretrained EfficientNet-B3
        self.backbone = models.efficientnet_b3(pretrained=pretrained)
        
        # Get feature dimension (1536 for EfficientNet-B3)
        feature_dim = self.backbone.classifier[1].in_features
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Add CBAM attention after backbone
        self.attention = CBAM(feature_dim, reduction=16)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout/2),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features
        features = self.backbone.features(x)
        
        # Apply attention
        features = self.attention(features)
        
        # Global pooling
        features = self.global_pool(features)
        features = torch.flatten(features, 1)
        
        # Classify
        output = self.classifier(features)
        return output
    
    def get_feature_extractor(self):
        """Return the feature extraction part (without classifier)"""
        return nn.Sequential(
            self.backbone.features,
            self.attention,
            self.global_pool
        )


def create_model(num_classes=7, pretrained=True, dropout=0.3, device='mps'):
    """
    Factory function to create EfficientNet-B3 model.
    
    Args:
        num_classes: Number of emotion classes (default: 7)
        pretrained: Use ImageNet pretrained weights (default: True)
        dropout: Dropout rate (default: 0.3)
        device: Device to use ('mps', 'cuda', 'cpu')
    
    Returns:
        model: EfficientNetB3Emotion model
    """
    model = EfficientNetB3Emotion(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )
    
    # Move to device
    if device == 'mps' and torch.backends.mps.is_available():
        model = model.to('mps')
    elif device == 'cuda' and torch.cuda.is_available():
        model = model.to('cuda')
    else:
        model = model.to('cpu')
    
    return model


if __name__ == "__main__":
    # Test model
    print("Testing EfficientNet-B3 with CBAM...")
    model = create_model(num_classes=7, pretrained=False)
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    if torch.backends.mps.is_available():
        x = x.to('mps')
    
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✅ Model test passed!")
