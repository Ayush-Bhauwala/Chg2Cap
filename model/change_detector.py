import torch
import torch.nn as nn

class SiameseChangeHead(nn.Module):
    def __init__(self, in_channels=2048, reduced_dim=512):
        super().__init__()
        
        # 1. Bottleneck Convolution
        # Input channels = 2048 * 2 (because we concatenate)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels * 2, reduced_dim, kernel_size=1),
            nn.BatchNorm2d(reduced_dim), # Added BatchNorm for stability
            nn.ReLU(inplace=True)
        )
        
        # 2. Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # 3. Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),  # Dropout helps prevent overfitting on small datasets
            nn.Linear(reduced_dim, 1)
        )

    def forward(self, x1, x2):
        # x1, x2 shape: (Batch, 2048, 8, 8)
        
        # Step A: Merge (Concatenate along channel dimension)
        x = torch.cat([x1, x2], dim=1)  # -> (Batch, 4096, 8, 8)
        
        # Step B: Reduce Channels (Spatial info is still preserved here)
        x = self.bottleneck(x)          # -> (Batch, 512, 8, 8)
        
        # Step C: Pool (Collapse spatial info)
        x = self.gap(x)                 # -> (Batch, 512, 1, 1)
        
        # Step D: Classify
        logits = self.classifier(x)     # -> (Batch, 1)
        
        return logits