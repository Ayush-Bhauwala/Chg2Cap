import torch
import torch.nn as nn
import torch.nn.functional as F

class VQAWithAttention(nn.Module):
    def __init__(self, in_channels, text_embed_dim, hidden_dim=512):
        super(VQAWithAttention, self).__init__()
        
        # Feature Reduction
        self.img_projector = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        
        # Text Projection
        self.txt_projector = nn.Linear(text_embed_dim, hidden_dim)
        
        # Attention Network
        # Takes Combined Features -> Outputs 1-channel heatmap
        self.attn_layer = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1)
        )

        # Final Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, img1, img2, qstn_emb):
        # A: Process Images
        # Concatenate inputs: (Batch, 2048, 8, 8) -> (Batch, 4096, 8, 8)
        img_features = torch.cat([img1, img2], dim=1)
        
        # Reduce channels: (Batch, 512, 8, 8)
        img_features = self.img_projector(img_features)
        
        # B: Process Question
        # Project text: (Batch, 384) -> (Batch, 512)
        txt_features = self.txt_projector(qstn_emb)
        
        # Expand text to match image spatial dims
        # (Batch, 512, 1, 1) -> (Batch, 512, 8, 8)
        txt_expanded = txt_features.unsqueeze(2).unsqueeze(3)
        txt_expanded = txt_expanded.expand_as(img_features)
        
        # C: Spatial Attention ---
        # Combine Image and Text (Element-wise multiplication)
        combined_features = img_features * txt_expanded 
        
        # Calculate Attention Map: (Batch, 1, 8, 8)
        attn_map = self.attn_layer(combined_features)
        attn_scores = F.softmax(attn_map.view(attn_map.size(0), -1), dim=1) # Softmax over H*W
        attn_scores = attn_scores.view(attn_map.size()) # Reshape back to (B, 1, 8, 8)
        
        # Apply Attention
        attended_features = img_features * attn_scores
        
        # D: Global Sum Pooling
        global_features = torch.sum(attended_features, dim=[2, 3]) # -> (Batch, 512)
        
        # E: Classify
        final_vector = global_features + txt_features
        logits = self.classifier(final_vector)
        
        return logits