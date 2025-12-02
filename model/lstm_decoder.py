import torch
from torch import nn


class LSTMCaptionGeneratorModel(nn.Module):
    def __init__(self, vocab_size, encoder_dim, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 512)
        self.input_encoder = nn.Linear(3 * encoder_dim, 512)
        self.lstm = nn.LSTM(512, 512, num_layers=2, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(512, vocab_size)

    def forward(self, img1, img2, input_seq):
        # fuse the images
        B, C, H, W = img1.shape
        f1 = nn.functional.adaptive_avg_pool2d(img1, 1).view(B, C)
        f2 = nn.functional.adaptive_avg_pool2d(img2, 1).view(B, C)
        img = torch.cat([f1, f2, f2 - f1], dim=1)  # (B, 3C)
        img = self.input_encoder(img)

        embeddings = self.embedding(input_seq)  # (batch, max_len, 512)
        embeddings = self.dropout(embeddings)
        # Insert image as first word
        img = img.unsqueeze(1)  # (batch, 1, 512)
        lstm_input = torch.cat([img, embeddings], dim=1)  # (batch, max_len+1, 512)
        hidden, _ = self.lstm(lstm_input)
        hidden = self.dropout(hidden)
        out = self.output(hidden)
        out = out[:, 1:, :]  # (batch, max_len, vocab_size)
        return out
