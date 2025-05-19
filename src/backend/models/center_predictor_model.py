import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return x + self._residual_path(x)

    def _residual_path(self, x):
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class CenterPredictorModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128):
        super().__init__()

        # Initial projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Residual blocks
        self.res_block1 = ResidualBlock(hidden_dim)
        self.res_block2 = ResidualBlock(hidden_dim)
        self.res_block3 = ResidualBlock(hidden_dim)

        # Attention
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention_linear = nn.Linear(hidden_dim, 1)

        # Output MLP for center prediction
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.output_relu = nn.ReLU()
        self.output_linear2 = nn.Linear(hidden_dim, 2)

    def forward(self, inputs, mask=None):
        # inputs: (B, N, 4), mask: (B, N)

        h = self.input_proj(inputs)  # (B, N, H)

        # Apply residual blocks
        h = self.res_block1(h)
        h = self.res_block2(h)
        h = self.res_block3(h)

        # Attention scores
        logits = self.attention_norm(h)
        logits = self.attention_linear(logits).squeeze(-1)  # (B, N)

        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)

        # Softmax
        attn = F.softmax(logits, dim=1).unsqueeze(-1)  # (B, N, 1)

        # Weighted sum of features
        pooled = torch.sum(h * attn, dim=1)  # (B, H)

        # Decode to center prediction
        center = self.output_norm(pooled)
        center = self.output_linear1(center)
        center = self.output_relu(center)
        center = self.output_linear2(center)  # (B, 2)

        return center
