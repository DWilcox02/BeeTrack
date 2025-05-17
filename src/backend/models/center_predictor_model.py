import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterPredictorModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128): 
        super().__init__()

        # Deeper point encoder with residual connections
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.res_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(3)
            ]
        )

        self.aggregator = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

        self.output_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, inputs, mask=None):
        # inputs: (B, N, 4), mask: (B, N) or None
        B, N, _ = inputs.shape

        # Initial projection
        h = self.input_proj(inputs)  # (B, N, H)

        # Apply residual blocks
        for res_block in self.res_blocks:
            h = h + res_block(h)  # Residual connection

        # Compute attention scores with improved numerical stability
        logits = self.aggregator(h).squeeze(-1)  # (B, N)

        if mask is not None:
            # Use a more numerically stable mask application
            mask_value = -1e9
            logits = logits.masked_fill(~mask, mask_value)

        # Use a more numerically stable softmax with temperature
        temperature = 0.1
        attn = F.softmax(logits / temperature, dim=1).unsqueeze(-1)  # (B, N, 1)

        # Weighted sum of features
        pooled = torch.sum(h * attn, dim=1)  # (B, H)

        # Decode to center prediction
        center = self.output_mlp(pooled)  # (B, 2)
        return center