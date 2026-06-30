import torch
import torch.nn as nn

from nlp.bert.model import EncoderBlock


class PatchEmbed(nn.Module):
    """Split image into patches and project to d_model."""

    def __init__(self, image_size=32, patch_size=4, in_channels=3, d_model=128):
        super().__init__()
        assert image_size % patch_size == 0
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ViT(nn.Module):
    """Vision Transformer: patch embedding → Transformer encoder → CLS head."""

    def __init__(self, image_size=32, patch_size=4, in_channels=3,
                 d_model=128, n_heads=4, n_layers=4, d_ff=None,
                 num_classes=10, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, d_model)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d_model))
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff or d_model * 4, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)

        for block in self.blocks:
            x, _ = block(x)

        x = self.norm(x)
        cls_out = x[:, 0]
        return self.head(cls_out)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
