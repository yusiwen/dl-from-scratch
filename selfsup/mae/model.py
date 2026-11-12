"""
Masked Autoencoder (He et al., 2021) in miniature.

The encoder sees only the visible patches (25% by default); learned mask
tokens + a shallow decoder reconstruct the raw pixels of the masked patches,
and the loss is computed on those masked patches only. That asymmetry is the
point of MAE: most of the image is a free reconstruction target, so the
encoder cannot copy its input.

The encoder reuses the ViT patch embedding and the BERT Transformer block.
"""

import torch
import torch.nn as nn

from cv.vit.model import PatchEmbed
from nlp.bert.model import EncoderBlock


class MaskedAutoencoder(nn.Module):
    def __init__(self, image_size=32, patch_size=4, in_channels=3, d_model=128,
                 n_heads=4, n_layers=4, decoder_dim=64, decoder_layers=2,
                 mask_ratio=0.75):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, d_model)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, d_model))
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_model * 4, dropout=0.0)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        self.decoder_embed = nn.Linear(d_model, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_dim))
        self.decoder_blocks = nn.ModuleList([
            EncoderBlock(decoder_dim, n_heads, decoder_dim * 4, dropout=0.0)
            for _ in range(decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(
            decoder_dim, patch_size * patch_size * in_channels
        )

        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.normal_(parameter, std=0.02)
        nn.init.zeros_(self.cls_token)
        nn.init.zeros_(self.mask_token)

    def _patchify(self, images):
        p = self.patch_size
        patches = images.unfold(2, p, p).unfold(3, p, p)
        b, c, gh, gw, _, _ = patches.shape
        return patches.permute(0, 2, 3, 1, 4, 5).reshape(b, gh * gw, p * p * c)

    def forward(self, images, mask_ratio=None):
        ratio = self.mask_ratio if mask_ratio is None else mask_ratio
        patches = self.patch_embed(images)
        b, n, _ = patches.shape
        n_keep = max(1, int(round(n * (1.0 - ratio))))

        ids = torch.argsort(torch.rand(b, n, device=images.device), dim=1)
        visible, masked = ids[:, :n_keep], ids[:, n_keep:]
        batch = torch.arange(b, device=images.device)[:, None]

        x_vis = patches[batch, visible]
        pos_vis = self.pos_embed[0, 1:][visible]
        cls = self.cls_token.expand(b, -1, -1) + self.pos_embed[:, :1]
        x = torch.cat([cls, x_vis + pos_vis], dim=1)
        for block in self.blocks:
            x, _ = block(x)
        x = self.norm(x)

        encoded = self.decoder_embed(x[:, 1:])
        full = self.mask_token.expand(b, n, -1).clone()
        full[batch, visible] = encoded
        decoder = torch.cat([self.decoder_embed(x[:, :1]), full], dim=1)
        decoder = decoder + self.decoder_pos_embed
        for block in self.decoder_blocks:
            decoder, _ = block(decoder)
        decoder = self.decoder_norm(decoder)
        predictions = self.decoder_pred(decoder[:, 1:])

        target = self._patchify(images)
        return torch.mean(
            (predictions[batch, masked] - target[batch, masked]) ** 2
        )

    def encode(self, images):
        """CLS feature for a batch (the representation the probe uses)."""
        patches = self.patch_embed(images)
        b = patches.size(0)
        x = torch.cat([self.cls_token.expand(b, -1, -1), patches], dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x, _ = block(x)
        return self.norm(x)[:, 0]

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
