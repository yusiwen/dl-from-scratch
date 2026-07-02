import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoid_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


def nongroup_norm(ch):
    """GroupNorm with auto group count (largest divisor ≤ 32)."""
    g = max(g for g in range(1, min(33, ch + 1)) if ch % g == 0)
    return nn.GroupNorm(g, ch)


class ResBlock(nn.Module):
    def __init__(self, ch, time_emb_dim):
        super().__init__()
        self.norm1 = nongroup_norm(ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nongroup_norm(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, ch))

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + x


class UNet(nn.Module):
    """DDPM UNet with timestep conditioning. Clean stage-by-stage design."""

    def __init__(self, in_channels=3, model_channels=64, channel_mult=(1, 2, 3, 4), num_res_blocks=2):
        super().__init__()
        time_emb_dim = model_channels * 4

        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_emb_dim), nn.SiLU(), nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Encoder: each stage has num_res_blocks ResBlocks + downsampling.
        ch = model_channels
        self.encoder_stages = nn.ModuleList()
        self.encoder_downs = nn.ModuleList()
        stage_channels = [ch]
        for mult in channel_mult:
            out_ch = model_channels * mult
            blocks = nn.ModuleList([ResBlock(ch, time_emb_dim) for _ in range(num_res_blocks)])
            self.encoder_stages.append(blocks)
            self.encoder_downs.append(nn.Conv2d(ch, out_ch, 3, stride=2, padding=1) if ch != out_ch else nn.AvgPool2d(2))
            ch = out_ch
            stage_channels.append(ch)

        # Bottleneck.
        self.bottleneck = nn.ModuleList([ResBlock(ch, time_emb_dim) for _ in range(num_res_blocks)])

        # Decoder: each stage has upsampling + skip concat + ResBlocks.
        self.decoder_ups = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()
        self.decoder_merges = nn.ModuleList()
        for skip_ch in reversed(stage_channels[:-1]):
            up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.decoder_ups.append(up)
            merge = nn.Conv2d(ch + skip_ch, ch, 1)  # concat → project
            self.decoder_merges.append(merge)
            blocks = nn.ModuleList([ResBlock(ch, time_emb_dim) for _ in range(num_res_blocks)])
            self.decoder_stages.append(blocks)

        self.conv_out = nn.Conv2d(ch, in_channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = sinusoid_embedding(t, self.time_embed[0].in_features)
        t_emb = self.time_embed(t_emb)

        x = self.conv_in(x)
        skips = [x]

        # Encode: save only the last block's output per stage (before down).
        for blocks, down in zip(self.encoder_stages, self.encoder_downs):
            for block in blocks:
                x = block(x, t_emb)
            skips.append(x)
            x = down(x)

        # Bottleneck.
        for block in self.bottleneck:
            x = block(x, t_emb)

        # Decode.
        for up, merge, blocks in zip(self.decoder_ups, self.decoder_merges, self.decoder_stages):
            x = up(x)
            skip = skips.pop()
            x = merge(torch.cat([x, skip], dim=1))
            for block in blocks:
                x = block(x, t_emb)

        return self.conv_out(x)


class DDPM(nn.Module):
    def __init__(self, in_channels=3, model_channels=64, channel_mult=(1, 2, 3, 4),
                 num_res_blocks=2, T=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.T = T
        self.unet = UNet(in_channels, model_channels, channel_mult, num_res_blocks)

        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def forward(self, x_0):
        T = self.T
        B = x_0.size(0)
        t = torch.randint(1, T + 1, (B,), device=x_0.device)
        noise = torch.randn_like(x_0)
        sqrt_ab = self.alpha_bars[t - 1].sqrt().view(-1, 1, 1, 1)
        sqrt_1m_ab = (1.0 - self.alpha_bars[t - 1]).sqrt().view(-1, 1, 1, 1)
        predicted = self.unet(sqrt_ab * x_0 + sqrt_1m_ab * noise, t.float())
        return F.mse_loss(predicted, noise)

    @torch.no_grad()
    def sample(self, batch_size=64, image_size=32, channels=3, device="cpu"):
        x = torch.randn(batch_size, channels, image_size, image_size, device=device)
        for t in range(self.T, 0, -1):
            t_tensor = torch.full((batch_size,), float(t), device=device)
            pred = self.unet(x, t_tensor)
            alpha = self.alphas[t - 1].view(1, 1, 1, 1)
            beta = self.betas[t - 1].view(1, 1, 1, 1)
            ab = self.alpha_bars[t - 1].view(1, 1, 1, 1)
            x = (1.0 / alpha.sqrt()) * (x - beta / (1.0 - ab).sqrt() * pred)
            if t > 1:
                x += beta.sqrt() * torch.randn_like(x)
        return x.clamp(-1, 1)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
