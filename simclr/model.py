import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet18.model import ResNet, BasicBlock


class Projector(nn.Module):
    """MLP projection head."""

    def __init__(self, in_dim=512, hidden_dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimCLR(nn.Module):
    """SimCLR: contrastive learning with NT-Xent loss."""

    def __init__(self, project_dim=128, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        # Encoder: ResNet18 without FC layer.
        self.encoder = ResNet(BasicBlock, [2, 2, 2, 2])
        # Remove the FC layer (avgpool is kept).
        self.encoder.fc = nn.Identity()
        self.projector = Projector(in_dim=512, hidden_dim=256, out_dim=project_dim)

    def forward(self, x):
        h = self.encoder(x)  # (B, 512)
        return self.projector(h)  # (B, project_dim)

    def nt_xent_loss(self, z1, z2):
        """NT-Xent loss between two augmentation views.

        z1, z2: (B, D) embeddings of two views.
        """
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)  # (2B, D)
        z = F.normalize(z, dim=1)

        # Cosine similarity matrix: (2B, 2B)
        sim = z @ z.T / self.temperature

        # Mask out self-similarity.
        mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, float("-inf"))

        # Positive pairs: (i, i+B) and (i+B, i)
        pos_mask = torch.zeros(2 * B, 2 * B, device=z.device, dtype=torch.bool)
        for i in range(B):
            pos_mask[i, i + B] = True
            pos_mask[i + B, i] = True

        # Compute loss for all 2B samples.
        pos = sim[pos_mask].view(2 * B, 1)
        neg = sim.masked_fill(pos_mask, float("-inf"))
        logits = torch.cat([pos, neg], dim=1)
        labels = torch.zeros(2 * B, device=z.device, dtype=torch.long)

        return F.cross_entropy(logits, labels)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
