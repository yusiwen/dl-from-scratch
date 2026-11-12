"""
BYOL (Bootstrap Your Own Latent, Grill et al. 2020) in miniature.

Two networks: an online encoder+projector+predictor trained by gradient
descent, and a target encoder+projector updated only as an exponential moving
average of the online one. Each view predicts the other view's target
projection, so there are no negative pairs and no labels.

Reusing the SimCLR augmentation keeps the comparison between the two methods
about the objective, not the data pipeline.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from cv.resnet18.model import BasicBlock, ResNet


class MLP(nn.Module):
    """Linear -> BatchNorm -> ReLU -> Linear, as used by the projector/predictor."""

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class BYOL(nn.Module):
    def __init__(self, projector_hidden=256, projector_dim=128, predictor_hidden=256):
        super().__init__()
        self.online_encoder = ResNet(BasicBlock, [2, 2, 2, 2])
        self.online_encoder.fc = nn.Identity()
        self.online_projector = MLP(512, projector_hidden, projector_dim)
        self.predictor = MLP(projector_dim, predictor_hidden, projector_dim)

        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.target_projector.parameters():
            parameter.requires_grad_(False)
        self.target_encoder.eval()
        self.target_projector.eval()

    def train(self, mode=True):
        """Keep the target networks in eval mode (their BN must use running stats)."""
        super().train(mode)
        self.target_encoder.eval()
        self.target_projector.eval()
        return self

    @staticmethod
    def _loss(prediction, target):
        prediction = F.normalize(prediction, dim=1)
        target = F.normalize(target, dim=1)
        return 2.0 - 2.0 * (prediction * target).sum(dim=1).mean()

    def forward(self, view1, view2):
        p1 = self.predictor(self.online_projector(self.online_encoder(view1)))
        p2 = self.predictor(self.online_projector(self.online_encoder(view2)))
        with torch.no_grad():
            z1 = self.target_projector(self.target_encoder(view1))
            z2 = self.target_projector(self.target_encoder(view2))
        return 0.5 * (self._loss(p1, z2) + self._loss(p2, z1))

    @torch.no_grad()
    def update_target(self, decay):
        """target <- decay * target + (1 - decay) * online, for weights and BN stats."""
        online_modules = (self.online_encoder, self.online_projector)
        target_modules = (self.target_encoder, self.target_projector)
        for online, target in zip(online_modules, target_modules):
            for online_parameter, target_parameter in zip(
                online.parameters(), target.parameters()
            ):
                target_parameter.mul_(decay).add_(online_parameter.detach(), alpha=1.0 - decay)
            for online_buffer, target_buffer in zip(online.buffers(), target.buffers()):
                target_buffer.copy_(online_buffer)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def _self_check():
    """The EMA update must equal decay*old + (1-decay)*online."""
    torch.manual_seed(0)
    model = BYOL()
    before = model.target_encoder.conv1.weight.detach().clone()
    with torch.no_grad():
        model.online_encoder.conv1.weight.add_(1.0)
    decay = 0.9
    model.update_target(decay)
    expected = decay * before + (1.0 - decay) * (before + 1.0)
    assert torch.allclose(model.target_encoder.conv1.weight, expected, atol=1e-6)
    print("byol self-check passed: EMA update is exact on conv1")


if __name__ == "__main__":
    _self_check()
