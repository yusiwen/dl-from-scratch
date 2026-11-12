"""
FID-style generative quality metric built on torchvision InceptionV3 features.

FID (Frechet Inception Distance) compares the distribution of real and generated
images in a feature space:

    FID = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2 (Sigma_r Sigma_g)^(1/2))

Both means and covariances come from 2048-d InceptionV3 pool features.

This implementation uses torchvision's InceptionV3 (the 2048-d penultimate
layer) and ImageNet normalization, not the original TensorFlow graph. Values
are therefore only comparable with values computed by this same class -- do
not compare them with published FID tables.

The matrix square root is computed from a symmetric eigendecomposition with
negative eigenvalues clamped to zero, which keeps a rank-deficient covariance
(the usual case when n_samples < 2048) from producing NaNs.
"""

import json
import os
import time

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import Inception_V3_Weights, inception_v3

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _sqrtm(matrix):
    """Symmetric positive semi-definite matrix square root."""
    values, vectors = torch.linalg.eigh(matrix)
    values = torch.clamp(values, min=0.0)
    return (vectors * values.sqrt()) @ vectors.T


def frechet_distance(mu1, cov1, mu2, cov2):
    """Frechet distance between two Gaussians."""
    diff = mu1 - mu2
    sqrt_cov1 = _sqrtm(cov1)
    covmean = _sqrtm(sqrt_cov1 @ cov2 @ sqrt_cov1)
    return float(
        diff @ diff
        + torch.trace(cov1)
        + torch.trace(cov2)
        - 2.0 * torch.trace(covmean)
    )


class FID:
    """Compute the FID-style distance between two image sets in [0, 1]."""

    def __init__(self, device="cpu", batch_size=64):
        self.device = device
        self.batch_size = batch_size
        self.model = inception_v3(
            weights=Inception_V3_Weights.DEFAULT, transform_input=False
        )
        self.model.fc = nn.Identity()
        self.model.to(device).eval()
        self.preprocess = transforms.Compose([
            transforms.Resize((299, 299), antialias=True),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    @torch.no_grad()
    def features(self, images):
        """Inception pool features for (n, 3, H, W) images in [0, 1]."""
        batches = []
        for chunk in torch.split(images, self.batch_size):
            batch = self.preprocess(chunk.to(self.device))
            batches.append(self.model(batch))
        return torch.cat(batches)

    @torch.no_grad()
    def __call__(self, real, fake):
        real_features = self.features(real)
        fake_features = self.features(fake)
        mu_real, mu_fake = real_features.mean(dim=0), fake_features.mean(dim=0)
        cov_real = torch.cov(real_features.T.double())
        cov_fake = torch.cov(fake_features.T.double())
        return frechet_distance(
            mu_real.double(), cov_real, mu_fake.double(), cov_fake
        )


def compute_and_save(real, fake, name, device, extra=None):
    """Compute FID, print it, and persist runs/fid/<name>.json."""
    evaluator = FID(device=device)
    started = time.time()
    score = evaluator(real, fake)
    elapsed = time.time() - started

    result = {
        "fid": round(score, 3),
        "num_samples": int(len(real)),
        "device": str(device),
        "elapsed_s": round(elapsed, 1),
    }
    if extra:
        result.update(extra)
    os.makedirs("runs/fid", exist_ok=True)
    with open(f"runs/fid/{name}.json", "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"FID = {score:.2f} (N={len(real)}, device={device}, elapsed={elapsed:.1f}s)")
    print(f"Results saved to runs/fid/{name}.json")
    return score


def _self_check():
    """Assert the distance on inputs whose value is known analytically."""
    torch.manual_seed(0)
    dim = 5
    eye = torch.eye(dim, dtype=torch.float64)
    zero = torch.zeros(dim, dtype=torch.float64)

    x = torch.randn(64, dim, dtype=torch.float64)
    cov = torch.cov(x.T)
    mu = x.mean(dim=0)
    assert abs(frechet_distance(mu, cov, mu, cov)) < 1e-6

    shift = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    assert abs(frechet_distance(zero, eye, shift, eye) - float(shift @ shift)) < 1e-4

    sigma_sq = 4.0
    expected = dim * (1.0 - sigma_sq ** 0.5) ** 2
    assert abs(frechet_distance(zero, eye, zero, sigma_sq * eye) - expected) < 1e-4

    y = torch.randn(32, dim, dtype=torch.float64)
    cov_y = torch.cov(y.T)
    mu_y = y.mean(dim=0)
    value = frechet_distance(mu, cov, mu_y, cov_y)
    permutation = torch.randperm(len(x))
    shuffled = frechet_distance(x[permutation].mean(dim=0), torch.cov(x[permutation].T), mu_y, cov_y)
    assert abs(shuffled - value) < 1e-6

    print(f"fid self-check passed (dim={dim}, value={value:.6f})")


if __name__ == "__main__":
    _self_check()
