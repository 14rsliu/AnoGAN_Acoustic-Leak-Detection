"""Paper-aligned model-only AnoGAN components.

This file contains the generator, discriminator, and latent inversion used by
the core method in "Learning from Silence". It intentionally excludes data,
training pipelines, checkpoints, experiments, and visualization code.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_hw(size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(size, int):
        return (size, size)
    if len(size) != 2:
        raise ValueError("size must be an int or a (height, width) tuple")
    height, width = int(size[0]), int(size[1])
    if height < 16 or width < 16:
        raise ValueError("height and width should both be at least 16")
    return (height, width)


class Generator(nn.Module):
    """DCGAN-style generator for single-channel spectrogram patches.

    The native transposed-convolution stack produces a 64x64 feature map. If
    another ``output_size`` is requested, the generated spectrogram is resized
    to that target size before being returned.
    """

    def __init__(
        self,
        z_dim: int = 128,
        ngf: int = 64,
        out_channels: int = 1,
        output_size: int | tuple[int, int] = 64,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.ngf = ngf
        self.out_channels = out_channels
        self.output_size = _as_hw(output_size)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, out_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.net(z)
        if x.shape[-2:] != self.output_size:
            x = F.interpolate(x, size=self.output_size, mode="bilinear", align_corners=False)
        return x


class Discriminator(nn.Module):
    """DCGAN-style discriminator with an intermediate feature output.

    The convolutional feature extractor accepts variable spatial sizes. The
    final classifier uses adaptive pooling, so inputs are not restricted to
    64x64 as long as height and width are large enough for the convolutional
    stack.
    """

    def __init__(self, ndf: int = 64, in_channels: int = 1) -> None:
        super().__init__()
        self.ndf = ndf
        self.in_channels = in_channels
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Conv2d(ndf * 8, 1, 1, 1, 0, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h1 = self.block1(x)
        h2 = self.block2(h1)
        features = self.block3(h2)
        h4 = self.block4(features)
        pooled = self.global_pool(h4)
        probability = torch.sigmoid(self.classifier(pooled))
        return probability, features


@dataclass(frozen=True)
class LatentOptimizationResult:
    z: torch.Tensor
    reconstruction: torch.Tensor
    anomaly_score: torch.Tensor


def optimize_latent(
    generator: Generator,
    discriminator: Discriminator,
    x: torch.Tensor,
    *,
    z_dim: int = 128,
    steps: int = 100,
    z_lr: float = 5e-3,
) -> LatentOptimizationResult:
    """Invert inputs into latent space and return residual anomaly scores.

    Latent inversion uses the paper's equally weighted residual and
    discriminator-feature objective (lambda = 0.5). The final anomaly score is
    the per-sample L1 residual, matching the configuration selected in the
    paper's ablation study.

    Args:
        generator: Trained generator.
        discriminator: Trained discriminator used for feature matching.
        x: Input tensor shaped ``[batch, channels, height, width]``.
        z_dim: Latent vector dimension.
        steps: Number of latent optimization iterations.
        z_lr: Learning rate for latent optimization.
    """

    if steps < 1:
        raise ValueError("steps must be at least 1")
    if z_lr <= 0:
        raise ValueError("z_lr must be positive")

    device = x.device
    batch = x.shape[0]
    z = torch.randn(batch, z_dim, 1, 1, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=z_lr)

    generator.eval()
    discriminator.eval()
    with torch.no_grad():
        _, target_features = discriminator(x)

    for _ in range(steps):
        optimizer.zero_grad()
        reconstruction = generator(z)
        if reconstruction.shape[-2:] != x.shape[-2:]:
            reconstruction = F.interpolate(
                reconstruction,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        _, reconstructed_features = discriminator(reconstruction)
        residual = F.l1_loss(reconstruction, x)
        feature = F.mse_loss(reconstructed_features, target_features)
        loss = 0.5 * residual + 0.5 * feature
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        reconstruction = generator(z)
        if reconstruction.shape[-2:] != x.shape[-2:]:
            reconstruction = F.interpolate(
                reconstruction,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        residual_per_sample = torch.mean(torch.abs(reconstruction - x), dim=(1, 2, 3))

    return LatentOptimizationResult(
        z=z.detach(),
        reconstruction=reconstruction.detach(),
        anomaly_score=residual_per_sample.detach(),
    )


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
