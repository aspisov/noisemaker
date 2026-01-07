import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) long/int -> float
        t = t.float()
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, half, device=t.device, dtype=torch.float32)
            / (half - 1 if half > 1 else 1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, kernel_size, padding, up=False):
        super().__init__()

        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        if up:
            self.conv1 = nn.Conv2d(2 * in_channels, out_channels, kernel_size, padding=padding)
            self.transform = nn.ConvTranspose2d(out_channels, out_channels, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
            self.transform = nn.Conv2d(out_channels, out_channels, 4, 2, 1)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.batch_norm_1 = nn.GroupNorm(32, out_channels)
        self.batch_norm_2 = nn.GroupNorm(32, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        # First Conv
        h = self.batch_norm_1(self.relu(self.conv1(x)))

        # Time embedding
        time_emb = self.relu(self.time_mlp(t))

        # Extend last 2 dimensions
        time_emb = time_emb[(...,) + (None,) * 2]

        # Add time channel
        h = h + time_emb

        # Second Conv
        h = self.batch_norm_2(self.relu(self.conv2(h)))

        # Down or Upsample
        return self.transform(h)


class UNet(nn.Module):
    """A minimal UNet implementation with timestep conditioning."""

    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=28):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim), nn.Linear(time_emb_dim, time_emb_dim), nn.ReLU()
        )

        # Initial projection
        self.conv0 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)

        # Downsample
        self.downs = nn.ModuleList(
            [
                Block(32, 64, time_emb_dim, kernel_size=5, padding=2),
                Block(64, 64, time_emb_dim, kernel_size=5, padding=2),
            ]
        )

        # Upsample
        self.ups = nn.ModuleList(
            [
                Block(64, 64, time_emb_dim, kernel_size=5, padding=2, up=True),
                Block(64, 32, time_emb_dim, kernel_size=5, padding=2, up=True),
            ]
        )

        self.output = nn.Conv2d(32, out_channels, kernel_size=5, padding=2)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)

        # Initial conv
        x = self.conv0(x)

        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)

        for i, up in enumerate(self.ups):
            residual_x = residual_inputs.pop()

            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)

        return self.output(x)
