import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
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


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time = nn.Linear(t_dim, out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(x))
        h = h + self.time(t_emb)[:, :, None, None]  # time bias
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class Down(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 4, stride=2, padding=1)  # /2

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)  # *2

    def forward(self, x):
        return self.deconv(x)


class UNet(nn.Module):
    """
    MNIST-friendly UNet:
      x: (B, C, 28, 28), t: (B,) -> eps_hat: (B, C, 28, 28)
    """

    def __init__(self, in_channels: int = 1, base: int = 64, out_channels: int = 1, t_dim: int = 128):
        super().__init__()

        self.t_embed = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )

        self.in_conv = nn.Conv2d(in_channels, base, 3, padding=1)

        # down
        self.b1 = ConvBlock(base, base, t_dim)
        self.down1 = Down(base)

        self.b2 = ConvBlock(base, base * 2, t_dim)
        self.down2 = Down(base * 2)

        # bottleneck
        self.mid = ConvBlock(base * 2, base * 2, t_dim)

        # up
        self.up2 = Up(base * 2)
        self.u2 = ConvBlock(base * 2 + base * 2, base, t_dim)  # concat skip from b2

        self.up1 = Up(base)
        self.u1 = ConvBlock(base + base, base, t_dim)  # concat skip from b1

        self.out = nn.Conv2d(base, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(x.size(0))
        t = t.to(device=x.device, dtype=torch.long)
        t_emb = self.t_embed(t)

        x = self.in_conv(x)

        s1 = self.b1(x, t_emb)  # (B, base, 28, 28)
        x = self.down1(s1)  # (B, base, 14, 14)

        s2 = self.b2(x, t_emb)  # (B, 2base, 14, 14)
        x = self.down2(s2)  # (B, 2base, 7, 7)

        x = self.mid(x, t_emb)  # (B, 2base, 7, 7)

        x = self.up2(x)  # (B, 2base, 14, 14)
        x = torch.cat([x, s2], dim=1)
        x = self.u2(x, t_emb)  # (B, base, 14, 14)

        x = self.up1(x)  # (B, base, 28, 28)
        x = torch.cat([x, s1], dim=1)
        x = self.u1(x, t_emb)  # (B, base, 28, 28)

        return self.out(F.silu(x))
