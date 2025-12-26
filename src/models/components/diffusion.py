import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.unet import UNet


class DDPM(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, time_dim: int):
        super().__init__()

        self.time_dim = time_dim
        self.in_channels = in_channels

        beta = torch.linspace(0.0001, 0.02, time_dim)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)

        self.net = UNet(in_channels, hidden_channels, out_channels, time_dim)

    def forward_process(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        return x_t

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(x_t, t)

    def sample(self, num_samples: int) -> torch.Tensor:
        device = next(self.parameters()).device
        x_t = torch.randn(num_samples, self.in_channels, 28, 28, device=device)

        for t in reversed(range(self.time_dim - 1)):
            noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
            t_tensor = torch.full((num_samples,), t + 1, device=device, dtype=torch.long)
            x_t = (
                1
                / torch.sqrt(self.alpha[t + 1])
                * (x_t - (1 - self.alpha[t + 1]) / torch.sqrt(1 - self.alpha_bar[t + 1]) * self.net(x_t, t_tensor))
                + torch.sqrt(self.beta[t + 1]) * noise
            )

        return x_t

    def compute_loss(self, batch, batch_idx: int) -> torch.Tensor:
        x_0, _ = batch
        device = x_0.device
        t = torch.randint(0, self.time_dim, (x_0.size(0),), device=device, dtype=torch.long)
        noise = torch.randn_like(x_0)
        x_t = self.forward_process(x_0, t, noise)
        pred = self.net(x_t, t)
        loss = F.mse_loss(pred, noise)

        return loss
