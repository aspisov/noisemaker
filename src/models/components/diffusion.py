import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DDMPScheduler(nn.Module):
    def __init__(self, num_train_timesteps: int, beta_start: float, beta_end: float, beta_schedule: str):
        super().__init__()

        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "squaredcos_cap_v2":

            def alpha_bar_fn(t):
                return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

            betas = []
            for i in range(num_train_timesteps):
                t1 = i / num_train_timesteps
                t2 = (i + 1) / num_train_timesteps
                betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), beta_end))
            betas = torch.tensor(betas, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported beta schedule: {beta_schedule}")
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        alpha_bar_t = self.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        x_t = torch.sqrt(alpha_bar_t) * original_samples + torch.sqrt(1 - alpha_bar_t) * noise
        return x_t


class DDPM(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, unet: nn.Module, scheduler: DDMPScheduler):
        super().__init__()

        self.time_dim = time_dim
        self.in_channels = in_channels

        self.scheduler = scheduler

        self.net = unet

    def reverse_process(self, x_t: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        pred_noise = self.net(x_t, timesteps).sample

        alpha_t = self.scheduler.alphas[timesteps].view(-1, 1, 1, 1)
        alpha_cumprod_t = self.scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        beta_t = self.scheduler.betas[timesteps].view(-1, 1, 1, 1)

        scaled_pred_noise = (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t) * pred_noise

        x_t_minus_1 = (1 / torch.sqrt(alpha_t)) * (x_t - scaled_pred_noise) + torch.sqrt(beta_t) * noise
        return x_t_minus_1

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(x_t, t).sample

    def sample(self, num_samples: int) -> torch.Tensor:
        device = next(self.parameters()).device
        x_t = torch.randn(num_samples, self.in_channels, 28, 28, device=device)

        for t in reversed(range(self.time_dim - 1)):
            timesteps = torch.full((x_t.size(0),), t, device=x_t.device, dtype=torch.long)
            noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
            x_t = self.reverse_process(x_t, timesteps, noise)

        return x_t

    def compute_loss(self, batch, batch_idx: int) -> torch.Tensor:
        x_0, _ = batch
        device = x_0.device
        timesteps = torch.randint(0, self.time_dim, (x_0.size(0),), device=device, dtype=torch.long)
        noise = torch.randn_like(x_0)
        x_t = self.scheduler.add_noise(x_0, noise, timesteps)
        pred = self.net(x_t, timesteps).sample
        loss = F.mse_loss(pred, noise)

        return loss
