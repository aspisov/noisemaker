import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        latent_dim: int = 32,
    ) -> None:
        super().__init__()

        self.latent_channels = latent_channels
        self.latent_dim = latent_dim

        self.encoder_base = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, latent_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        feature_dim = latent_channels * 7 * 7
        self.encoder_mean = nn.Linear(feature_dim, latent_dim)
        self.encoder_logvar = nn.Linear(feature_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, feature_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_base(x)
        h_flat = h.view(h.size(0), -1)
        mu = self.encoder_mean(h_flat)
        log_var = self.encoder_logvar(h_flat)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.decoder_input(z)
        z = z.view(z.size(0), self.latent_channels, 7, 7)
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decode(z)
        return x_reconstructed

    def sample(self, num_samples: int, device: torch.device | None = None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)

    def compute_loss(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, _ = batch
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decode(z)

        recon_loss = F.mse_loss(x_reconstructed, x)

        kl_loss = 0.5 * torch.sum(mu.pow(2) + log_var.exp() - 1 - log_var, dim=1).mean()

        beta = 0.01
        loss = recon_loss + beta * kl_loss

        return loss
