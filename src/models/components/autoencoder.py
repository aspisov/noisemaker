import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
    ) -> None:
        super().__init__()

        self.latent_channels = latent_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, latent_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed

    def sample(self, num_samples: int) -> torch.Tensor:
        noise = torch.randn(num_samples, self.latent_channels, 7, 7)
        return self.decode(noise)

    def compute_loss(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, _ = batch
        x_reconstructed = self(x)
        loss = F.mse_loss(x_reconstructed, x)
        return loss
