import torch
import torch.nn as nn
from einops import rearrange


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    x: (B, T, d_model)
    shift: (B, d_model)
    scale: (B, d_model)
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Patchify(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, d_model: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, C, I, I) -> (B, d_model, I/patch_size, I/patch_size)
        x = rearrange(x, "b c h w -> b (h w) c")  # (B, d_model, H, W) -> (B, T, d_model)
        return x


class TimeEmbedder(nn.Module):
    pass


class LabelEmbedder(nn.Module):
    pass


class DiTBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))

        self.layer_norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(d_model, elementwise_affine=False)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.LeakyReLU(0.2),
            nn.Linear(4 * d_model, d_model),
        )

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn = self.adaLN_modulation(c).chunk(6, dim=1)

        x_norm1 = modulate(self.layer_norm1(x), shift_msa, scale_msa)
        x_attn, _ = self.self_attn(x_norm1, x_norm1, x_norm1)
        x = x + gate_msa.unsqueeze(1) * x_attn

        x_norm2 = modulate(self.layer_norm2(x), shift_ffn, scale_ffn)
        x_ffn = self.ffn(x_norm2)
        x = x + gate_ffn.unsqueeze(1) * x_ffn

        return x


class FinalLayer(nn.Module):
    def __init__(self, d_model: int, latent_channels: int, patch_size: int) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.linear_head = nn.Linear(d_model, latent_channels * patch_size * patch_size)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.layer_norm(x), shift, scale)
        x = self.linear_head(x)
        return x


class DiT(nn.Module):
    def __init__(self, d_model: int, num_blocks: int, patch_size: int, num_heads: int, latent_channels: int) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.patch_size = patch_size

        self.patchify = Patchify(in_channels=latent_channels, patch_size=patch_size, d_model=d_model)
        self.t_embedder = TimeEmbedder()
        self.label_embedder = LabelEmbedder()

        self.transformer_blocks = nn.ModuleList(
            [DiTBlock(d_model=d_model, num_heads=num_heads) for _ in range(num_blocks)]
        )
        self.final_layer = FinalLayer(d_model=d_model, latent_channels=latent_channels, patch_size=patch_size)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        t = self.t_embedder(timestep)
        y = self.label_embedder(label)
        c = t + y

        x = self.patchify(x)

        for block in self.transformer_blocks:
            x = block(x, c)

        x = self.final_layer(x, c)

        return self.unpatchify(x)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, patch_size**2 * latent_channels)
        """
        B, T, D = x.shape
        H = W = int(T**0.5)
        x = x.reshape(shape=(B, H, W, self.patch_size, self.patch_size, self.latent_channels))
        x = rearrange(x, "b h w p q c -> b c h p w q")
        imgs = x.reshape(shape=(B, self.latent_channels, H * self.patch_size, W * self.patch_size))
        return imgs
