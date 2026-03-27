"""Native-image v36 variant built on the v35 frontier collective backbone.

This keeps the text path checkpoint-compatible with the current v35 model and adds
an internal image decoder head. The resulting checkpoint is one model file rather
than a text model plus an external image generator.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from model_frontier_v35 import FrontierCollectiveExpertHead
from run import ChampionNet


class NativeImageDecoder(nn.Module):
    """Small prompt-conditioned decoder for native low-resolution image synthesis."""

    def __init__(self, cond_dim: int = 512, image_size: int = 64, seed_hw: int = 8, seed_channels: int = 96) -> None:
        super().__init__()
        if int(image_size) != 64:
            raise ValueError("This decoder currently supports image_size=64 only.")
        self.image_size = int(image_size)
        self.seed_hw = int(seed_hw)
        self.seed_channels = int(seed_channels)

        self.seed_proj = nn.Sequential(
            nn.Linear(cond_dim, 1024),
            nn.SiLU(),
            nn.Linear(1024, self.seed_hw * self.seed_hw * self.seed_channels),
        )
        self.film1 = nn.Linear(cond_dim, 64 * 2)
        self.film2 = nn.Linear(cond_dim, 32 * 2)
        self.film3 = nn.Linear(cond_dim, 16 * 2)

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(self.seed_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
        )
        self.to_rgb = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 3, kernel_size=1),
        )

    @staticmethod
    def _apply_film(h: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        scale, bias = params.chunk(2, dim=-1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        bias = bias.unsqueeze(-1).unsqueeze(-1)
        return h * (1.0 + 0.10 * scale) + 0.10 * bias

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        batch = cond.shape[0]
        h = self.seed_proj(cond).view(batch, self.seed_channels, self.seed_hw, self.seed_hw)
        h = self._apply_film(self.up1(h), self.film1(cond))
        h = self._apply_film(self.up2(h), self.film2(cond))
        h = self._apply_film(self.up3(h), self.film3(cond))
        return torch.sigmoid(self.to_rgb(h))


class ChampionNetFrontierCollectiveNativeImage(nn.Module):
    """Single-checkpoint text+image variant using the v35 frontier text backbone."""

    def __init__(self, dropout: float = 0.1, image_size: int = 64) -> None:
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(
            FrontierCollectiveExpertHead(
                256,
                10,
                n_experts=10,
                n_mem_slots=24,
                reasoning_steps=5,
                n_sub_routers=4,
                dropout=dropout,
            )
        )
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

        self.image_size = int(image_size)
        self.image_cond_norm = nn.LayerNorm(512)
        self.image_mix = nn.Linear(512, 512, bias=True)
        self.image_decoder = NativeImageDecoder(cond_dim=512, image_size=self.image_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def encode_image_condition(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers[:10]:
            h = layer(h)
        pooled_mean = h.mean(dim=1)
        pooled_max = h.amax(dim=1)
        cond = torch.cat([pooled_mean, pooled_max], dim=-1)
        cond = self.image_cond_norm(cond)
        cond = cond + 0.15 * torch.tanh(self.image_mix(cond))
        return cond

    def forward_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.image_decoder(self.encode_image_condition(x))

    @torch.no_grad()
    def render_prompt(self, prompt: str, feature_mode: str = "context_mix_v4", device: torch.device | None = None) -> torch.Tensor:
        from chat_pipeline import text_to_model_input

        if device is None:
            device = next(self.parameters()).device
        inputs = text_to_model_input(prompt, feature_mode=feature_mode).to(device)
        image = self.forward_image(inputs)[0].detach().cpu().clamp(0.0, 1.0)
        return image


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    data = image_tensor.detach().clamp(0.0, 1.0).permute(1, 2, 0).mul(255.0).round().byte().cpu().numpy()
    return Image.fromarray(data, mode="RGB")


def save_prompt_image(
    model: ChampionNetFrontierCollectiveNativeImage,
    prompt: str,
    output_path: str | Path,
    *,
    feature_mode: str = "context_mix_v4",
    device: torch.device | None = None,
) -> Path:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = model.render_prompt(prompt, feature_mode=feature_mode, device=device)
    tensor_to_pil(image).save(out_path)
    return out_path
