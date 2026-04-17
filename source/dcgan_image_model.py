from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch import nn


def _prompt_seed(prompt: str, *, offset: int = 0) -> int:
    digest = hashlib.sha1(str(prompt or "").encode("utf-8")).hexdigest()
    return (int(digest[:12], 16) + int(offset)) % (2**31 - 1)


def _safe_slug(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(text or ""))
    cooked = "-".join(part for part in cleaned.split("-") if part)
    return cooked[:72] or "artifact"


class DCGANMnistGenerator(nn.Module):
    def __init__(self, latent_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 7 * 7 * 128),
            nn.BatchNorm1d(7 * 7 * 128),
            nn.ReLU(True),
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DCGANCifarV2Generator(nn.Module):
    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * 512, bias=False),
            nn.Unflatten(1, (512, 4, 4)),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


@dataclass(frozen=True)
class DCGANSpec:
    key: str
    label: str
    latent_dim: int
    image_size: int
    channels: int
    preferred_weights: Tuple[str, ...]
    description: str


DCGAN_SPECS: Dict[str, DCGANSpec] = {
    "dcgan_mnist_model": DCGANSpec(
        key="dcgan_mnist_model",
        label="DCGAN MNIST",
        latent_dim=64,
        image_size=28,
        channels=1,
        preferred_weights=("generator_final.pth",),
        description="Unconditional DCGAN trained on 28x28 grayscale MNIST digits.",
    ),
    "dcgan_v2_in_progress": DCGANSpec(
        key="dcgan_v2_in_progress",
        label="DCGAN V2 CIFAR",
        latent_dim=128,
        image_size=32,
        channels=3,
        preferred_weights=("generator_epoch_015.pth", "generator_epoch_010.pth", "generator_epoch_005.pth"),
        description="Unconditional DCGAN v2 trained on 32x32 RGB CIFAR-style images.",
    ),
}


def _build_generator(spec: DCGANSpec) -> nn.Module:
    if spec.key == "dcgan_mnist_model":
        return DCGANMnistGenerator(latent_dim=spec.latent_dim)
    if spec.key == "dcgan_v2_in_progress":
        return DCGANCifarV2Generator(latent_dim=spec.latent_dim)
    raise KeyError(f"Unsupported DCGAN spec: {spec.key}")


def _find_generator_weights(extracted_dir: Path, preferred_names: Sequence[str]) -> Path:
    for name in preferred_names:
        candidate = extracted_dir / name
        if candidate.exists():
            return candidate.resolve()
    for name in preferred_names:
        matches = list(extracted_dir.rglob(Path(name).name))
        if matches:
            return sorted(matches)[0].resolve()
    all_generators = sorted(
        [
            path
            for path in extracted_dir.rglob("*.pth")
            if "generator" in path.name.lower()
        ],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if all_generators:
        return all_generators[0].resolve()
    raise FileNotFoundError(f"Could not find a DCGAN generator checkpoint under {extracted_dir}")


def save_image_grid(images: torch.Tensor, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images = images.detach().cpu().clamp(-1.0, 1.0)
    images = (images + 1.0) / 2.0
    count = int(images.size(0))
    cols = int(math.ceil(math.sqrt(count)))
    rows = int(math.ceil(count / max(cols, 1)))
    channels = int(images.size(1))
    height = int(images.size(2))
    width = int(images.size(3))

    canvas = torch.zeros(channels, rows * height, cols * width)
    for index, image in enumerate(images):
        row = index // cols
        col = index % cols
        canvas[:, row * height : (row + 1) * height, col * width : (col + 1) * width] = image
    array = canvas.mul(255.0).byte().permute(1, 2, 0).numpy()
    if channels == 1:
        Image.fromarray(array[:, :, 0], mode="L").save(output_path)
    else:
        Image.fromarray(array, mode="RGB").save(output_path)


class DCGANImageEngine:
    def __init__(self, *, key: str, weights_path: Path, device: Optional[torch.device] = None) -> None:
        self.spec = DCGAN_SPECS[key]
        self.weights_path = Path(weights_path).resolve()
        self.device = device or torch.device("cpu")
        self.generator = _build_generator(self.spec).to(self.device)
        state = torch.load(self.weights_path, map_location=self.device, weights_only=False)
        if not isinstance(state, dict):
            raise TypeError(f"Unexpected checkpoint payload in {self.weights_path}")
        self.generator.load_state_dict(state, strict=True)
        self.generator.eval()

    def status(self) -> Dict[str, Any]:
        return {
            "key": self.spec.key,
            "label": self.spec.label,
            "weights_path": str(self.weights_path),
            "latent_dim": self.spec.latent_dim,
            "image_size": self.spec.image_size,
            "channels": self.spec.channels,
            "device": str(self.device),
            "parameter_count": int(sum(parameter.numel() for parameter in self.generator.parameters())),
        }

    @torch.inference_mode()
    def sample(self, *, prompt: str, sample_count: int = 16, seed: Optional[int] = None) -> torch.Tensor:
        generator = torch.Generator(device=str(self.device))
        generator.manual_seed(int(seed if seed is not None else _prompt_seed(prompt)))
        noise = torch.randn(int(sample_count), self.spec.latent_dim, generator=generator, device=self.device)
        return self.generator(noise)

    def render_to_path(self, *, prompt: str, output_path: Path, sample_count: int = 16, seed: Optional[int] = None) -> Path:
        images = self.sample(prompt=prompt, sample_count=sample_count, seed=seed)
        save_image_grid(images, output_path)
        return output_path

    def benchmark(self, *, sample_count: int = 64, seed: int = 1337) -> Dict[str, float]:
        images = self.sample(prompt=f"benchmark::{self.spec.key}", sample_count=sample_count, seed=seed).detach().cpu()
        images = (images + 1.0) / 2.0
        images = images.clamp(0.0, 1.0)
        pixel_std = float(images.std(dim=(1, 2, 3)).mean().item())
        contrast = float((images.amax(dim=(1, 2, 3)) - images.amin(dim=(1, 2, 3))).mean().item())
        edge_h = torch.abs(images[:, :, 1:, :] - images[:, :, :-1, :]).mean()
        edge_w = torch.abs(images[:, :, :, 1:] - images[:, :, :, :-1]).mean()
        edge_energy = float(((edge_h + edge_w) / 2.0).item())
        non_blank_rate = float(images.std(dim=(1, 2, 3)).gt(0.05).float().mean().item())
        flat = images.view(images.size(0), -1)
        diversity = float(torch.abs(flat[1:] - flat[:-1]).mean().item()) if flat.size(0) > 1 else 0.0
        score = float(
            (
                non_blank_rate
                + min(1.0, pixel_std / 0.28)
                + min(1.0, contrast / 0.72)
                + min(1.0, edge_energy / 0.18)
                + min(1.0, diversity / 0.22)
            )
            / 5.0
        )
        return {
            "specialist_score": round(score, 6),
            "non_blank_rate": round(non_blank_rate, 6),
            "pixel_std": round(pixel_std, 6),
            "contrast": round(contrast, 6),
            "edge_energy": round(edge_energy, 6),
            "diversity": round(diversity, 6),
        }
