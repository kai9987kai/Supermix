from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

try:
    from image_feature_utils import describe_image_for_text
    from image_recognition_model import CLASS_LABELS, SCIENCE_IMAGE_CLASSES, load_image_tensor
    from math_equation_model import heuristic_intent, solve_intent
except ImportError:  # pragma: no cover
    from .image_feature_utils import describe_image_for_text
    from .image_recognition_model import CLASS_LABELS, SCIENCE_IMAGE_CLASSES, load_image_tensor
    from .math_equation_model import heuristic_intent, solve_intent


OMNI_INTENTS: tuple[str, ...] = (
    "general",
    "coding",
    "creative",
    "comparison",
    "planning",
    "current_info",
    "model_selection",
    "math",
    "vision",
    "command",
    "image_prompt",
)

INTENT_TO_INDEX = {name: idx for idx, name in enumerate(OMNI_INTENTS)}


def build_char_vocab(texts: Sequence[str], *, min_frequency: int = 1) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for text in texts:
        for ch in str(text or "").lower():
            counts[ch] = counts.get(ch, 0) + 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for ch, count in sorted(counts.items()):
        if count >= min_frequency and ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def encode_text(text: str, vocab: Dict[str, int], max_len: int) -> List[int]:
    cooked = str(text or "").strip().lower()
    ids = [vocab.get(ch, 1) for ch in cooked[: int(max_len)]]
    if len(ids) < int(max_len):
        ids.extend([0] * (int(max_len) - len(ids)))
    return ids


class OmniCollectiveNet(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        num_intents: int,
        num_responses: int,
        num_vision_classes: int,
        embed_dim: int = 56,
        text_hidden: int = 96,
        image_channels: int = 24,
        fusion_hidden: int = 176,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.text_encoder = nn.GRU(embed_dim, text_hidden, batch_first=True, bidirectional=True)
        self.text_norm = nn.LayerNorm(text_hidden * 2)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, image_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(image_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(image_channels, image_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(image_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(image_channels * 2, image_channels * 3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(image_channels * 3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.image_norm = nn.LayerNorm(image_channels * 3)
        self.fusion = nn.Sequential(
            nn.Linear(text_hidden * 2 + image_channels * 3 + 1, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.18),
        )
        self.intent_head = nn.Linear(fusion_hidden, num_intents)
        self.response_head = nn.Linear(fusion_hidden, num_responses)
        self.vision_head = nn.Linear(fusion_hidden, num_vision_classes)

    def forward(self, token_ids: torch.Tensor, image_tensor: torch.Tensor, has_image: torch.Tensor) -> Dict[str, torch.Tensor]:
        embedded = self.embedding(token_ids)
        _, hidden = self.text_encoder(embedded)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        if hidden.dim() == 3:
            text_state = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            text_state = hidden
        text_state = self.text_norm(text_state)
        image_state = self.image_norm(self.image_encoder(image_tensor))
        fused = self.fusion(torch.cat([text_state, image_state, has_image.view(-1, 1)], dim=1))
        return {
            "intent": self.intent_head(fused),
            "response": self.response_head(fused),
            "vision": self.vision_head(fused),
        }


@dataclass
class OmniPrediction:
    intent: str
    response_text: str
    vision_concept: str
    vision_confidence: float
    top_vision: List[Dict[str, float]]


class OmniCollectiveEngine:
    def __init__(self, *, weights_path: Path, meta_path: Path, device: Optional[torch.device] = None) -> None:
        self.weights_path = Path(weights_path).resolve()
        self.meta_path = Path(meta_path).resolve()
        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.vocab = {str(key): int(value) for key, value in dict(self.meta.get("vocab") or {}).items()}
        self.responses = tuple(str(item) for item in list(self.meta.get("response_bank") or []))
        self.class_info = {
            str(key): dict(value)
            for key, value in dict(self.meta.get("class_info") or {}).items()
        }
        self.image_size = int(self.meta.get("image_size") or 96)
        self.max_len = int(self.meta.get("max_len") or 220)
        embed_dim = int(self.meta.get("embed_dim") or 56)
        text_hidden = int(self.meta.get("text_hidden") or 96)
        image_channels = int(self.meta.get("image_channels") or 24)
        fusion_hidden = int(self.meta.get("fusion_hidden") or 176)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = OmniCollectiveNet(
            vocab_size=max(len(self.vocab), 2),
            num_intents=len(OMNI_INTENTS),
            num_responses=max(len(self.responses), 1),
            num_vision_classes=len(SCIENCE_IMAGE_CLASSES),
            embed_dim=embed_dim,
            text_hidden=text_hidden,
            image_channels=image_channels,
            fusion_hidden=fusion_hidden,
        ).to(self.device)
        state = torch.load(self.weights_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def status(self) -> Dict[str, Any]:
        return {
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "responses": len(self.responses),
            "image_size": self.image_size,
            "device": str(self.device),
        }

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        ids = encode_text(prompt, self.vocab, self.max_len)
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def _encode_image(self, image_path: Optional[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if not image_path:
            size = int(self.image_size)
            return (
                torch.zeros((1, 3, size, size), dtype=torch.float32, device=self.device),
                torch.zeros((1,), dtype=torch.float32, device=self.device),
            )
        tensor = load_image_tensor(str(image_path), self.image_size).unsqueeze(0).to(self.device)
        return tensor, torch.ones((1,), dtype=torch.float32, device=self.device)

    def predict(self, prompt: str, image_path: Optional[str] = None) -> OmniPrediction:
        with torch.inference_mode():
            outputs = self.model(
                self._encode_prompt(prompt),
                *self._encode_image(image_path),
            )
            intent_idx = int(outputs["intent"][0].argmax(dim=0).item())
            response_idx = int(outputs["response"][0].argmax(dim=0).item()) if self.responses else 0
            vision_probs = torch.softmax(outputs["vision"][0], dim=0).detach().cpu()
        values, indices = torch.topk(vision_probs, k=min(3, len(SCIENCE_IMAGE_CLASSES)))
        vision_key = SCIENCE_IMAGE_CLASSES[int(indices[0].item())]
        top_vision = [
            {
                "concept": SCIENCE_IMAGE_CLASSES[int(index.item())],
                "confidence": round(float(value.item()), 4),
            }
            for value, index in zip(values, indices)
        ]
        return OmniPrediction(
            intent=OMNI_INTENTS[intent_idx],
            response_text=self.responses[response_idx] if self.responses else "",
            vision_concept=vision_key,
            vision_confidence=float(values[0].item()),
            top_vision=top_vision,
        )

    def answer(self, prompt: str, image_path: Optional[str] = None) -> str:
        prompt_text = str(prompt or "").strip()
        math_intent = heuristic_intent(prompt_text)
        prediction = self.predict(prompt_text, image_path=image_path)
        if math_intent:
            try:
                solved = solve_intent(prompt_text, math_intent)
                return str(solved.get("response") or prediction.response_text or "Math intent detected.")
            except Exception:
                return prediction.response_text or "This looks like a math problem, but I could not solve it cleanly."
        if image_path:
            info = self.class_info.get(prediction.vision_concept, {})
            descriptor = describe_image_for_text(str(image_path))
            lines = [
                f"Image read: {CLASS_LABELS.get(prediction.vision_concept, prediction.vision_concept.replace('_', ' '))} ({prediction.vision_confidence * 100:.1f}% confidence).",
            ]
            if str(info.get("caption") or "").strip():
                lines.append(f"Likely concept: {str(info.get('caption')).strip()}.")
            lines.append(f"Visual descriptor: {descriptor}.")
            alt = [
                f"{CLASS_LABELS.get(item['concept'], item['concept'].replace('_', ' '))} {item['confidence'] * 100:.1f}%"
                for item in prediction.top_vision[1:]
            ]
            if alt:
                lines.append("Other matches: " + ", ".join(alt) + ".")
            if prediction.response_text:
                lines.append(prediction.response_text)
            return "\n".join(lines)
        if prediction.intent == "current_info":
            return prediction.response_text or "Turn on Web Search when you need current external information."
        if prediction.intent == "command":
            return prediction.response_text or "I can ask the runtime to open Command Prompt when you explicitly request it."
        return prediction.response_text or "I did not have a strong fused answer for that prompt."
