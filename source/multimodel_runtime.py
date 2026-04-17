from __future__ import annotations

import gc
import hashlib
import io
import json
import logging
import re
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote
from urllib.request import urlopen

import torch
from PIL import Image

import chat_web_app
from chat_export import copy_generated_image, render_chat_transcript_image
from chat_image_variant_app import (
    DEFAULT_IMAGE_MODEL,
    DEFAULT_NEGATIVE_PROMPT,
    ImageVariantEngine,
)
from device_utils import configure_torch_runtime, resolve_device
from multimodel_catalog import (
    DEFAULT_COMMON_SUMMARY,
    DEFAULT_MODELS_DIR,
    ModelRecord,
    choose_auto_model,
    describe_model_artifact_name,
    discover_model_records,
)
from multimodel_memory import ConversationMemoryStore
from multimodel_tools import (
    CmdOpenTool,
    ToolEvent,
    WebSearchTool,
    format_tool_results,
    parse_tool_requests,
    should_offer_web_search,
    should_offer_open_cmd,
    strip_tool_calls,
)
from dcgan_image_model import DCGANImageEngine, DCGAN_SPECS, _find_generator_weights
from math_equation_model import MathEquationEngine, format_math_response
from mattergen_generation_model import MatterGenMicroEngine, format_mattergen_response
from protein_folding_model import ProteinFoldingEngine, format_protein_response
from three_d_generation_model import ThreeDGenerationEngine, format_3d_generation_response
from native_image_infer_v36 import ChampionNetFrontierCollectiveNativeImage, save_prompt_image as save_prompt_image_v36
from native_image_infer_v37_lite import ChampionNetUltraExpertNativeImageLite, save_prompt_image as save_prompt_image_v37
from native_image_infer_v38_xlite import ChampionNetUltraExpertNativeImageExtraLite, save_prompt_image as save_prompt_image_v38
from image_recognition_model import ScienceImageRecognitionEngine, looks_like_vision_prompt
from omni_collective_model import OmniCollectiveEngine
from omni_collective_v3_model import OmniCollectiveEngineV3
from omni_collective_v4_model import OmniCollectiveEngineV4
from omni_collective_v5_model import OmniCollectiveEngineV5
from omni_collective_v6_model import OmniCollectiveEngineV6
from omni_collective_v7_model import OmniCollectiveEngineV7
from omni_collective_v8_model import OmniCollectiveEngineV8
from omni_collective_v42_model import OmniCollectiveEngineV42
from omni_collective_v41_model import OmniCollectiveEngineV41
from omni_collective_v46_model import OmniCollectiveEngineV46
from omni_collective_v47_model import OmniCollectiveEnginev47, OmniPredictionv47
from omni_collective_v48_model import OmniCollectiveEnginev48, OmniPredictionv48
from run import safe_load_state_dict


@dataclass
class ChatResult:
    kind: str
    model_key: str
    model_label: str
    route_reason: str
    response: str = ""
    timing: Optional[Dict[str, Any]] = None
    image_url: str = ""
    output_path: str = ""
    prompt_used: str = ""
    refined_prompt: str = ""
    agent_trace: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "kind": self.kind,
            "model_key": self.model_key,
            "model_label": self.model_label,
            "route_reason": self.route_reason,
            "response": self.response,
            "timing": self.timing or {},
            "image_url": self.image_url,
            "output_path": self.output_path,
            "prompt_used": self.prompt_used,
            "refined_prompt": self.refined_prompt,
            "agent_trace": self.agent_trace or {},
        }


def _safe_slug(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(text or ""))
    cooked = "-".join(part for part in cleaned.split("-") if part)
    return cooked[:72] or "artifact"


def _trim_text(text: str, limit: int = 320) -> str:
    cooked = " ".join(str(text or "").strip().split())
    return cooked[:limit]


def _extract_labeled_section(text: str, label: str) -> str:
    cooked = str(text or "")
    if not cooked.strip():
        return ""
    pattern = re.compile(
        rf"(?ims)^\s*{re.escape(label)}\s*:\s*(.*?)\s*(?=^\s*[A-Z_ ]+\s*:|\Z)"
    )
    match = pattern.search(cooked)
    if match:
        return " ".join(match.group(1).strip().split())
    return ""


def _parse_yes_no_section(text: str, label: str) -> Optional[bool]:
    section = _extract_labeled_section(text, label)
    if not section:
        return None
    lowered = section.strip().lower()
    if lowered.startswith(("yes", "true", "done", "complete")):
        return True
    if lowered.startswith(("no", "false", "not", "continue", "incomplete")):
        return False
    return None


def _safe_upload_name(filename: str) -> str:
    cooked = re.sub(r"[^A-Za-z0-9._-]+", "-", str(filename or "").strip()).strip(".-")
    return cooked[:96] or "upload.png"


DEFAULT_MODEL_STORE_REPO_ID = "Kai9987kai/supermix-model-zoo"
MODEL_STORE_CACHE_TTL_SECONDS = 300.0
LOOP_AGENT_DEFAULT_MAX_STEPS = 4
LOOP_AGENT_HARD_MAX_STEPS = 8


def _hf_dataset_file_url(repo_id: str, filename: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{quote(filename)}?download=true"


def _missing_zip_members(archive: zipfile.ZipFile, target: Path) -> List[str]:
    missing: List[str] = []
    for member in archive.infolist():
        if member.is_dir():
            continue
        candidate = target / member.filename
        if not candidate.exists():
            missing.append(member.filename)
    return missing


def _extract_zip_once(zip_path: Path, extraction_root: Path) -> Path:
    # If the artifact is already a raw model file (.pth), skip extraction
    if zip_path.suffix.lower() == ".pth":
        return zip_path.parent
    extraction_root.mkdir(parents=True, exist_ok=True)
    stamp = f"{zip_path.name}|{zip_path.stat().st_size}|{zip_path.stat().st_mtime_ns}"
    digest = hashlib.sha1(stamp.encode("utf-8")).hexdigest()[:12]
    target = extraction_root / f"{_safe_slug(zip_path.stem)}-{digest}"
    marker = target / ".extract_complete.json"
    expected_meta = {
        "zip_name": zip_path.name,
        "zip_size": zip_path.stat().st_size,
        "zip_mtime_ns": zip_path.stat().st_mtime_ns,
    }
    if marker.exists():
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        if payload == expected_meta:
            return target
    if target.exists():
        for child in sorted(target.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink(missing_ok=True)
            elif child.is_dir():
                try:
                    child.rmdir()
                except OSError:
                    pass
        try:
            target.rmdir()
        except OSError:
            pass
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target)
        missing = _missing_zip_members(archive, target)
        if missing:
            for member in missing:
                archive.extract(member, target)
            missing = _missing_zip_members(archive, target)
        if missing:
            raise FileNotFoundError(
                f"Extraction of {zip_path.name} is incomplete; missing {len(missing)} file(s), including {missing[:3]}"
            )
    marker.write_text(json.dumps(expected_meta, indent=2), encoding="utf-8")
    return target


def _find_matching_file(root: Path, preferred_names: Tuple[str, ...], suffix: str) -> Optional[Path]:
    if preferred_names:
        for name in preferred_names:
            candidate = root / name
            if candidate.exists():
                return candidate
        for name in preferred_names:
            matches = list(root.rglob(Path(name).name))
            if matches:
                return sorted(matches)[0]
    matches = sorted(root.rglob(f"*{suffix}"))
    return matches[0] if matches else None


def _find_adapter_dir(root: Path, markers: Tuple[str, ...]) -> Path:
    for marker in markers:
        candidate = root / marker
        if candidate.exists():
            return candidate.parent.resolve()
    matches = sorted(root.rglob("adapter_config.json"))
    for match in matches:
        if (match.parent / "adapter_model.safetensors").exists():
            return match.parent.resolve()
    raise FileNotFoundError(f"Could not find a Qwen adapter directory under {root}")


def _compose_text_prompt(prompt: str, settings: Dict[str, Any]) -> str:
    blocks: List[str] = []
    if str(settings.get("tool_instruction") or "").strip():
        blocks.append(str(settings.get("tool_instruction")).strip())
    if str(settings.get("memory_context") or "").strip():
        blocks.append(str(settings.get("memory_context")).strip())
    if str(settings.get("tool_context") or "").strip():
        blocks.append("Tool results:\n" + str(settings.get("tool_context")).strip())
    if str(settings.get("consultation_context") or "").strip():
        blocks.append("Cross-model consultation:\n" + str(settings.get("consultation_context")).strip())
    if not blocks:
        return str(prompt)
    blocks.append("Current user request:\n" + str(prompt).strip())
    blocks.append("Answer the current user request directly and use the context above when it is relevant.")
    return "\n\n".join(blocks)


def _compose_image_prompt(prompt: str, settings: Dict[str, Any]) -> str:
    notes: List[str] = []
    if str(settings.get("memory_context") or "").strip():
        notes.append(str(settings.get("memory_context")).strip())
    if str(settings.get("consultation_context") or "").strip():
        notes.append("Prompt planning notes:\n" + str(settings.get("consultation_context")).strip())
    if not notes:
        return str(prompt)
    return "\n\n".join([str(prompt).strip(), *notes])


class BaseBackend:
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        self.record = record
        self.extracted_dir = extracted_dir
        self.generated_dir = generated_dir
        self.generated_dir.mkdir(parents=True, exist_ok=True)

    def status(self) -> Dict[str, Any]:
        raise NotImplementedError

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        raise NotImplementedError

    def generate_image(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        raise RuntimeError(f"{self.record.label} does not support image generation")

    def clear(self, session_id: str) -> None:
        return None

    def unload(self) -> None:
        return None


class ChampionChatBackend(BaseBackend):
    def __init__(
        self,
        record: ModelRecord,
        extracted_dir: Path,
        generated_dir: Path,
        device: Any,
        device_info: Dict[str, Any],
    ) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        weights_path = _find_matching_file(extracted_dir, record.preferred_weights, ".pth")
        meta_path = _find_matching_file(extracted_dir, record.preferred_meta, ".json")
        if weights_path is None or meta_path is None:
            raise FileNotFoundError(f"Missing weights/meta for {record.label} in {extracted_dir}")
        self.weights_path = weights_path.resolve()
        self.meta_path = meta_path.resolve()
        self.engine = chat_web_app.Engine(
            device,
            device_info,
            {
                "model_size": "auto",
                "max_turns": 2,
                "top_labels": 3,
                "pool_mode": "all",
                "response_temperature": 0.08,
                "temperature": 0.0,
                "style_mode": "auto",
                "creativity": 0.25,
            },
        )
        self.engine.load(str(self.weights_path), str(self.meta_path))

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "champion_chat",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "runtime": self.engine.status(),
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        effective_prompt = _compose_text_prompt(prompt, settings)
        payload = self.engine.chat(
            session_id=session_id,
            user_text=effective_prompt,
            style_mode=str(settings.get("style_mode") or "auto"),
            response_temperature=float(settings.get("response_temperature") or 0.08),
            show_top_responses=int(settings.get("show_top_responses") or 0),
        )
        return ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=str(payload.get("response") or ""),
            timing=dict(payload.get("timing_ms") or {}),
            prompt_used=effective_prompt,
        )

    def clear(self, session_id: str) -> None:
        self.engine.clear(session_id)

    def unload(self) -> None:
        if hasattr(self.engine, "model"):
            self.engine.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ImageWrapperBackend(ChampionChatBackend):
    def __init__(
        self,
        record: ModelRecord,
        extracted_dir: Path,
        generated_dir: Path,
        device: Any,
        device_info: Dict[str, Any],
    ) -> None:
        super().__init__(record, extracted_dir, generated_dir, device, device_info)
        self.image_engine = ImageVariantEngine(
            text_engine=self.engine,
            output_dir=self.generated_dir / self.record.key,
            default_image_model=DEFAULT_IMAGE_MODEL,
            default_negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        )

    def status(self) -> Dict[str, Any]:
        payload = super().status()
        payload["backend"] = "image_wrapper"
        payload["image_status"] = self.image_engine.status()
        return payload

    def generate_image(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        effective_prompt = _compose_image_prompt(prompt, settings)
        result = self.image_engine.generate_image(
            prompt=effective_prompt,
            image_model=str(settings.get("image_model") or DEFAULT_IMAGE_MODEL),
            negative_prompt=str(settings.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT),
            style=str(settings.get("image_style") or "auto"),
            width=int(settings.get("image_width") or 512),
            height=int(settings.get("image_height") or 512),
            steps=int(settings.get("image_steps") or 2),
            seed=None if settings.get("image_seed") in (None, "") else int(settings.get("image_seed")),
            guidance_scale=float(settings.get("guidance_scale") or 0.0),
            use_text_refiner=bool(settings.get("use_text_refiner", True)),
        )
        return ChatResult(
            kind="image",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            timing={"total_ms": result.get("timing_ms")},
            image_url=str(result.get("image_url") or ""),
            output_path=str(result.get("output_path") or ""),
            prompt_used=str(result.get("prompt_used") or effective_prompt),
            refined_prompt=str(result.get("refined_prompt") or ""),
        )


class QwenBackend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        import qwen_chat_web_app  # lazy import so source runtime can start without the Qwen stack installed

        self._qwen = qwen_chat_web_app
        self.adapter_dir = _find_adapter_dir(extracted_dir, record.adapter_markers)
        self.device = self._qwen.resolve_device("auto")
        self.base_model = self._qwen.resolve_base_model_path("", self.adapter_dir)
        self.engine = self._qwen.load_engine(
            base_model=self.base_model,
            adapter_dir=self.adapter_dir,
            device=self.device,
        )

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "qwen_adapter",
            "record": self.record.to_dict(),
            "adapter_dir": str(self.adapter_dir),
            "base_model": str(self.base_model),
            "runtime": self.engine.status(),
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        style_mode = str(settings.get("style_mode") or "auto").strip().lower()
        preset = {
            "concise": "direct",
            "creative": "creative",
            "analyst": "reasoning",
            "coding": "coding",
        }.get(style_mode, "balanced")
        effective_prompt = _compose_text_prompt(prompt, settings)
        payload = self.engine.chat(
            session_id=session_id,
            user_text=effective_prompt,
            max_new_tokens=int(settings.get("max_new_tokens") or 160),
            temperature=float(settings.get("temperature") or 0.18),
            top_p=float(settings.get("top_p") or 0.92),
            preset=preset,
            system_hint=str(settings.get("system_hint") or ""),
        )
        return ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=str(payload.get("response") or ""),
            timing=dict(payload.get("timing") or {}),
            prompt_used=effective_prompt,
        )

    def clear(self, session_id: str) -> None:
        self.engine.clear(session_id)

    def unload(self) -> None:
        if hasattr(self.engine, "model"):
            self.engine.model = None
        if hasattr(self.engine, "tokenizer"):
            self.engine.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class NativeImageBackend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path, device: Any) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        weights_path = _find_matching_file(extracted_dir, record.preferred_weights, ".pth")
        meta_path = _find_matching_file(extracted_dir, record.preferred_meta, ".json")
        if weights_path is None or meta_path is None:
            raise FileNotFoundError(f"Missing weights/meta for {record.label} in {extracted_dir}")
        self.weights_path = weights_path.resolve()
        self.meta_path = meta_path.resolve()
        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.device = torch.device("cuda" if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu")
        self.model, self.save_fn = self._load_model()

    def _load_model(self) -> Tuple[Any, Any]:
        image_size = int(self.meta.get("image_size") or 64)
        if self.record.key == "v36_native":
            model = ChampionNetFrontierCollectiveNativeImage(image_size=image_size).to(self.device).eval()
            state_dict = safe_load_state_dict(str(self.weights_path))
            model.load_state_dict(state_dict, strict=False)
            return model, save_prompt_image_v36
        if self.record.key == "v37_native_lite":
            model = ChampionNetUltraExpertNativeImageLite(image_size=image_size).to(self.device).eval()
            model.load_state_dict(safe_load_state_dict(str(self.weights_path)), strict=True)
            return model, save_prompt_image_v37
        model = ChampionNetUltraExpertNativeImageExtraLite(image_size=image_size).to(self.device).eval()
        model.load_state_dict(safe_load_state_dict(str(self.weights_path)), strict=True)
        return model, save_prompt_image_v38

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "native_image",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "device": str(self.device),
            "image_size": int(self.meta.get("image_size") or 64),
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        return self.generate_image(session_id=session_id, prompt=prompt, settings=settings)

    def generate_image(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        effective_prompt = _compose_image_prompt(prompt, settings)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = self.generated_dir / self.record.key
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stamp}_{_safe_slug(effective_prompt)[:40]}.png"
        feature_mode = str(self.meta.get("feature_mode") or "context_mix_v4")
        started = time.perf_counter()
        self.save_fn(self.model, str(effective_prompt), str(out_path), feature_mode=feature_mode, device=self.device)
        total_ms = round((time.perf_counter() - started) * 1000.0, 1)
        return ChatResult(
            kind="image",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            timing={"total_ms": total_ms},
            image_url=f"/generated/{self.record.key}/{out_path.name}",
            output_path=str(out_path),
            prompt_used=str(effective_prompt),
            refined_prompt=str(settings.get("consultation_context") or ""),
        )

    def unload(self) -> None:
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class DCGANImageBackend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        spec = DCGAN_SPECS.get(record.key)
        if spec is None:
            raise KeyError(f"Unsupported DCGAN model key: {record.key}")
        self.spec = spec
        self.weights_path = _find_generator_weights(extracted_dir, record.preferred_weights or spec.preferred_weights)
        self.engine = DCGANImageEngine(key=record.key, weights_path=self.weights_path)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "dcgan_image",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "runtime": self.engine.status(),
        }

    def generate_image(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        del session_id
        effective_prompt = _compose_image_prompt(prompt, settings)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = self.generated_dir / self.record.key
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stamp}_{_safe_slug(effective_prompt)[:40]}.png"
        sample_count = 16
        lowered = effective_prompt.lower()
        if "single" in lowered or "one sample" in lowered:
            sample_count = 1
        elif "large grid" in lowered or "many samples" in lowered:
            sample_count = 25
        started = time.perf_counter()
        self.engine.render_to_path(prompt=effective_prompt, output_path=out_path, sample_count=sample_count)
        total_ms = round((time.perf_counter() - started) * 1000.0, 1)
        return ChatResult(
            kind="image",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=(
                f"{self.record.label} is an unconditional GAN, so the prompt only seeds the sample grid rather than controlling content directly."
            ),
            timing={"total_ms": total_ms},
            image_url=f"/generated/{self.record.key}/{out_path.name}",
            output_path=str(out_path),
            prompt_used=str(effective_prompt),
            refined_prompt=str(settings.get("consultation_context") or ""),
        )


class MathEquationBackend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        weights_path = _find_matching_file(extracted_dir, record.preferred_weights, ".pth")
        meta_path = _find_matching_file(extracted_dir, record.preferred_meta, ".json")
        if weights_path is None or meta_path is None:
            raise FileNotFoundError(f"Missing math weights/meta for {record.label} in {extracted_dir}")
        self.weights_path = weights_path.resolve()
        self.meta_path = meta_path.resolve()
        self.engine = MathEquationEngine(weights_path=self.weights_path, meta_path=self.meta_path)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "math_equation",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "runtime": self.engine.status(),
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        solved = self.engine.solve(prompt)
        return ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=format_math_response(solved),
            timing={},
            prompt_used=str(prompt),
        )


class ProteinFoldingBackend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        weights_path = _find_matching_file(extracted_dir, record.preferred_weights, ".pth")
        meta_path = _find_matching_file(extracted_dir, record.preferred_meta, ".json")
        if weights_path is None or meta_path is None:
            raise FileNotFoundError(f"Missing protein-folding weights/meta for {record.label} in {extracted_dir}")
        self.weights_path = weights_path.resolve()
        self.meta_path = meta_path.resolve()
        self.engine = ProteinFoldingEngine(weights_path=self.weights_path, meta_path=self.meta_path)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "protein_folding",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "runtime": self.engine.status(),
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        prediction = self.engine.predict(prompt)
        response = format_protein_response(self.engine.answer(prompt), prediction)
        return ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=response,
            timing={},
            prompt_used=str(prompt),
        )


class MatterGenGenerationBackend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        weights_path = _find_matching_file(extracted_dir, record.preferred_weights, ".pth")
        meta_path = _find_matching_file(extracted_dir, record.preferred_meta, ".json")
        if weights_path is None or meta_path is None:
            raise FileNotFoundError(f"Missing MatterGen weights/meta for {record.label} in {extracted_dir}")
        self.weights_path = weights_path.resolve()
        self.meta_path = meta_path.resolve()
        self.engine = MatterGenMicroEngine(weights_path=self.weights_path, meta_path=self.meta_path)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "mattergen_generation",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "runtime": self.engine.status(),
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        prediction = self.engine.predict(prompt)
        response = format_mattergen_response(self.engine.answer(prompt), prediction)
        return ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=response,
            timing={},
            prompt_used=str(prompt),
        )


class ThreeDGenerationBackend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        weights_path = _find_matching_file(extracted_dir, record.preferred_weights, ".pth")
        meta_path = _find_matching_file(extracted_dir, record.preferred_meta, ".json")
        if weights_path is None or meta_path is None:
            raise FileNotFoundError(f"Missing 3D-generation weights/meta for {record.label} in {extracted_dir}")
        self.weights_path = weights_path.resolve()
        self.meta_path = meta_path.resolve()
        self.engine = ThreeDGenerationEngine(weights_path=self.weights_path, meta_path=self.meta_path)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "three_d_generation",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "runtime": self.engine.status(),
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        prediction = self.engine.predict(prompt)
        response = format_3d_generation_response(self.engine.answer(prompt), prediction)
        return ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=response,
            timing={},
            prompt_used=str(prompt),
        )


class ImageRecognitionBackend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        weights_path = _find_matching_file(extracted_dir, record.preferred_weights, ".pth")
        meta_path = _find_matching_file(extracted_dir, record.preferred_meta, ".json")
        if weights_path is None or meta_path is None:
            raise FileNotFoundError(f"Missing image-recognition weights/meta for {record.label} in {extracted_dir}")
        self.weights_path = weights_path.resolve()
        self.meta_path = meta_path.resolve()
        self.engine = ScienceImageRecognitionEngine(weights_path=self.weights_path, meta_path=self.meta_path)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "image_recognition",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "runtime": self.engine.status(),
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        image_path = str(settings.get("uploaded_image_path") or "").strip()
        if not image_path:
            response = "Upload an image first, then I can identify the science concept and explain the visual clues."
        else:
            response = self.engine.answer(prompt, image_path=image_path)
        return ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=response,
            timing={},
            prompt_used=str(prompt),
        )


class OmniCollectiveBackend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        weights_path = _find_matching_file(extracted_dir, record.preferred_weights, ".pth")
        meta_path = _find_matching_file(extracted_dir, record.preferred_meta, ".json")
        if weights_path is None or meta_path is None:
            raise FileNotFoundError(f"Missing omnibus weights/meta for {record.label} in {extracted_dir}")
        self.weights_path = weights_path.resolve()
        self.meta_path = meta_path.resolve()
        self.engine = OmniCollectiveEngine(weights_path=self.weights_path, meta_path=self.meta_path)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "omni_collective",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "runtime": self.engine.status(),
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        image_path = str(settings.get("uploaded_image_path") or "").strip()
        response = self.engine.answer(prompt, image_path=image_path or None)
        return ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=response,
            timing={},
            prompt_used=str(prompt),
        )


class OmniCollectiveV3Backend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        weights_path = _find_matching_file(extracted_dir, record.preferred_weights, ".pth")
        meta_path = _find_matching_file(extracted_dir, record.preferred_meta, ".json")
        if weights_path is None or meta_path is None:
            raise FileNotFoundError(f"Missing omnibus weights/meta for {record.label} in {extracted_dir}")
        self.weights_path = weights_path.resolve()
        self.meta_path = meta_path.resolve()
        self.engine = OmniCollectiveEngineV3(weights_path=self.weights_path, meta_path=self.meta_path)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "omni_collective_v3",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "runtime": {
                "device": str(self.engine.device),
                "image_size": int(self.engine.image_size),
                "vocab_size": len(self.engine.vocab),
                "response_count": len(self.engine.responses),
            },
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        image_path = str(settings.get("uploaded_image_path") or "").strip()
        effective_prompt = _compose_text_prompt(prompt, settings)
        response = self.engine.answer(effective_prompt, image_path=image_path or None)
        return ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=response,
            timing={},
            prompt_used=effective_prompt,
        )


class OmniCollectiveV4Backend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        weights_path = _find_matching_file(extracted_dir, record.preferred_weights, ".pth")
        meta_path = _find_matching_file(extracted_dir, record.preferred_meta, ".json")
        if weights_path is None or meta_path is None:
            raise FileNotFoundError(f"Missing omnibus weights/meta for {record.label} in {extracted_dir}")
        self.weights_path = weights_path.resolve()
        self.meta_path = meta_path.resolve()
        self.engine = OmniCollectiveEngineV4(weights_path=self.weights_path, meta_path=self.meta_path)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "omni_collective_v4",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "runtime": {
                "device": str(self.engine.device),
                "image_size": int(self.engine.image_size),
                "vocab_size": len(self.engine.vocab),
                "response_count": len(self.engine.responses),
            },
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        image_path = str(settings.get("uploaded_image_path") or "").strip()
        effective_prompt = _compose_text_prompt(prompt, settings)
        response = self.engine.answer(effective_prompt, image_path=image_path or None)
        return ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=response,
            timing={},
            prompt_used=effective_prompt,
        )


class OmniCollectiveV5Backend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        weights_path = _find_matching_file(extracted_dir, record.preferred_weights, ".pth")
        meta_path = _find_matching_file(extracted_dir, record.preferred_meta, ".json")
        if weights_path is None or meta_path is None:
            raise FileNotFoundError(f"Missing omnibus weights/meta for {record.label} in {extracted_dir}")
        self.weights_path = weights_path.resolve()
        self.meta_path = meta_path.resolve()
        self.engine = OmniCollectiveEngineV5(weights_path=self.weights_path, meta_path=self.meta_path)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "omni_collective_v5",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "runtime": {
                "device": str(self.engine.device),
                "image_size": int(self.engine.image_size),
                "vocab_size": len(self.engine.vocab),
                "response_count": len(self.engine.responses),
                "deliberation_passes": int(self.engine.deliberation_passes),
            },
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        image_path = str(settings.get("uploaded_image_path") or "").strip()
        effective_prompt = _compose_text_prompt(prompt, settings)
        response = self.engine.answer(effective_prompt, image_path=image_path or None)
        return ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=response,
            timing={},
            prompt_used=effective_prompt,
        )


class OmniCollectiveV6Backend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        weights_path = _find_matching_file(extracted_dir, record.preferred_weights, ".pth")
        meta_path = _find_matching_file(extracted_dir, record.preferred_meta, ".json")
        if weights_path is None or meta_path is None:
            raise FileNotFoundError(f"Missing omnibus weights/meta for {record.label} in {extracted_dir}")
        self.weights_path = weights_path.resolve()
        self.meta_path = meta_path.resolve()
        self.engine = OmniCollectiveEngineV6(weights_path=self.weights_path, meta_path=self.meta_path)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "omni_collective_v6",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "runtime": {
                "device": str(self.engine.device),
                "image_size": int(self.engine.image_size),
                "vocab_size": len(self.engine.vocab),
                "response_count": len(self.engine.responses),
                "deliberation_passes": int(self.engine.deliberation_passes),
                "minimum_passes": int(self.engine.minimum_passes),
            },
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        image_path = str(settings.get("uploaded_image_path") or "").strip()
        effective_prompt = _compose_text_prompt(prompt, settings)
        response = self.engine.answer(effective_prompt, image_path=image_path or None)
        return ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=response,
            timing={},
            prompt_used=effective_prompt,
        )


class OmniCollectiveV7Backend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        weights_path = _find_matching_file(extracted_dir, record.preferred_weights, ".pth")
        meta_path = _find_matching_file(extracted_dir, record.preferred_meta, ".json")
        if weights_path is None or meta_path is None:
            raise FileNotFoundError(f"Missing omnibus weights/meta for {record.label} in {extracted_dir}")
        self.weights_path = weights_path.resolve()
        self.meta_path = meta_path.resolve()
        self.engine = OmniCollectiveEngineV7(weights_path=self.weights_path, meta_path=self.meta_path)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "omni_collective_v7",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "runtime": {
                "device": str(self.engine.device),
                "image_size": int(self.engine.image_size),
                "vocab_size": len(self.engine.vocab),
                "response_count": len(self.engine.responses),
                "deliberation_passes": int(self.engine.deliberation_passes),
                "minimum_passes": int(self.engine.minimum_passes),
            },
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        image_path = str(settings.get("uploaded_image_path") or "").strip()
        effective_prompt = _compose_text_prompt(prompt, settings)
        response = self.engine.answer(effective_prompt, image_path=image_path or None)
        return ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=response,
            timing={},
            prompt_used=effective_prompt,
        )


class OmniCollectiveV8Backend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        weights_path = _find_matching_file(extracted_dir, record.preferred_weights, ".pth")
        meta_path = _find_matching_file(extracted_dir, record.preferred_meta, ".json")
        if weights_path is None or meta_path is None:
            raise FileNotFoundError(f"Missing omnibus weights/meta for {record.label} in {extracted_dir}")
        self.weights_path = weights_path.resolve()
        self.meta_path = meta_path.resolve()
        self.engine = OmniCollectiveEngineV8(weights_path=self.weights_path, meta_path=self.meta_path)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "omni_collective_v8",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "runtime": {
                "device": str(self.engine.device),
                "image_size": int(self.engine.image_size),
                "vocab_size": len(self.engine.vocab),
                "response_count": len(self.engine.responses),
                "deliberation_passes": int(self.engine.deliberation_passes),
                "minimum_passes": int(self.engine.minimum_passes),
            },
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        image_path = str(settings.get("uploaded_image_path") or "").strip()
        effective_prompt = _compose_text_prompt(prompt, settings)
        response = self.engine.answer(effective_prompt, image_path=image_path or None)
        return ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=response,
            timing={},
            prompt_used=effective_prompt,
        )


class OmniCollectiveV41Backend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        weights_path = _find_matching_file(extracted_dir, record.preferred_weights, ".pth")
        meta_path = _find_matching_file(extracted_dir, record.preferred_meta, ".json")
        if weights_path is None or meta_path is None:
            raise FileNotFoundError(f"Missing omnibus weights/meta for {record.label} in {extracted_dir}")
        self.weights_path = weights_path.resolve()
        self.meta_path = meta_path.resolve()
        self.engine = OmniCollectiveEngineV41(weights_path=self.weights_path, meta_path=self.meta_path)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "omni_collective_v41",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "runtime": {
                "device": str(self.engine.device),
                "image_size": int(self.engine.image_size),
                "vocab_size": len(self.engine.vocab),
                "response_count": len(self.engine.responses),
                "deliberation_passes": int(self.engine.deliberation_passes),
                "minimum_passes": int(self.engine.minimum_passes),
                "grounding_threshold": float(self.engine.grounding_threshold),
            },
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        image_path = str(settings.get("uploaded_image_path") or "").strip()
        effective_prompt = _compose_text_prompt(prompt, settings)
        response = self.engine.answer(effective_prompt, image_path=image_path or None)
        return ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=response,
            timing={},
            prompt_used=effective_prompt,
        )


class OmniCollectiveV42Backend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        weights_path = _find_matching_file(extracted_dir, record.preferred_weights, ".pth")
        meta_path = _find_matching_file(extracted_dir, record.preferred_meta, ".json")
        if weights_path is None or meta_path is None:
            raise FileNotFoundError(f"Missing omnibus weights/meta for {record.label} in {extracted_dir}")
        self.weights_path = weights_path.resolve()
        self.meta_path = meta_path.resolve()
        self.engine = OmniCollectiveEngineV42(weights_path=self.weights_path, meta_path=self.meta_path)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "omni_collective_v42",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "runtime": {
                "device": str(self.engine.device),
                "image_size": int(self.engine.image_size),
                "vocab_size": len(self.engine.vocab),
                "response_count": len(self.engine.responses),
                "deliberation_passes": int(self.engine.deliberation_passes),
                "minimum_passes": int(self.engine.minimum_passes),
                "grounding_threshold": float(self.engine.grounding_threshold),
            },
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        image_path = str(settings.get("uploaded_image_path") or "").strip()
        effective_prompt = _compose_text_prompt(prompt, settings)
        response = self.engine.answer(effective_prompt, image_path=image_path or None)
        return ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=response,
            timing={},
            prompt_used=effective_prompt,
        )


class OmniCollectiveV46Backend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        weights_path = _find_matching_file(extracted_dir, record.preferred_weights, ".pth")
        meta_path = _find_matching_file(extracted_dir, record.preferred_meta, ".json")
        if weights_path is None or meta_path is None:
            raise FileNotFoundError(f"Missing omnibus weights/meta for {record.label} in {extracted_dir}")
        self.weights_path = weights_path.resolve()
        self.meta_path = meta_path.resolve()
        self.engine = OmniCollectiveEngineV46(weights_path=self.weights_path, meta_path=self.meta_path)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "omni_collective_v46",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "runtime": {
                "device": str(self.engine.device),
                "image_size": int(self.engine.image_size),
                "vocab_size": len(self.engine.vocab),
                "response_count": len(self.engine.responses),
                "grounding_threshold": float(self.engine.grounding_threshold),
            },
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        image_path = str(settings.get("uploaded_image_path") or "").strip()
        effective_prompt = _compose_text_prompt(prompt, settings)
        pred = self.engine.predict(effective_prompt, image_path=image_path or None)
        timing = {
            "reasoning_passes": pred.reasoning_passes,
            "planned_budget": pred.planned_budget,
            "difficulty_estimate": pred.difficulty_estimate,
            "mixture_of_depths_skipped": pred.mixture_of_depths_skipped,
            "graph_synthesis_applied": pred.graph_synthesis_applied,
            "continuous_latent_active": pred.continuous_latent_active,
        }
        res = ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=pred.response_text,
            timing=timing,
            prompt_used=effective_prompt,
        )
        res.agent_trace = {
            "reasoning_mode": pred.reasoning_mode,
            "speculative_accepted": pred.speculative_accepted,
            "adversarial_verified": pred.adversarial_verified,
            "grpo_group_size": pred.grpo_group_size,
        }
        return res


class OmniCollectiveV47Backend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        weights_path = _find_matching_file(extracted_dir, record.preferred_weights, ".pth")
        meta_path = _find_matching_file(extracted_dir, record.preferred_meta, ".json")
        if weights_path is None or meta_path is None:
            raise FileNotFoundError(f"Missing omnibus weights/meta for {record.label} in {extracted_dir}")
        self.weights_path = weights_path.resolve()
        self.meta_path = meta_path.resolve()
        self.engine = OmniCollectiveEnginev47(weights_path=self.weights_path, meta_path=self.meta_path)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "omni_collective_v47",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "runtime": {
                "device": str(self.engine.device),
                "image_size": int(self.engine.image_size),
                "vocab_size": len(self.engine.vocab),
                "response_count": len(self.engine.responses),
                "grounding_threshold": float(self.engine.grounding_threshold),
            },
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        image_path = str(settings.get("uploaded_image_path") or "").strip()
        effective_prompt = _compose_text_prompt(prompt, settings)
        pred = self.engine.predict(effective_prompt, image_path=image_path or None)
        timing = {
            "reasoning_passes": pred.reasoning_passes,
            "planned_budget": pred.planned_budget,
            "difficulty_estimate": pred.difficulty_estimate,
            "mixture_of_depths_skipped": pred.mixture_of_depths_skipped,
            "graph_synthesis_applied": pred.graph_synthesis_applied,
            "continuous_latent_active": pred.continuous_latent_active,
        }
        res = ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=pred.response_text,
            timing=timing,
            prompt_used=effective_prompt,
        )
        res.agent_trace = {
            "reasoning_mode": pred.reasoning_mode,
            "speculative_accepted": pred.speculative_accepted,
            "adversarial_verified": pred.adversarial_verified,
            "grpo_group_size": pred.grpo_group_size,
            "mixture_of_depths_skipped": pred.mixture_of_depths_skipped,
            "graph_synthesis_applied": pred.graph_synthesis_applied,
            "continuous_latent_active": pred.continuous_latent_active,
        }
        return res


class OmniCollectiveV48Backend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        weights_path = _find_matching_file(extracted_dir, record.preferred_weights, ".pth")
        meta_path = _find_matching_file(extracted_dir, record.preferred_meta, ".json")
        if weights_path is None or meta_path is None:
            raise FileNotFoundError(f"Missing omnibus weights/meta for {record.label} in {extracted_dir}")
        self.weights_path = weights_path.resolve()
        self.meta_path = meta_path.resolve()
        self.engine = OmniCollectiveEnginev48(weights_path=self.weights_path, meta_path=self.meta_path)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "omni_collective_v48",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "runtime": {
                "device": str(self.engine.device),
                "image_size": int(self.engine.image_size),
                "vocab_size": len(self.engine.vocab),
                "response_count": len(self.engine.responses),
                "grounding_threshold": float(self.engine.grounding_threshold),
            },
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        image_path = str(settings.get("uploaded_image_path") or "").strip()
        effective_prompt = _compose_text_prompt(prompt, settings)
        pred = self.engine.predict(effective_prompt, image_path=image_path or None)
        timing = {
            "reasoning_passes": pred.reasoning_passes,
            "planned_budget": pred.planned_budget,
            "difficulty_estimate": pred.difficulty_estimate,
            "mixture_of_depths_skipped": pred.mixture_of_depths_skipped,
            "graph_synthesis_applied": pred.graph_synthesis_applied,
            "continuous_latent_active": pred.continuous_latent_active,
            "hierarchical_routing_applied": pred.hierarchical_routing_applied,
            "agat_node_count": pred.agat_node_count,
        }
        res = ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=pred.response_text,
            timing=timing,
            prompt_used=effective_prompt,
        )
        res.agent_trace = {
            "reasoning_mode": pred.reasoning_mode,
            "speculative_accepted": pred.speculative_accepted,
            "adversarial_verified": pred.adversarial_verified,
            "dpo_alignment_score": pred.dpo_alignment_score,
        }
        return res


class UnifiedModelManager:
    def __init__(
        self,
        records: Tuple[ModelRecord, ...],
        extraction_root: Path,
        generated_dir: Path,
        device_preference: str = "cuda,npu,xpu,cpu,dml,mps",
        models_dir: Path = DEFAULT_MODELS_DIR,
        common_summary_path: Path = DEFAULT_COMMON_SUMMARY,
        model_store_repo_id: str = DEFAULT_MODEL_STORE_REPO_ID,
        backend_cache_size: int = 1,
    ) -> None:
        configure_torch_runtime(
            torch_num_threads=0,
            torch_interop_threads=0,
            allow_tf32=True,
            matmul_precision="high",
        )
        device, device_info = resolve_device("auto", preference=device_preference)
        self.records = list(records)
        self.record_map = {record.key: record for record in self.records}
        self.models_dir = Path(models_dir).resolve()
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.common_summary_path = Path(common_summary_path).resolve()
        self.model_store_repo_id = str(model_store_repo_id or DEFAULT_MODEL_STORE_REPO_ID).strip() or DEFAULT_MODEL_STORE_REPO_ID
        self.extraction_root = extraction_root.resolve()
        self.generated_dir = generated_dir.resolve()
        self.uploads_dir = self.extraction_root.parent / "uploads"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir = self.extraction_root.parent / "exports"
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.device_info = device_info
        self.selected_model_key = "auto"
        self.last_route_reason = ""
        self._backend_cache_size = max(1, int(backend_cache_size))
        self._backend: Optional[BaseBackend] = None
        self._backend_key = ""
        self._backend_cache: Dict[str, BaseBackend] = {}
        self._backend_lru: List[str] = []
        self._backend_init_failures: Dict[str, str] = {}
        self._lock = threading.RLock()
        self._model_store_manifest_cache: Optional[Dict[str, Any]] = None
        self._model_store_manifest_ts = 0.0
        self._model_store_jobs: Dict[str, Dict[str, Any]] = {}
        self.memory_store = ConversationMemoryStore(self.extraction_root.parent / "memory")
        self.web_search = WebSearchTool()
        self.cmd_open = CmdOpenTool()

    def _build_backend(self, record: ModelRecord) -> BaseBackend:
        extracted_dir = _extract_zip_once(record.zip_path, self.extraction_root)
        if record.kind == "champion_chat":
            return ChampionChatBackend(record, extracted_dir, self.generated_dir, self.device, self.device_info)
        if record.kind == "image_wrapper":
            return ImageWrapperBackend(record, extracted_dir, self.generated_dir, self.device, self.device_info)
        if record.kind == "native_image":
            return NativeImageBackend(record, extracted_dir, self.generated_dir, self.device)
        if record.kind == "dcgan_image":
            return DCGANImageBackend(record, extracted_dir, self.generated_dir)
        if record.kind == "math_equation":
            return MathEquationBackend(record, extracted_dir, self.generated_dir)
        if record.kind == "protein_folding":
            return ProteinFoldingBackend(record, extracted_dir, self.generated_dir)
        if record.kind == "mattergen_generation":
            return MatterGenGenerationBackend(record, extracted_dir, self.generated_dir)
        if record.kind == "three_d_generation":
            return ThreeDGenerationBackend(record, extracted_dir, self.generated_dir)
        if record.kind == "image_recognition":
            return ImageRecognitionBackend(record, extracted_dir, self.generated_dir)
        if record.kind == "omni_collective":
            return OmniCollectiveBackend(record, extracted_dir, self.generated_dir)
        if record.kind == "omni_collective_v3":
            return OmniCollectiveV3Backend(record, extracted_dir, self.generated_dir)
        if record.kind == "omni_collective_v4":
            return OmniCollectiveV4Backend(record, extracted_dir, self.generated_dir)
        if record.kind == "omni_collective_v5":
            return OmniCollectiveV5Backend(record, extracted_dir, self.generated_dir)
        if record.kind == "omni_collective_v6":
            return OmniCollectiveV6Backend(record, extracted_dir, self.generated_dir)
        if record.kind == "omni_collective_v7":
            return OmniCollectiveV7Backend(record, extracted_dir, self.generated_dir)
        if record.kind == "omni_collective_v8":
            return OmniCollectiveV8Backend(record, extracted_dir, self.generated_dir)
        if record.kind == "omni_collective_v42":
            return OmniCollectiveV42Backend(record, extracted_dir, self.generated_dir)
        if record.kind == "omni_collective_v46":
            return OmniCollectiveV46Backend(record, extracted_dir, self.generated_dir)
        if record.kind == "omni_collective_v47":
            return OmniCollectiveV47Backend(record, extracted_dir, self.generated_dir)
        if record.kind == "omni_collective_v48":
            return OmniCollectiveV48Backend(record, extracted_dir, self.generated_dir)
        if record.kind == "omni_collective_v41":
            return OmniCollectiveV41Backend(record, extracted_dir, self.generated_dir)
        if record.kind == "qwen_adapter":
            return QwenBackend(record, extracted_dir, self.generated_dir)
        raise RuntimeError(f"Unsupported model kind: {record.kind}")

    def _fallback_model_keys(self, failed_record: ModelRecord) -> List[str]:
        ordered: List[str] = []
        if failed_record.supports_image and not failed_record.supports_chat:
            ordered.extend(("v36_native", "v37_native_lite", "v38_native_xlite", "v38_native_xlite_fp16"))
        elif failed_record.supports_chat and failed_record.supports_vision:
            ordered.extend((
                "omni_collective_v47",
                "omni_collective_v46",
                "omni_collective_v42",
                "omni_collective_v41",
                "v40_benchmax",
                "omni_collective_v8",
                "omni_collective_v7",
                "omni_collective_v6",
                "omni_collective_v5",
                "omni_collective_v4",
                "omni_collective_v3",
                "omni_collective_v2",
                "omni_collective_v1",
                "v33_final",
                "v35_final",
                "v34_final",
                "qwen_v28",
                "v31_final",
                "v30_lite",
            ))
        else:
            ordered.extend((
                "omni_collective_v47",
                "omni_collective_v46",
                "omni_collective_v42",
                "omni_collective_v41",
                "v40_benchmax",
                "v33_final",
                "qwen_v28",
                "v35_final",
                "v34_final",
                "v31_final",
                "v30_lite",
            ))
        seen = {failed_record.key}
        keys: List[str] = []
        for key in ordered:
            if key in seen or key not in self.record_map:
                continue
            seen.add(key)
            keys.append(key)
        for record in self.records:
            if record.key in seen:
                continue
            if failed_record.supports_chat and not record.supports_chat:
                continue
            if failed_record.supports_vision and not record.supports_vision:
                continue
            if failed_record.supports_image and not record.supports_image and not record.supports_chat:
                continue
            seen.add(record.key)
            keys.append(record.key)
        return keys

    def _ensure_backend_direct_locked(self, record: ModelRecord) -> Tuple[ModelRecord, BaseBackend]:
        if self._backend_cache_size <= 1:
            if self._backend is not None and self._backend_key == record.key:
                return record, self._backend
            if self._backend is not None:
                self._backend.unload()
            self._backend = self._build_backend(record)
            self._backend_key = record.key
            self._backend_init_failures.pop(record.key, None)
            return record, self._backend

        cached = self._backend_cache.get(record.key)
        if cached is not None:
            if record.key in self._backend_lru:
                self._backend_lru.remove(record.key)
            self._backend_lru.append(record.key)
            self._backend = cached
            self._backend_key = record.key
            return record, cached

        backend = self._build_backend(record)
        self._backend_cache[record.key] = backend
        if record.key in self._backend_lru:
            self._backend_lru.remove(record.key)
        self._backend_lru.append(record.key)
        while len(self._backend_lru) > self._backend_cache_size:
            evict_key = self._backend_lru.pop(0)
            evicted = self._backend_cache.pop(evict_key, None)
            if evicted is not None:
                evicted.unload()
            if self._backend_key == evict_key:
                self._backend = None
                self._backend_key = ""

        self._backend = backend
        self._backend_key = record.key
        self._backend_init_failures.pop(record.key, None)
        return record, backend

    def ensure_backend(self, model_key: str) -> Tuple[ModelRecord, BaseBackend]:
        with self._lock:
            requested = self.record_map[model_key]
            candidate_keys = [requested.key]
            if requested.key in self._backend_init_failures:
                candidate_keys = []
            candidate_keys.extend(self._fallback_model_keys(requested))
            attempts: List[str] = []

            for key in candidate_keys:
                record = self.record_map.get(key)
                if record is None:
                    continue
                try:
                    return self._ensure_backend_direct_locked(record)
                except Exception as exc:
                    cooked = _trim_text(str(exc), limit=220)
                    self._backend_init_failures[record.key] = cooked
                    attempts.append(f"{record.key}: {cooked}")
                    logging.exception("Failed to initialize backend for %s", record.key)

            summary = "; ".join(attempts[:3]) or f"{requested.key}: no fallback candidates were available"
            raise RuntimeError(f"Failed to initialize a usable backend for {requested.label}. {summary}")

    def _refresh_records_locked(self) -> None:
        refreshed = discover_model_records(
            models_dir=self.models_dir,
            common_summary_path=self.common_summary_path,
        )
        self.records = list(refreshed)
        self.record_map = {record.key: record for record in self.records}
        self._backend_init_failures = {
            key: value for key, value in self._backend_init_failures.items() if key in self.record_map
        }
        valid_keys = set(self.record_map)
        for key in list(self._backend_cache):
            if key not in valid_keys:
                backend = self._backend_cache.pop(key, None)
                if backend is not None:
                    backend.unload()
                if key in self._backend_lru:
                    self._backend_lru.remove(key)
        if self.selected_model_key != "auto" and self.selected_model_key not in self.record_map:
            self.selected_model_key = "auto"
        if self._backend_key and self._backend_key not in self.record_map:
            if self._backend is not None:
                self._backend.unload()
            self._backend = None
            self._backend_key = ""

    def _fetch_model_store_manifest_locked(self, force_refresh: bool = False) -> Dict[str, Any]:
        now = time.time()
        if (
            not force_refresh
            and self._model_store_manifest_cache is not None
            and (now - self._model_store_manifest_ts) < MODEL_STORE_CACHE_TTL_SECONDS
        ):
            return dict(self._model_store_manifest_cache)
        url = _hf_dataset_file_url(self.model_store_repo_id, "manifest.json")
        with urlopen(url, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        self._model_store_manifest_cache = payload
        self._model_store_manifest_ts = now
        return dict(payload)

    def model_store_catalog(self, force_refresh: bool = False) -> Dict[str, Any]:
        with self._lock:
            manifest = self._fetch_model_store_manifest_locked(force_refresh=force_refresh)
            installed_names = {record.zip_path.name for record in self.records}
            rows: List[Dict[str, Any]] = []
            for item in manifest.get("models") or []:
                if not isinstance(item, dict):
                    continue
                file_name = str(item.get("file_name") or "").strip()
                if not file_name:
                    continue
                details = describe_model_artifact_name(file_name)
                local_path = self.models_dir / file_name
                installed = local_path.exists() or file_name in installed_names
                rows.append(
                    {
                        "file_name": file_name,
                        "size_bytes": int(item.get("size_bytes") or 0),
                        "size_mb": float(item.get("size_mb") or 0.0),
                        "family": str(item.get("family") or details.get("family") or "other"),
                        "known": bool(details.get("known")),
                        "model_key": str(details.get("key") or ""),
                        "label": str(details.get("label") or Path(file_name).stem),
                        "kind": str(details.get("kind") or ""),
                        "capabilities": list(details.get("capabilities") or []),
                        "note": str(details.get("note") or ""),
                        "benchmark_hint": str(details.get("benchmark_hint") or ""),
                        "download_url": _hf_dataset_file_url(self.model_store_repo_id, file_name),
                        "installed": installed,
                        "local_path": str(local_path.resolve()) if local_path.exists() else "",
                        "selectable": bool(details.get("known")) and str(details.get("key") or "") in self.record_map,
                    }
                )
            rows.sort(key=lambda item: (not item["installed"], item["label"].lower(), item["file_name"].lower()))
            return {
                "repo_id": self.model_store_repo_id,
                "model_count": len(rows),
                "models": rows,
            }

    def model_store_jobs(self) -> Dict[str, Any]:
        with self._lock:
            jobs = sorted(
                (dict(job) for job in self._model_store_jobs.values()),
                key=lambda item: str(item.get("started_at") or ""),
                reverse=True,
            )
            return {"jobs": jobs}

    def _set_model_store_job(self, job_id: str, **updates: Any) -> None:
        with self._lock:
            payload = dict(self._model_store_jobs.get(job_id) or {})
            payload.update(updates)
            self._model_store_jobs[job_id] = payload

    def _install_model_store_worker(self, job_id: str, file_name: str, expected_size: int) -> None:
        target = self.models_dir / file_name
        temp_target = self.models_dir / f"{file_name}.{job_id}.part"
        try:
            self._set_model_store_job(job_id, status="downloading")
            url = _hf_dataset_file_url(self.model_store_repo_id, file_name)
            downloaded = 0
            with urlopen(url, timeout=60) as response:
                header_size = int(response.headers.get("Content-Length") or 0)
                total_bytes = expected_size or header_size
                self._set_model_store_job(job_id, total_bytes=total_bytes)
                with temp_target.open("wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)
                        self._set_model_store_job(job_id, downloaded_bytes=downloaded)
            if target.exists():
                target.unlink()
            temp_target.replace(target)
            with self._lock:
                self._refresh_records_locked()
            self._set_model_store_job(
                job_id,
                status="completed",
                downloaded_bytes=target.stat().st_size,
                total_bytes=target.stat().st_size,
                local_path=str(target.resolve()),
                finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                selectable=bool(describe_model_artifact_name(file_name).get("key") in self.record_map),
            )
        except Exception as exc:
            try:
                temp_target.unlink(missing_ok=True)
            except Exception:
                pass
            self._set_model_store_job(
                job_id,
                status="error",
                error=str(exc),
                finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            )

    def install_model_store_artifact(self, file_name: str) -> Dict[str, Any]:
        cooked = str(file_name or "").strip()
        if not cooked:
            raise ValueError("file_name is required")
        with self._lock:
            manifest = self._fetch_model_store_manifest_locked(force_refresh=False)
            manifest_rows = {
                str(item.get("file_name") or "").strip(): item
                for item in (manifest.get("models") or [])
                if isinstance(item, dict)
            }
            if cooked not in manifest_rows:
                raise FileNotFoundError(f"{cooked} is not present in {self.model_store_repo_id}")
            for job in self._model_store_jobs.values():
                if job.get("file_name") == cooked and job.get("status") in {"queued", "downloading"}:
                    return dict(job)
            target = self.models_dir / cooked
            expected_size = int(manifest_rows[cooked].get("size_bytes") or 0)
            if target.exists() and (expected_size <= 0 or target.stat().st_size == expected_size):
                self._refresh_records_locked()
                payload = {
                    "job_id": f"already-{int(time.time())}",
                    "file_name": cooked,
                    "status": "completed",
                    "downloaded_bytes": target.stat().st_size,
                    "total_bytes": target.stat().st_size,
                    "local_path": str(target.resolve()),
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                self._model_store_jobs[payload["job_id"]] = payload
                return payload
            job_id = f"store-{int(time.time() * 1000)}"
            payload = {
                "job_id": job_id,
                "file_name": cooked,
                "status": "queued",
                "downloaded_bytes": 0,
                "total_bytes": expected_size,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "local_path": "",
                "error": "",
            }
            self._model_store_jobs[job_id] = payload
            worker = threading.Thread(
                target=self._install_model_store_worker,
                args=(job_id, cooked, expected_size),
                daemon=True,
                name=f"model-store-{job_id}",
            )
            worker.start()
            return dict(payload)

    def _session_scope(self, session_id: str, record_key: str, purpose: str) -> str:
        return f"{session_id}::{purpose}::{record_key}"

    def _default_text_record(self) -> ModelRecord:
        for key in ("omni_collective_v46", "omni_collective_v42", "omni_collective_v41", "v40_benchmax", "omni_collective_v47", "omni_collective_v8", "omni_collective_v7", "omni_collective_v6", "omni_collective_v5", "omni_collective_v4", "omni_collective_v3", "v33_final", "omni_collective_v2", "v35_final", "v34_final", "qwen_v28", "v31_final", "v30_lite"):
            if key in self.record_map and self.record_map[key].supports_chat:
                return self.record_map[key]
        for record in self.records:
            if record.supports_chat:
                return record
        raise RuntimeError("No text-capable local models were discovered.")

    def _collective_consultants(
        self,
        settings: Optional[Dict[str, Any]] = None,
        chosen_record: Optional[ModelRecord] = None,
    ) -> List[ModelRecord]:
        consultants = [record for record in self.records if record.supports_chat]
        if not consultants:
            return []

        settings = dict(settings or {})
        raw_keys = settings.get("collective_consultant_keys")
        preferred_keys: List[str] = []
        if isinstance(raw_keys, str):
            preferred_keys = [part.strip() for part in raw_keys.split(",") if part.strip()]
        elif isinstance(raw_keys, (list, tuple)):
            preferred_keys = [str(part).strip() for part in raw_keys if str(part).strip()]

        ranked: List[ModelRecord] = []
        seen: set[str] = set()

        def add_record(record: Optional[ModelRecord]) -> None:
            if record is None or not record.supports_chat or record.key in seen:
                return
            ranked.append(record)
            seen.add(record.key)

        if bool(settings.get("collective_include_chosen", True)):
            add_record(chosen_record)
        for key in preferred_keys:
            add_record(self.record_map.get(key))
        for record in consultants:
            add_record(record)

        limit = int(settings.get("collective_consultant_limit") or 0)
        if limit > 0:
            ranked = ranked[:limit]
        return ranked

    def _prepare_memory_bundle(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        if settings.get("memory_enabled", True) is False:
            return {"memory_notes": [], "context_block": "", "example_count": 0, "turn_count": 0}
        return self.memory_store.build_context(session_id, prompt)

    def _seed_auto_tool_events(self, prompt: str, settings: Dict[str, Any]) -> List[ToolEvent]:
        events: List[ToolEvent] = []
        if bool(settings.get("web_search_enabled", False)) and should_offer_web_search(prompt):
            try:
                events.append(self.web_search.search(prompt, max_results=int(settings.get("web_search_results") or 5)))
            except Exception:
                pass
        if bool(settings.get("cmd_open_enabled", True)) and should_offer_open_cmd(prompt):
            try:
                events.append(self.cmd_open.open(""))
            except Exception:
                pass
        return events

    def _run_web_query_cached(
        self,
        query: str,
        tool_cache: Dict[str, ToolEvent],
        settings: Dict[str, Any],
    ) -> Optional[ToolEvent]:
        key = _trim_text(query, limit=220).lower()
        if not key:
            return None
        if key in tool_cache:
            return tool_cache[key]
        if len(tool_cache) >= int(settings.get("web_search_budget") or 3):
            return None
        try:
            event = self.web_search.search(query, max_results=int(settings.get("web_search_results") or 5))
        except Exception:
            return None
        tool_cache[key] = event
        return event

    def _run_cmd_open_cached(
        self,
        working_dir: str,
        tool_cache: Dict[str, ToolEvent],
    ) -> Optional[ToolEvent]:
        cooked_dir = _trim_text(working_dir or "", limit=220)
        key = f"open_cmd::{cooked_dir.lower()}"
        if key in tool_cache:
            return tool_cache[key]
        try:
            event = self.cmd_open.open(cooked_dir)
        except Exception:
            return None
        tool_cache[key] = event
        return event

    def _run_text_model(
        self,
        record: ModelRecord,
        *,
        session_id: str,
        prompt: str,
        settings: Dict[str, Any],
        route_reason: str,
        tool_cache: Dict[str, ToolEvent],
        allow_tool_calls: bool,
    ) -> Tuple[ChatResult, List[ToolEvent]]:
        resolved_record, backend = self.ensure_backend(record.key)
        local_events: List[ToolEvent] = []
        run_settings = dict(settings)
        effective_route_reason = route_reason
        if resolved_record.key != record.key:
            effective_route_reason = (
                f"{route_reason} Requested {record.label} could not be initialized, so the system fell back to "
                f"{resolved_record.label}."
            )
        run_settings["route_reason"] = effective_route_reason
        if tool_cache:
            run_settings["tool_context"] = format_tool_results(list(tool_cache.values()))
        if allow_tool_calls and (bool(settings.get("web_search_enabled", False)) or bool(settings.get("cmd_open_enabled", True))):
            run_settings["tool_instruction"] = (
                "Available tools:\n"
                "TOOL:web_search: <query>\n"
                "TOOL:open_cmd: <optional working directory>\n"
                "Use a tool line only when it is explicitly needed, otherwise answer normally."
            )
        result = backend.chat(session_id, prompt, run_settings)
        raw_response = str(result.response or "")
        requests = parse_tool_requests(raw_response) if allow_tool_calls else []
        result.response = strip_tool_calls(raw_response)
        if requests:
            for request in requests:
                if request["name"] == "web_search" and bool(settings.get("web_search_enabled", False)):
                    tool_event = self._run_web_query_cached(request["argument"], tool_cache, settings)
                elif request["name"] == "open_cmd" and bool(settings.get("cmd_open_enabled", True)):
                    tool_event = self._run_cmd_open_cached(request["argument"], tool_cache)
                else:
                    tool_event = None
                if tool_event is not None:
                    local_events.append(tool_event)
            if local_events:
                follow_settings = dict(settings)
                follow_settings["route_reason"] = route_reason
                follow_settings["memory_context"] = str(settings.get("memory_context") or "")
                follow_settings["consultation_context"] = str(settings.get("consultation_context") or "")
                follow_settings["tool_context"] = format_tool_results(list(tool_cache.values()))
                follow_settings["tool_instruction"] = "Tool results are already available below. Use them and answer directly."
                follow = backend.chat(session_id, prompt, follow_settings)
                follow.response = strip_tool_calls(follow.response)
                result = follow
        return result, local_events

    def _build_consult_prompt(self, prompt: str, action_mode: str) -> str:
        if action_mode == "image":
            return (
                "You are one consultant in a multimodel image-planning panel.\n"
                "Return short art-direction notes only: subject, composition, style, colors, and any constraints.\n\n"
                f"Request:\n{prompt}"
            )
        return (
            "You are one consultant in a multimodel answer panel.\n"
            "Give a concise answer draft with the key reasoning or caveat in under 130 words.\n\n"
            f"Request:\n{prompt}"
        )

    def _format_consultations(self, consultation_rows: Sequence[Dict[str, str]]) -> str:
        lines: List[str] = []
        for row in consultation_rows:
            label = _trim_text(row.get("model_label") or row.get("model_key") or "model", limit=80)
            response = _trim_text(row.get("response") or "", limit=280)
            if not response:
                continue
            lines.append(f"- {label}: {response}")
        return "\n".join(lines)

    def _build_synthesis_prompt(self, prompt: str, action_mode: str) -> str:
        if action_mode == "image":
            return (
                "Synthesize the cross-model notes into one final image prompt.\n"
                "Output a single polished prompt only, without bullets or explanation.\n\n"
                f"Original request:\n{prompt}"
            )
        return (
            "Synthesize the memory, tool results, and cross-model consultation into one final answer.\n"
            "Be direct, coherent, and avoid repeating the panel format.\n\n"
            f"Original request:\n{prompt}"
        )

    def _normalized_agent_mode(self, raw_mode: Any) -> str:
        cooked = str(raw_mode or "off").strip().lower()
        aliases = {
            "collective_all": "collective",
            "panel": "collective",
            "loop_agent": "loop",
            "collective_loop_agent": "collective_loop",
        }
        normalized = aliases.get(cooked, cooked)
        if normalized in {"off", "collective", "loop", "collective_loop"}:
            return normalized
        return "off"

    def _format_loop_history(self, steps: Sequence[Dict[str, str]]) -> str:
        if not steps:
            return "No prior loop work yet."
        lines: List[str] = []
        for row in steps:
            lines.append(f"Step {row.get('step')}:")
            if row.get("goal"):
                lines.append(f"Goal: {row['goal']}")
            if row.get("worker_excerpt"):
                lines.append(f"Worker: {row['worker_excerpt']}")
            if row.get("review_note"):
                lines.append(f"Reviewer: {row['review_note']}")
            if row.get("next_step"):
                lines.append(f"Next: {row['next_step']}")
        return "\n".join(lines[-20:])

    def _build_loop_planner_prompt(self, prompt: str, history: str, step_index: int, max_steps: int) -> str:
        return (
            "You are the planner sub-agent inside a loop agent.\n"
            "Break the user's task into the single highest-value next action.\n"
            "Respond using exactly these headings:\n"
            "DONE: yes or no\n"
            "STEP_GOAL: one concise sentence\n"
            "SUCCESS_SIGNAL: how to know the task is complete\n"
            "WORKING_NOTES: short reasoning\n\n"
            f"Original task:\n{prompt}\n\n"
            f"Loop step: {step_index} of {max_steps}\n\n"
            f"Work log:\n{history}"
        )

    def _build_loop_worker_prompt(
        self,
        prompt: str,
        *,
        history: str,
        step_goal: str,
        step_index: int,
        max_steps: int,
    ) -> str:
        return (
            "You are the worker sub-agent inside a loop agent.\n"
            "Execute the next best action for the task. Prefer doing the work over talking about the work.\n"
            "Respond using exactly these headings:\n"
            "DONE: yes or no\n"
            "OUTPUT: the actual deliverable or best current progress\n"
            "NEXT_FOCUS: what should happen next if not done\n\n"
            f"Original task:\n{prompt}\n\n"
            f"Loop step: {step_index} of {max_steps}\n"
            f"Current goal:\n{step_goal}\n\n"
            f"Work log:\n{history}"
        )

    def _build_loop_review_prompt(
        self,
        prompt: str,
        *,
        history: str,
        latest_output: str,
        success_signal: str,
        step_index: int,
        max_steps: int,
    ) -> str:
        return (
            "You are the reviewer sub-agent inside a loop agent.\n"
            "Judge whether the user's task is complete enough to stop.\n"
            "Respond using exactly these headings:\n"
            "COMPLETE: yes or no\n"
            "FINAL_RESPONSE: the reply that should be shown to the user right now\n"
            "REASON: why you marked it complete or incomplete\n"
            "NEXT_STEP: the next action if it is not complete\n\n"
            f"Original task:\n{prompt}\n\n"
            f"Loop step: {step_index} of {max_steps}\n"
            f"Success signal:\n{success_signal or 'Task is complete when the user request has been fully addressed.'}\n\n"
            f"Work log:\n{history}\n\n"
            f"Latest worker output:\n{latest_output}"
        )

    def _run_loop_agent_text(
        self,
        *,
        session_id: str,
        prompt: str,
        chosen_record: ModelRecord,
        settings: Dict[str, Any],
        route_reason: str,
        action_mode: str,
        memory_bundle: Dict[str, Any],
        collective_mode: bool,
    ) -> ChatResult:
        controller_record = chosen_record if chosen_record.supports_chat else self._default_text_record()
        max_steps = int(settings.get("loop_max_steps") or settings.get("loop_budget") or LOOP_AGENT_DEFAULT_MAX_STEPS)
        max_steps = max(2, min(LOOP_AGENT_HARD_MAX_STEPS, max_steps))
        tool_cache = {
            _trim_text(event.query, limit=220).lower(): event
            for event in self._seed_auto_tool_events(prompt, settings)
        }
        tool_event_rows: List[Dict[str, Any]] = [event.to_dict() for event in list(tool_cache.values())]
        consult_rows: List[Dict[str, str]] = []
        consulted_labels: List[str] = []
        skipped_models: List[Dict[str, str]] = []
        loop_steps: List[Dict[str, str]] = []
        final_response = ""
        completion_reason = ""
        completed = False

        for step_index in range(1, max_steps + 1):
            history = self._format_loop_history(loop_steps)
            planner_result, planner_events = self._run_text_model(
                controller_record,
                session_id=self._session_scope(session_id, controller_record.key, f"loop-plan-{step_index}"),
                prompt=self._build_loop_planner_prompt(prompt, history, step_index, max_steps),
                settings={
                    **settings,
                    "memory_context": memory_bundle.get("context_block") or "",
                },
                route_reason=f"{route_reason} Loop agent planner step {step_index} via {controller_record.label}.",
                tool_cache=tool_cache,
                allow_tool_calls=True,
            )
            tool_event_rows.extend(event.to_dict() for event in planner_events)
            step_goal = (
                _extract_labeled_section(planner_result.response, "STEP_GOAL")
                or _extract_labeled_section(planner_result.response, "WORKING_NOTES")
                or _trim_text(planner_result.response, limit=220)
                or "Advance the task toward a complete solution."
            )
            success_signal = _extract_labeled_section(planner_result.response, "SUCCESS_SIGNAL")

            if collective_mode:
                worker_result = self._run_agent_text(
                    session_id=self._session_scope(session_id, controller_record.key, f"loop-work-{step_index}"),
                    prompt=self._build_loop_worker_prompt(
                        prompt,
                        history=history,
                        step_goal=step_goal,
                        step_index=step_index,
                        max_steps=max_steps,
                    ),
                    chosen_record=chosen_record,
                    settings=settings,
                    route_reason=f"{route_reason} Loop agent worker step {step_index} used the collective panel.",
                    action_mode=action_mode,
                    memory_bundle=memory_bundle,
                )
                consult_rows.extend(list((worker_result.agent_trace or {}).get("consultation_rows") or []))
                consulted_labels.extend(list((worker_result.agent_trace or {}).get("consulted_models") or []))
                skipped_models.extend(list((worker_result.agent_trace or {}).get("skipped_models") or []))
                tool_event_rows.extend(list((worker_result.agent_trace or {}).get("tool_events") or []))
            else:
                worker_result, worker_events = self._run_text_model(
                    controller_record,
                    session_id=self._session_scope(session_id, controller_record.key, f"loop-work-{step_index}"),
                    prompt=self._build_loop_worker_prompt(
                        prompt,
                        history=history,
                        step_goal=step_goal,
                        step_index=step_index,
                        max_steps=max_steps,
                    ),
                    settings={
                        **settings,
                        "memory_context": memory_bundle.get("context_block") or "",
                    },
                    route_reason=f"{route_reason} Loop agent worker step {step_index} via {controller_record.label}.",
                    tool_cache=tool_cache,
                    allow_tool_calls=True,
                )
                tool_event_rows.extend(event.to_dict() for event in worker_events)

            worker_output = (
                _extract_labeled_section(worker_result.response, "OUTPUT")
                or _extract_labeled_section(worker_result.response, "FINAL_RESPONSE")
                or _trim_text(worker_result.response, limit=1800)
            )
            reviewer_result, reviewer_events = self._run_text_model(
                controller_record,
                session_id=self._session_scope(session_id, controller_record.key, f"loop-review-{step_index}"),
                prompt=self._build_loop_review_prompt(
                    prompt,
                    history=history,
                    latest_output=worker_output,
                    success_signal=success_signal,
                    step_index=step_index,
                    max_steps=max_steps,
                ),
                settings={
                    **settings,
                    "memory_context": memory_bundle.get("context_block") or "",
                },
                route_reason=f"{route_reason} Loop agent reviewer step {step_index} via {controller_record.label}.",
                tool_cache=tool_cache,
                allow_tool_calls=True,
            )
            tool_event_rows.extend(event.to_dict() for event in reviewer_events)

            review_complete = _parse_yes_no_section(reviewer_result.response, "COMPLETE")
            if review_complete is None:
                review_complete = _parse_yes_no_section(reviewer_result.response, "DONE")
            review_reason = (
                _extract_labeled_section(reviewer_result.response, "REASON")
                or _extract_labeled_section(reviewer_result.response, "MISSING")
            )
            next_step = (
                _extract_labeled_section(reviewer_result.response, "NEXT_STEP")
                or _extract_labeled_section(worker_result.response, "NEXT_FOCUS")
                or step_goal
            )
            final_candidate = (
                _extract_labeled_section(reviewer_result.response, "FINAL_RESPONSE")
                or worker_output
                or _trim_text(reviewer_result.response, limit=1800)
            )
            loop_steps.append(
                {
                    "step": str(step_index),
                    "goal": step_goal,
                    "worker_excerpt": _trim_text(worker_output, limit=320),
                    "review_note": _trim_text(review_reason or "", limit=220),
                    "next_step": _trim_text(next_step, limit=220),
                }
            )
            final_response = final_candidate
            completion_reason = review_reason or success_signal or f"Loop agent finished step {step_index}."

            if review_complete is True:
                completed = True
                break

        if not completed and final_response:
            final_response = (
                f"{final_response}\n\nLoop agent note: the loop budget ended before it confidently marked the task complete."
            )
            completion_reason = completion_reason or "Loop budget reached before completion."

        deduped_consulted = list(dict.fromkeys(label for label in consulted_labels if label))
        result = ChatResult(
            kind="text",
            model_key=controller_record.key,
            model_label=controller_record.label,
            route_reason=(
                f"{route_reason} Loop agent ran {len(loop_steps)} autonomous step(s)"
                + (" with collective worker consultations." if collective_mode else f" on {controller_record.label}.")
            ),
            response=final_response or "Loop agent did not produce a final answer.",
            timing={"loop_steps": len(loop_steps), "loop_max_steps": max_steps},
            prompt_used=prompt,
        )
        result.agent_trace = {
            "agent_mode": "collective_loop_agent" if collective_mode else "loop_agent",
            "memory_notes": list(memory_bundle.get("memory_notes") or []),
            "consulted_models": deduped_consulted,
            "consultation_rows": consult_rows,
            "skipped_models": skipped_models,
            "tool_events": tool_event_rows,
            "loop_steps": loop_steps,
            "loop_completed": completed,
            "loop_completion_reason": completion_reason,
            "loop_budget": max_steps,
            "loop_controller_model": controller_record.label,
            "loop_worker_mode": "collective" if collective_mode else "single_model",
        }
        return result

    def _run_agent_text(
        self,
        *,
        session_id: str,
        prompt: str,
        chosen_record: ModelRecord,
        settings: Dict[str, Any],
        route_reason: str,
        action_mode: str,
        memory_bundle: Dict[str, Any],
    ) -> ChatResult:
        tool_events = self._seed_auto_tool_events(prompt, settings)
        tool_cache = {_trim_text(event.query, limit=220).lower(): event for event in tool_events}
        consult_rows: List[Dict[str, str]] = []
        skipped_rows: List[Dict[str, str]] = []
        for consultant in self._collective_consultants(settings=settings, chosen_record=chosen_record):
            consult_settings = dict(settings)
            consult_settings["memory_context"] = memory_bundle.get("context_block") or ""
            try:
                consult_result, new_tools = self._run_text_model(
                    consultant,
                    session_id=self._session_scope(session_id, consultant.key, "consult"),
                    prompt=self._build_consult_prompt(prompt, action_mode=action_mode),
                    settings=consult_settings,
                    route_reason=f"{route_reason} Agent consultation via {consultant.label}.",
                    tool_cache=tool_cache,
                    allow_tool_calls=True,
                )
            except Exception as exc:
                skipped_rows.append(
                    {
                        "model_key": consultant.key,
                        "model_label": consultant.label,
                        "error": _trim_text(str(exc), limit=220),
                    }
                )
                continue
            tool_events.extend(new_tools)
            consult_rows.append(
                {
                    "model_key": consultant.key,
                    "model_label": consultant.label,
                    "response": consult_result.response,
                }
            )

        consultation_context = self._format_consultations(consult_rows)
        synthesis_record = chosen_record if chosen_record.supports_chat else self._default_text_record()
        synthesis_settings = dict(settings)
        synthesis_settings["memory_context"] = memory_bundle.get("context_block") or ""
        synthesis_settings["consultation_context"] = consultation_context
        if tool_cache:
            synthesis_settings["tool_context"] = format_tool_results(list(tool_cache.values()))
        synthesis_result, new_tools = self._run_text_model(
            synthesis_record,
            session_id=self._session_scope(session_id, synthesis_record.key, "answer"),
            prompt=self._build_synthesis_prompt(prompt, action_mode=action_mode),
            settings=synthesis_settings,
            route_reason=f"{route_reason} Agent mode consulted {len(consult_rows)} text models before synthesis.",
            tool_cache=tool_cache,
            allow_tool_calls=True,
        )
        tool_events.extend(new_tools)

        synthesis_result.agent_trace = {
            "agent_mode": "collective_panel",
            "memory_notes": list(memory_bundle.get("memory_notes") or []),
            "consulted_models": [row["model_label"] for row in consult_rows],
            "consultation_rows": consult_rows,
            "skipped_models": skipped_rows,
            "tool_events": [event.to_dict() for event in list(tool_cache.values())],
        }
        return synthesis_result

    def _run_agent_image(
        self,
        *,
        session_id: str,
        prompt: str,
        chosen_record: ModelRecord,
        settings: Dict[str, Any],
        route_reason: str,
        memory_bundle: Dict[str, Any],
    ) -> ChatResult:
        if not chosen_record.supports_image:
            raise RuntimeError(f"{chosen_record.label} does not support image generation.")
        tool_events = self._seed_auto_tool_events(prompt, settings)
        tool_cache = {_trim_text(event.query, limit=220).lower(): event for event in tool_events}
        consult_rows: List[Dict[str, str]] = []
        skipped_rows: List[Dict[str, str]] = []
        for consultant in self._collective_consultants(settings=settings, chosen_record=chosen_record):
            consult_settings = dict(settings)
            consult_settings["memory_context"] = memory_bundle.get("context_block") or ""
            try:
                consult_result, new_tools = self._run_text_model(
                    consultant,
                    session_id=self._session_scope(session_id, consultant.key, "consult-image"),
                    prompt=self._build_consult_prompt(prompt, action_mode="image"),
                    settings=consult_settings,
                    route_reason=f"{route_reason} Agent image consultation via {consultant.label}.",
                    tool_cache=tool_cache,
                    allow_tool_calls=True,
                )
            except Exception as exc:
                skipped_rows.append(
                    {
                        "model_key": consultant.key,
                        "model_label": consultant.label,
                        "error": _trim_text(str(exc), limit=220),
                    }
                )
                continue
            tool_events.extend(new_tools)
            consult_rows.append(
                {
                    "model_key": consultant.key,
                    "model_label": consultant.label,
                    "response": consult_result.response,
                }
            )

        planner = self._default_text_record()
        planner_settings = dict(settings)
        planner_settings["memory_context"] = memory_bundle.get("context_block") or ""
        planner_settings["consultation_context"] = self._format_consultations(consult_rows)
        if tool_cache:
            planner_settings["tool_context"] = format_tool_results(list(tool_cache.values()))
        planner_result, new_tools = self._run_text_model(
            planner,
            session_id=self._session_scope(session_id, planner.key, "image-synth"),
            prompt=self._build_synthesis_prompt(prompt, action_mode="image"),
            settings=planner_settings,
            route_reason=f"{route_reason} Agent mode refined the image prompt with {len(consult_rows)} text consultants.",
            tool_cache=tool_cache,
            allow_tool_calls=True,
        )
        tool_events.extend(new_tools)

        _record, backend = self.ensure_backend(chosen_record.key)
        final_settings = dict(settings)
        final_settings["memory_context"] = memory_bundle.get("context_block") or ""
        final_settings["consultation_context"] = self._format_consultations(consult_rows)
        effective_route_reason = (
            f"{route_reason} Agent mode consulted {len(consult_rows)} text models and refined the final image prompt."
        )
        if _record.key != chosen_record.key:
            effective_route_reason = (
                f"{effective_route_reason} Requested {chosen_record.label} could not be initialized, so the system fell "
                f"back to {_record.label}."
            )
        final_settings["route_reason"] = effective_route_reason
        image_result = backend.generate_image(session_id, planner_result.response or prompt, final_settings)
        image_result.agent_trace = {
            "agent_mode": "collective_panel",
            "memory_notes": list(memory_bundle.get("memory_notes") or []),
            "consulted_models": [row["model_label"] for row in consult_rows],
            "consultation_rows": consult_rows,
            "skipped_models": skipped_rows,
            "tool_events": [event.to_dict() for event in list(tool_cache.values())],
            "planner_model": planner.label,
        }
        image_result.refined_prompt = planner_result.response or prompt
        return image_result

    def status(self) -> Dict[str, Any]:
        with self._lock:
            active = self.record_map.get(self._backend_key) if self._backend_key else None
            return {
                "selected_model_key": self.selected_model_key,
                "active_model_key": active.key if active else "",
                "active_model_label": active.label if active else "",
                "last_route_reason": self.last_route_reason,
                "models_available": len(self.records),
                "device": self.device_info.get("resolved", str(self.device)),
                "generated_dir": str(self.generated_dir),
                "uploads_dir": str(self.uploads_dir),
                "exports_dir": str(self.exports_dir),
                "extraction_root": str(self.extraction_root),
                "memory_status": self.memory_store.global_status(),
                "active_backend_status": self._backend.status() if self._backend is not None else None,
            }

    def session_memory_snapshot(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            return self.memory_store.session_snapshot(session_id)

    def three_d_model_view(self) -> Dict[str, Any]:
        record = self.record_map.get("three_d_generation_micro_v1")
        if record is None:
            raise FileNotFoundError("three_d_generation_micro_v1 is not available in the local catalog.")
        extracted_dir = _extract_zip_once(record.zip_path, self.extraction_root)
        summary_path = _find_matching_file(
            extracted_dir,
            ("three_d_generation_micro_v1_summary.json",),
            "_summary.json",
        )
        meta_path = _find_matching_file(
            extracted_dir,
            ("three_d_generation_micro_v1_meta.json",),
            ".json",
        )
        summary_payload: Dict[str, Any] = {}
        meta_payload: Dict[str, Any] = {}
        if summary_path is not None:
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if meta_path is not None:
            meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
        row_summary = summary_payload.get("row_summary") if isinstance(summary_payload.get("row_summary"), dict) else {}
        sample_predictions = summary_payload.get("sample_predictions") if isinstance(summary_payload.get("sample_predictions"), list) else []
        return {
            "key": record.key,
            "label": record.label,
            "zip_path": str(record.zip_path.resolve()),
            "zip_name": record.zip_path.name,
            "zip_size_bytes": record.zip_path.stat().st_size,
            "summary_path": str(summary_path.resolve()) if summary_path is not None else "",
            "summary_name": summary_path.name if summary_path is not None else "",
            "meta_path": str(meta_path.resolve()) if meta_path is not None else "",
            "meta_name": meta_path.name if meta_path is not None else "",
            "parameter_count": summary_payload.get("parameter_count"),
            "train_accuracy": summary_payload.get("train_accuracy"),
            "val_accuracy": summary_payload.get("val_accuracy"),
            "concept_count": row_summary.get("concepts"),
            "source_rows": row_summary.get("source_rows"),
            "train_rows": row_summary.get("train_rows"),
            "val_rows": row_summary.get("val_rows"),
            "concept_labels": list(meta_payload.get("labels") or []),
            "sample_predictions": sample_predictions[:6],
        }

    def select_model(self, model_key: str, eager: bool = False) -> Dict[str, Any]:
        if model_key != "auto" and model_key not in self.record_map:
            raise KeyError(f"Unknown model key: {model_key}")
        self.selected_model_key = model_key
        if eager and model_key != "auto":
            record, _backend = self.ensure_backend(model_key)
            self.last_route_reason = f"Loaded {record.label}."
        return self.status()

    def clear(self, session_id: str) -> None:
        with self._lock:
            self.memory_store.clear_session(session_id)
            if self._backend is not None:
                self._backend.clear(session_id)
                self._backend.clear(self._session_scope(session_id, self._backend_key, "consult"))
                self._backend.clear(self._session_scope(session_id, self._backend_key, "answer"))
            session_upload_dir = self.uploads_dir / _safe_slug(session_id)
            if session_upload_dir.exists():
                for child in sorted(session_upload_dir.rglob("*"), reverse=True):
                    if child.is_file():
                        child.unlink(missing_ok=True)
                    else:
                        try:
                            child.rmdir()
                        except OSError:
                            pass
                try:
                    session_upload_dir.rmdir()
                except OSError:
                    pass

    def store_uploaded_image(self, *, session_id: str, filename: str, raw_bytes: bytes) -> Dict[str, Any]:
        with self._lock:
            image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            session_dir = self.uploads_dir / _safe_slug(session_id)
            session_dir.mkdir(parents=True, exist_ok=True)
            safe_name = _safe_upload_name(filename)
            target = session_dir / f"{Path(safe_name).stem}.png"
            image.save(target, format="PNG")
            return {
                "ok": True,
                "saved_path": str(target),
                "image_url": f"/uploads/{_safe_slug(session_id)}/{target.name}",
                "filename": target.name,
            }

    def save_generated_image(self, *, source_path: str, destination_hint: str = "") -> Dict[str, Any]:
        with self._lock:
            target = copy_generated_image(source_path, destination_hint, self.exports_dir / "saved_images")
            return {
                "ok": True,
                "saved_path": str(target),
            }

    def export_chat_image(
        self,
        *,
        session_id: str,
        transcript: Sequence[Dict[str, object]],
        destination_hint: str = "",
    ) -> Dict[str, Any]:
        with self._lock:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            default_name = f"supermix_chat_{stamp}.png"
            target = render_chat_transcript_image(
                transcript,
                destination_hint=destination_hint or str(self.exports_dir / "chat_images" / default_name),
                default_dir=self.exports_dir / "chat_images",
                session_id=session_id,
            )
            return {
                "ok": True,
                "saved_path": str(target),
            }

    def handle_prompt(
        self,
        *,
        session_id: str,
        prompt: str,
        model_key: str,
        action_mode: str,
        settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        with self._lock:
            requested_key = model_key or self.selected_model_key or "auto"
            if requested_key == "auto":
                chosen_record, route_reason = choose_auto_model(
                    self.records,
                    prompt,
                    action_mode=action_mode,
                    uploaded_image_path=str((settings or {}).get("uploaded_image_path") or ""),
                )
                if chosen_record is None:
                    raise RuntimeError("No local models were discovered.")
            else:
                chosen_record = self.record_map.get(requested_key)
                if chosen_record is None:
                    raise KeyError(f"Unknown model key: {requested_key}")
                route_reason = f"Manual selection kept {chosen_record.label}."

            resolved_action = action_mode
            if resolved_action == "auto":
                if chosen_record.supports_image and not chosen_record.supports_chat:
                    resolved_action = "image"
                else:
                    resolved_action = "text"

            settings = dict(settings or {})
            settings.setdefault("memory_enabled", True)
            settings.setdefault("agent_mode", "off")
            settings.setdefault("web_search_enabled", False)
            settings.setdefault("cmd_open_enabled", True)
            settings.setdefault("web_search_budget", 3)
            settings.setdefault("web_search_results", 5)
            settings.setdefault("loop_max_steps", LOOP_AGENT_DEFAULT_MAX_STEPS)
            memory_bundle = self._prepare_memory_bundle(session_id, prompt, settings)
            agent_mode = self._normalized_agent_mode(settings.get("agent_mode"))
            collective_enabled = agent_mode in {"collective", "collective_loop"}
            loop_enabled = agent_mode in {"loop", "collective_loop"}
            if loop_enabled and resolved_action == "image":
                loop_enabled = False
                route_reason = (
                    f"{route_reason} Loop agent currently supports text and vision replies; "
                    "image generation stayed on the normal one-pass path."
                )

            if loop_enabled:
                result = self._run_loop_agent_text(
                    session_id=session_id,
                    prompt=prompt,
                    chosen_record=chosen_record,
                    settings=settings,
                    route_reason=route_reason,
                    action_mode=resolved_action,
                    memory_bundle=memory_bundle,
                    collective_mode=collective_enabled,
                )
            elif collective_enabled:
                if resolved_action == "image":
                    result = self._run_agent_image(
                        session_id=session_id,
                        prompt=prompt,
                        chosen_record=chosen_record,
                        settings=settings,
                        route_reason=route_reason,
                        memory_bundle=memory_bundle,
                    )
                else:
                    result = self._run_agent_text(
                        session_id=session_id,
                        prompt=prompt,
                        chosen_record=chosen_record,
                        settings=settings,
                        route_reason=route_reason,
                        action_mode=resolved_action,
                        memory_bundle=memory_bundle,
                    )
            else:
                base_settings = dict(settings)
                base_settings["memory_context"] = memory_bundle.get("context_block") or ""
                base_settings["route_reason"] = route_reason
                tool_cache = {
                    _trim_text(event.query, limit=220).lower(): event
                    for event in self._seed_auto_tool_events(prompt, settings)
                }
                if tool_cache:
                    base_settings["tool_context"] = format_tool_results(list(tool_cache.values()))
                record, backend = self.ensure_backend(chosen_record.key)
                effective_route_reason = route_reason
                if record.key != chosen_record.key:
                    effective_route_reason = (
                        f"{route_reason} Requested {chosen_record.label} could not be initialized, so the system fell "
                        f"back to {record.label}."
                    )
                base_settings["route_reason"] = effective_route_reason
                if resolved_action == "image":
                    if not record.supports_image:
                        raise RuntimeError(f"{record.label} does not support image generation.")
                    result = backend.generate_image(session_id, prompt, base_settings)
                    result.agent_trace = {
                        "agent_mode": "off",
                        "memory_notes": list(memory_bundle.get("memory_notes") or []),
                        "tool_events": [event.to_dict() for event in list(tool_cache.values())],
                    }
                else:
                    if not record.supports_chat and record.supports_image:
                        result = backend.generate_image(session_id, prompt, base_settings)
                        result.agent_trace = {
                            "agent_mode": "off",
                            "memory_notes": list(memory_bundle.get("memory_notes") or []),
                            "tool_events": [event.to_dict() for event in list(tool_cache.values())],
                        }
                    else:
                        result, _new_tools = self._run_text_model(
                            record,
                            session_id=self._session_scope(session_id, record.key, "model"),
                            prompt=prompt,
                            settings=base_settings,
                            route_reason=effective_route_reason,
                            tool_cache=tool_cache,
                            allow_tool_calls=bool(settings.get("web_search_enabled", False)),
                        )
                        result.agent_trace = {
                            "agent_mode": "off",
                            "memory_notes": list(memory_bundle.get("memory_notes") or []),
                            "tool_events": [event.to_dict() for event in list(tool_cache.values())],
                        }

            assistant_summary = result.response or result.prompt_used or result.refined_prompt or ""
            tools_for_memory = list((result.agent_trace or {}).get("tool_events") or [])
            consultants_for_memory = list((result.agent_trace or {}).get("consultation_rows") or [])
            self.memory_store.update(
                session_id=session_id,
                user_text=prompt,
                assistant_text=assistant_summary,
                model_key=result.model_key,
                route_reason=result.route_reason,
                tools=tools_for_memory,
                consultants=consultants_for_memory,
            )

            self.selected_model_key = model_key or self.selected_model_key
            self.last_route_reason = result.route_reason
            payload = result.to_dict()
            payload["selected_model_key"] = self.selected_model_key
            payload["active_model_key"] = result.model_key
            payload["active_model_label"] = result.model_label
            payload["active_model_kind"] = self.record_map[result.model_key].kind
            return payload
