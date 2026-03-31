from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_MODELS_DIR = Path(r"C:\Users\kai99\Desktop\models")


def _resolve_default_common_summary() -> Path:
    output_dir = Path(__file__).resolve().parents[1] / "output"
    candidates = sorted(output_dir.glob("benchmark_all_models_common_plus_summary_*.json"))
    if candidates:
        return candidates[-1]
    return output_dir / "benchmark_all_models_common_plus_summary_20260327.json"


DEFAULT_COMMON_SUMMARY = _resolve_default_common_summary()

IMAGE_PROMPT_RE = re.compile(
    r"\b("
    r"image|picture|photo|draw|drawing|paint|poster|logo|icon|illustration|"
    r"render|rendering|sketch|wallpaper|scene|cover art|concept art|thumbnail"
    r")\b",
    re.IGNORECASE,
)
VISION_PROMPT_RE = re.compile(
    r"\b(upload|uploaded|analyze image|analyse image|recognize|recognise|identify|what is in this image|"
    r"what does this image show|look at the image|describe the image|visual clue|photo analysis)\b",
    re.IGNORECASE,
)
FAST_PROMPT_RE = re.compile(r"\b(fast|quick|brief|short|lite|tiny|minimal)\b", re.IGNORECASE)
CODE_PROMPT_RE = re.compile(
    r"\b(code|python|javascript|typescript|bug|debug|traceback|stack trace|sql|api|regex|function)\b",
    re.IGNORECASE,
)
ANALYTIC_PROMPT_RE = re.compile(
    r"\b(reason|solve|analysis|analyze|math|prove|logic|compare|tradeoff|algorithm|benchmark)\b",
    re.IGNORECASE,
)
MATH_PROMPT_RE = re.compile(
    r"(\bsolve\b|\bsimplify\b|\bfactor\b|\bexpand\b|\bdifferentiate\b|\bderivative\b|"
    r"\bintegrate\b|\bintegral\b|\bcalculate\b|\bevaluate\b|\bequation\b|\bpolynomial\b|"
    r"\balgebra\b|=|[\d\)\]][\+\-\*/\^])",
    re.IGNORECASE,
)
PROTEIN_PROMPT_RE = re.compile(
    r"\b(protein|folding|fold|amino acid|residue|alpha helix|beta sheet|hydrophobic|"
    r"chaperone|disulfide|contact map|plddt|msa|intrinsically disordered|"
    r"membrane protein|tm-score|rmsd)\b",
    re.IGNORECASE,
)
CREATIVE_PROMPT_RE = re.compile(
    r"\b(story|creative|poem|novel|character|rewrite|style|brainstorm|lyrics|dialogue)\b",
    re.IGNORECASE,
)
EXPERIMENTAL_PROMPT_RE = re.compile(r"\b(experimental|frontier|newest|latest|v39)\b", re.IGNORECASE)
GAN_IMAGE_PROMPT_RE = re.compile(r"\b(dcgan|gan|mnist|digit grid|digit sheet|cifar|retro sample|unconditional image)\b", re.IGNORECASE)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    family: str
    kind: str
    filename_tokens: Sequence[str]
    common_row_key: Optional[str]
    capabilities: Tuple[str, ...]
    note: str = ""
    recipe_eval_accuracy: Optional[float] = None
    benchmark_hint: str = ""
    preferred_weights: Sequence[str] = ()
    preferred_meta: Sequence[str] = ()
    adapter_markers: Sequence[str] = ()


@dataclass
class ModelRecord:
    key: str
    label: str
    family: str
    kind: str
    capabilities: Tuple[str, ...]
    zip_path: Path
    common_row_key: Optional[str]
    common_overall_exact: Optional[float]
    per_benchmark: Dict[str, float] = field(default_factory=dict)
    recipe_eval_accuracy: Optional[float] = None
    score_source: str = "common"
    note: str = ""
    benchmark_hint: str = ""
    preferred_weights: Tuple[str, ...] = ()
    preferred_meta: Tuple[str, ...] = ()
    adapter_markers: Tuple[str, ...] = ()

    @property
    def display_score(self) -> Optional[float]:
        if self.common_overall_exact is not None:
            return self.common_overall_exact
        return self.recipe_eval_accuracy

    @property
    def supports_chat(self) -> bool:
        return "chat" in self.capabilities

    @property
    def supports_image(self) -> bool:
        return "image" in self.capabilities

    @property
    def supports_vision(self) -> bool:
        return "vision" in self.capabilities

    def to_dict(self) -> Dict[str, object]:
        return {
            "key": self.key,
            "label": self.label,
            "family": self.family,
            "kind": self.kind,
            "capabilities": list(self.capabilities),
            "zip_path": str(self.zip_path),
            "zip_name": self.zip_path.name,
            "zip_size_bytes": self.zip_path.stat().st_size,
            "common_row_key": self.common_row_key,
            "common_overall_exact": self.common_overall_exact,
            "recipe_eval_accuracy": self.recipe_eval_accuracy,
            "score_source": self.score_source,
            "note": self.note,
            "benchmark_hint": self.benchmark_hint,
            "per_benchmark": dict(self.per_benchmark),
        }


MODEL_SPECS: Tuple[ModelSpec, ...] = (
    ModelSpec(
        key="science_vision_micro_v1",
        label="Science Vision Micro",
        family="vision",
        kind="image_recognition",
        filename_tokens=("supermix_science_image_recognition_micro_v1_",),
        common_row_key=None,
        capabilities=("chat", "vision"),
        note="Small local image-recognition specialist trained on the bundled science diagram set.",
        benchmark_hint="Upload-image recognition specialist.",
        preferred_weights=("science_image_recognition_micro_v1.pth",),
        preferred_meta=("science_image_recognition_micro_v1_meta.json",),
    ),
    ModelSpec(
        key="dcgan_mnist_model",
        label="DCGAN MNIST",
        family="gan",
        kind="dcgan_image",
        filename_tokens=("dcgan_mnist_model",),
        common_row_key=None,
        capabilities=("image",),
        note="Unconditional DCGAN trained on 28x28 grayscale MNIST digits. Prompt text only seeds the latent sample grid.",
        benchmark_hint="MNIST digit-grid generator.",
        preferred_weights=("generator_final.pth",),
    ),
    ModelSpec(
        key="dcgan_v2_in_progress",
        label="DCGAN V2 CIFAR",
        family="gan",
        kind="dcgan_image",
        filename_tokens=("dcgan_v2_in_progress",),
        common_row_key=None,
        capabilities=("image",),
        note="Unconditional RGB DCGAN v2 trained on CIFAR-style images. Prompt text only seeds the latent sample grid.",
        benchmark_hint="RGB CIFAR-style generator.",
        preferred_weights=("generator_epoch_015.pth", "generator_epoch_010.pth", "generator_epoch_005.pth"),
    ),
    ModelSpec(
        key="omni_collective_v1",
        label="Omni Collective V1",
        family="fusion",
        kind="omni_collective",
        filename_tokens=("supermix_omni_collective_v1_",),
        common_row_key=None,
        capabilities=("chat", "vision"),
        note="Fused local checkpoint trained from the model catalog plus the math and science-image corpora.",
        benchmark_hint="Multimodal fused assistant.",
        preferred_weights=("omni_collective_v1.pth",),
        preferred_meta=("omni_collective_v1_meta.json",),
    ),
    ModelSpec(
        key="omni_collective_v2",
        label="Omni Collective V2 Frontier",
        family="fusion",
        kind="omni_collective",
        filename_tokens=("supermix_omni_collective_v2_frontier_",),
        common_row_key=None,
        capabilities=("chat", "vision"),
        note="Warm-started larger omni continuation trained on wider coding, knowledge, language, image-prompt, science-image, and 3D/video data.",
        benchmark_hint="Expanded multimodal fused assistant.",
        preferred_weights=("omni_collective_v2_frontier.pth",),
        preferred_meta=("omni_collective_v2_frontier_meta.json",),
    ),
    ModelSpec(
        key="omni_collective_v3",
        label="Omni Collective V3 Frontier",
        family="fusion",
        kind="omni_collective_v3",
        filename_tokens=("supermix_omni_collective_v3_frontier_",),
        common_row_key=None,
        capabilities=("chat", "vision"),
        recipe_eval_accuracy=0.45126262626262625,
        note="Larger routed-depth fused continuation with expanded language, science-image, 3D, video-contact, and Qwen-repair distillation data.",
        benchmark_hint="Newest fused multimodal frontier checkpoint.",
        preferred_weights=("omni_collective_v3_frontier.pth",),
        preferred_meta=("omni_collective_v3_frontier_meta.json",),
    ),
    ModelSpec(
        key="omni_collective_v4",
        label="Omni Collective V4 Frontier",
        family="fusion",
        kind="omni_collective_v4",
        filename_tokens=("supermix_omni_collective_v4_frontier_",),
        common_row_key="omni_collective_v4",
        capabilities=("chat", "vision"),
        recipe_eval_accuracy=0.5175903490230448,
        note="Larger sparse-routed multimodal frontier checkpoint with teacher-league repair rows, deeper memory, and wider text/image/video/3D data.",
        benchmark_hint="Latest fused multimodal frontier checkpoint.",
        preferred_weights=("omni_collective_v4_frontier.pth",),
        preferred_meta=("omni_collective_v4_frontier_meta.json",),
    ),
    ModelSpec(
        key="omni_collective_v5",
        label="Omni Collective V5 Frontier",
        family="fusion",
        kind="omni_collective_v5",
        filename_tokens=("supermix_omni_collective_v5_frontier_",),
        common_row_key="omni_collective_v5",
        capabilities=("chat", "vision"),
        recipe_eval_accuracy=0.5270367807000198,
        note="V4 continuation with extra coding, OpenSCAD, prompt-understanding rows, and a longer multi-pass deliberation path.",
        benchmark_hint="Latest fused multimodal coding-aware frontier checkpoint.",
        preferred_weights=("omni_collective_v5_frontier.pth",),
        preferred_meta=("omni_collective_v5_frontier_meta.json",),
    ),
    ModelSpec(
        key="omni_collective_v6",
        label="Omni Collective V6 Frontier",
        family="fusion",
        kind="omni_collective_v6",
        filename_tokens=("supermix_omni_collective_v6_frontier_",),
        common_row_key="omni_collective_v6",
        capabilities=("chat", "vision"),
        recipe_eval_accuracy=0.4939150169000483,
        note="Larger all-model distillation frontier with forced small-Qwen teachers, heavier conversation/math/protein grounding, and longer multi-pass deliberation.",
        benchmark_hint="Latest all-model distilled multimodal frontier checkpoint.",
        preferred_weights=("omni_collective_v6_frontier.pth",),
        preferred_meta=("omni_collective_v6_frontier_meta.json",),
    ),
    ModelSpec(
        key="v40_benchmax",
        label="V40 Benchmax",
        family="fusion",
        kind="omni_collective_v5",
        filename_tokens=("supermix_v40_benchmax_",),
        common_row_key="v40_benchmax",
        capabilities=("chat", "vision"),
        recipe_eval_accuracy=0.4363799283154122,
        note="Benchmark-maximization continuation using the v39-style benchmix recipe plus support rows and protein-folding knowledge rows.",
        benchmark_hint="Latest benchmark-focused multimodal frontier checkpoint.",
        preferred_weights=("omni_collective_v40_benchmax.pth",),
        preferred_meta=("omni_collective_v40_benchmax_meta.json",),
    ),
    ModelSpec(
        key="math_equation_micro_v1",
        label="Math Equation Micro",
        family="math",
        kind="math_equation",
        filename_tokens=("supermix_math_equation_micro_v1_",),
        common_row_key=None,
        capabilities=("chat",),
        note="Tiny trained math-intent model with exact symbolic solving for equations and calculus prompts.",
        benchmark_hint="Exact math/equation specialist.",
        preferred_weights=("math_equation_micro_v1.pth",),
        preferred_meta=("math_equation_micro_v1_meta.json",),
    ),
    ModelSpec(
        key="protein_folding_micro_v1",
        label="Protein Folding Micro",
        family="protein",
        kind="protein_folding",
        filename_tokens=("supermix_protein_folding_micro_v1_",),
        common_row_key=None,
        capabilities=("chat",),
        note="Mini protein-folding specialist trained on the benchmark protein pack plus extra structure-prediction prompt templates.",
        benchmark_hint="Protein folding and structure-prediction specialist.",
        preferred_weights=("protein_folding_micro_v1.pth",),
        preferred_meta=("protein_folding_micro_v1_meta.json",),
    ),
    ModelSpec(
        key="qwen_v28",
        label="Qwen V28",
        family="qwen",
        kind="qwen_adapter",
        filename_tokens=("qwen_supermix_enhanced_v28_cloud_plus_runpod_budget_final_adapter",),
        common_row_key="qwen_v28",
        capabilities=("chat",),
        note="LoRA adapter benchmarked with the local Qwen base model.",
        benchmark_hint="Grounded Qwen adapter.",
        adapter_markers=("adapter/adapter_config.json",),
    ),
    ModelSpec(
        key="qwen_v30",
        label="Qwen V30 Experimental",
        family="qwen",
        kind="qwen_adapter",
        filename_tokens=(
            "qwen_supermix_enhanced_v30_anchor_refresh_20260326_experimental_adapter",
            "qwen_supermix_enhanced_v30_anchor_refresh_20260326_experimental_bundle",
        ),
        common_row_key="qwen_v30",
        capabilities=("chat",),
        note="Experimental Qwen adapter.",
        benchmark_hint="Experimental Qwen branch.",
        adapter_markers=("adapter/adapter_config.json",),
    ),
    ModelSpec(
        key="v30_lite",
        label="V30 Lite FP16",
        family="champion",
        kind="champion_chat",
        filename_tokens=("champion_v30_lite_student_fp16_bundle_20260326",),
        common_row_key="v30_lite",
        capabilities=("chat",),
        note="Fast lightweight Champion student.",
        benchmark_hint="Fastest small text model.",
        preferred_weights=("champion_model_chat_v30_lite_student_fp16.pth",),
        preferred_meta=("chat_model_meta_v30_lite_student.json",),
    ),
    ModelSpec(
        key="v31_final",
        label="V31 Hybrid Plus Refresh",
        family="champion",
        kind="champion_chat",
        filename_tokens=(
            "champion_v31_hybrid_plus_refresh_final_model_20260326",
            "champion_v31_hybrid_plus_refresh_bundle_20260326",
        ),
        common_row_key="v31_final",
        capabilities=("chat",),
        benchmark_hint="Hybrid continuation model.",
        preferred_weights=("champion_model_chat_v31_hybrid_plus_refresh.pth",),
        preferred_meta=("chat_model_meta_v31_hybrid_plus_refresh.json",),
    ),
    ModelSpec(
        key="v31_image_variant",
        label="V31 Image Variant",
        family="wrapper",
        kind="image_wrapper",
        filename_tokens=("champion_v31_image_variant_bundle_20260326",),
        common_row_key="v31_final",
        capabilities=("chat", "image"),
        note="Wrapper artifact around the v31 text checkpoint.",
        benchmark_hint="Chat + external image backend.",
        preferred_weights=("model/champion_model_chat_v31_hybrid_plus_refresh.pth",),
        preferred_meta=("model/chat_model_meta_v31_hybrid_plus_refresh.json",),
    ),
    ModelSpec(
        key="v32_final",
        label="V32 Omnifuse",
        family="champion",
        kind="champion_chat",
        filename_tokens=(
            "champion_v32_omnifuse_final_model_20260326",
            "champion_v32_omnifuse_bundle_20260326",
        ),
        common_row_key="v32_final",
        capabilities=("chat",),
        benchmark_hint="Omnifuse student.",
        preferred_weights=("champion_model_chat_v32_omnifuse_final.pth",),
        preferred_meta=("chat_model_meta_v32_omnifuse_final.json",),
    ),
    ModelSpec(
        key="v33_final",
        label="V33 Frontier",
        family="champion",
        kind="champion_chat",
        filename_tokens=(
            "champion_v33_frontier_full_model_20260326",
            "champion_v33_frontier_full_bundle_20260326",
        ),
        common_row_key="v33_final",
        capabilities=("chat",),
        benchmark_hint="Best saved common-benchmark text model.",
        preferred_weights=("champion_model_chat_v33_frontier_full_final.pth",),
        preferred_meta=("chat_model_meta_v33_frontier_full_final.json",),
    ),
    ModelSpec(
        key="v34_final",
        label="V34 Frontier Plus",
        family="champion",
        kind="champion_chat",
        filename_tokens=("champion_v34_frontier_plus_full_model_20260326",),
        common_row_key="v34_stage2",
        capabilities=("chat",),
        note="Official v34 artifact chosen from stage2.",
        benchmark_hint="Expanded frontier checkpoint.",
        preferred_weights=("champion_model_chat_v34_frontier_plus_stage2.pth",),
        preferred_meta=("chat_model_meta_v34_frontier_plus_stage2.json",),
    ),
    ModelSpec(
        key="v35_final",
        label="V35 Collective",
        family="champion",
        kind="champion_chat",
        filename_tokens=("champion_v35_collective_allteachers_full_model_20260326",),
        common_row_key="v35_stage2",
        capabilities=("chat",),
        note="Mapped to the stronger v35 stage2 row.",
        benchmark_hint="All-teachers collective checkpoint.",
        preferred_weights=("champion_model_chat_v35_collective_allteachers_stage2.pth",),
        preferred_meta=("chat_model_meta_v35_collective_allteachers_stage2.json",),
    ),
    ModelSpec(
        key="v36_native",
        label="V36 Native Image",
        family="native_image",
        kind="native_image",
        filename_tokens=("champion_v36_native_image_single_checkpoint_model_20260327",),
        common_row_key="v36_native",
        capabilities=("image",),
        benchmark_hint="Largest native image model.",
        preferred_weights=("champion_model_chat_v36_native_image_single_checkpoint.pth",),
        preferred_meta=("chat_model_meta_v36_native_image_single_checkpoint.json",),
    ),
    ModelSpec(
        key="v37_native_lite",
        label="V37 Native Image Lite",
        family="native_image",
        kind="native_image",
        filename_tokens=("champion_v37_native_image_lite_single_checkpoint_model_20260327",),
        common_row_key="v37_native_lite",
        capabilities=("image",),
        benchmark_hint="Small native image model.",
        preferred_weights=("champion_model_chat_v37_native_image_lite_single_checkpoint.pth",),
        preferred_meta=("chat_model_meta_v37_native_image_lite_single_checkpoint.json",),
    ),
    ModelSpec(
        key="v38_native_xlite",
        label="V38 Native Image XLite",
        family="native_image",
        kind="native_image",
        filename_tokens=("champion_v38_native_image_xlite_single_checkpoint_model_20260327",),
        common_row_key="v38_native_xlite",
        capabilities=("image",),
        benchmark_hint="Extra-lite native image model.",
        preferred_weights=("champion_model_chat_v38_native_image_xlite_single_checkpoint.pth",),
        preferred_meta=("chat_model_meta_v38_native_image_xlite_single_checkpoint.json",),
    ),
    ModelSpec(
        key="v38_native_xlite_fp16",
        label="V38 Native Image XLite FP16",
        family="native_image",
        kind="native_image",
        filename_tokens=("champion_v38_native_image_xlite_single_checkpoint_model_fp16_20260327",),
        common_row_key="v38_native_xlite",
        capabilities=("image",),
        note="Half-precision packaging of the same v38 XLite line.",
        benchmark_hint="Smallest image-capable package.",
        preferred_weights=("champion_model_chat_v38_native_image_xlite_single_checkpoint_fp16.pth",),
        preferred_meta=("chat_model_meta_v38_native_image_xlite_single_checkpoint.json",),
    ),
    ModelSpec(
        key="v39_final",
        label="V39 Frontier Reasoning",
        family="champion",
        kind="champion_chat",
        filename_tokens=("champion_v39_frontier_reasoning_plus_full_model_20260327",),
        common_row_key=None,
        capabilities=("chat",),
        recipe_eval_accuracy=0.0549,
        note="No common-benchmark pass completed after the pod ran out of credit.",
        benchmark_hint="Newest experimental reasoning checkpoint.",
        preferred_weights=("champion_model_chat_v39_frontier_reasoning_plus_stage2.pth",),
        preferred_meta=("chat_model_meta_v39_frontier_reasoning_plus_stage2.json",),
    ),
)


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _load_common_rows(summary_path: Path) -> Dict[str, Dict[str, object]]:
    if not summary_path.exists():
        return {}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = payload.get("summary_rows")
    if not isinstance(rows, list):
        return {}
    out: Dict[str, Dict[str, object]] = {}
    for row in rows:
        if isinstance(row, dict) and row.get("model"):
            out[str(row["model"])] = row
    return out


def _match_token_index(spec: ModelSpec, path: Path) -> Optional[int]:
    name = path.name
    for idx, token in enumerate(spec.filename_tokens):
        if token in name:
            return idx
    return None


def _candidate_rank(spec: ModelSpec, path: Path) -> Optional[Tuple[int, int, int, float]]:
    token_index = _match_token_index(spec, path)
    if token_index is None:
        return None
    name = path.name.lower()
    is_duplicate_copy = 1 if " (1)" in path.name else 0
    is_bundle = 1 if "bundle" in name else 0
    return (token_index, is_bundle, is_duplicate_copy, -path.stat().st_mtime)


def _discover_artifacts(models_dir: Path) -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    ranks: Dict[str, Tuple[int, int, int, float]] = {}
    files = [p for p in models_dir.iterdir() if p.is_file() and p.suffix.lower() == ".zip"]
    for spec in MODEL_SPECS:
        for path in files:
            rank = _candidate_rank(spec, path)
            if rank is None:
                continue
            current_rank = ranks.get(spec.key)
            if current_rank is None or rank < current_rank:
                found[spec.key] = path
                ranks[spec.key] = rank
    return found


def discover_model_records(
    models_dir: Path = DEFAULT_MODELS_DIR,
    common_summary_path: Path = DEFAULT_COMMON_SUMMARY,
) -> List[ModelRecord]:
    common_rows = _load_common_rows(common_summary_path)
    artifacts = _discover_artifacts(models_dir)
    records: List[ModelRecord] = []
    for spec in MODEL_SPECS:
        path = artifacts.get(spec.key)
        if path is None:
            continue
        common_row = common_rows.get(spec.common_row_key) if spec.common_row_key else None
        common_score = _safe_float(common_row.get("overall_exact")) if common_row else None
        per_benchmark = {}
        if common_row and isinstance(common_row.get("benchmarks"), dict):
            per_benchmark = {
                str(name): float(score)
                for name, score in common_row["benchmarks"].items()
                if _safe_float(score) is not None
            }
        score_source = spec.kind
        if common_score is not None:
            score_source = "common_alias" if spec.common_row_key and spec.common_row_key != spec.key else "common"
        elif spec.recipe_eval_accuracy is not None:
            score_source = "recipe_eval_only"

        records.append(
            ModelRecord(
                key=spec.key,
                label=spec.label,
                family=spec.family,
                kind=spec.kind,
                capabilities=tuple(spec.capabilities),
                zip_path=path,
                common_row_key=spec.common_row_key,
                common_overall_exact=common_score,
                per_benchmark=per_benchmark,
                recipe_eval_accuracy=spec.recipe_eval_accuracy,
                score_source=score_source,
                note=spec.note,
                benchmark_hint=spec.benchmark_hint,
                preferred_weights=tuple(spec.preferred_weights),
                preferred_meta=tuple(spec.preferred_meta),
                adapter_markers=tuple(spec.adapter_markers),
            )
        )
    records.sort(key=lambda item: ((item.display_score or -1.0), item.label.lower()), reverse=True)
    return records


def choose_auto_model(
    records: Sequence[ModelRecord],
    prompt: str,
    action_mode: str = "auto",
    uploaded_image_path: str = "",
) -> Tuple[Optional[ModelRecord], str]:
    available = {record.key: record for record in records}
    text_models = [record for record in records if record.supports_chat]
    image_models = [record for record in records if record.supports_image]
    vision_models = [record for record in records if record.supports_vision]
    prompt_text = str(prompt or "").strip()
    lowered = prompt_text.lower()
    has_uploaded_image = bool(str(uploaded_image_path or "").strip())

    if not prompt_text:
        return (
            available.get("v40_benchmax")
            or available.get("omni_collective_v6")
            or available.get("omni_collective_v5")
            or available.get("omni_collective_v4")
            or available.get("omni_collective_v3")
            or available.get("v33_final")
            or (text_models[0] if text_models else None),
            "Empty prompt fell back to the default text model.",
        )

    wants_image = action_mode == "image" or (
        action_mode == "auto"
        and bool(IMAGE_PROMPT_RE.search(prompt_text) or GAN_IMAGE_PROMPT_RE.search(prompt_text))
    )
    wants_vision = (
        action_mode == "vision"
        or has_uploaded_image
        or (action_mode == "auto" and bool(VISION_PROMPT_RE.search(prompt_text)))
    )
    wants_fast = bool(FAST_PROMPT_RE.search(prompt_text)) or len(prompt_text) < 34
    wants_math = action_mode != "image" and bool(MATH_PROMPT_RE.search(prompt_text))
    wants_protein = action_mode != "image" and bool(PROTEIN_PROMPT_RE.search(prompt_text))
    wants_model_selection = any(token in lowered for token in ("which model", "best model", "select a model", "pick a model"))

    if wants_model_selection:
        for key in ("omni_collective_v6", "v40_benchmax", "omni_collective_v5", "omni_collective_v4", "omni_collective_v3", "omni_collective_v2", "omni_collective_v1", "v33_final", "qwen_v28"):
            if key in available:
                return available[key], "Auto picked the fused catalog model because the prompt asks about model choice."

    if wants_vision and vision_models:
        if has_uploaded_image and any(token in lowered for token in ("compare", "explain", "teach", "why", "analyze", "analyse")):
            for key in ("omni_collective_v6", "omni_collective_v5", "omni_collective_v4", "omni_collective_v3", "omni_collective_v2", "omni_collective_v1", "science_vision_micro_v1"):
                if key in available:
                    return available[key], "Auto picked a vision-capable chat model because an uploaded image needs analysis."
        for key in ("science_vision_micro_v1", "omni_collective_v6", "omni_collective_v5", "omni_collective_v4", "omni_collective_v3", "omni_collective_v2", "omni_collective_v1", "v40_benchmax"):
            if key in available:
                return available[key], "Auto picked the uploaded-image recognition model because the prompt looks visual."

    if wants_image and image_models:
        if GAN_IMAGE_PROMPT_RE.search(prompt_text):
            for key in ("dcgan_v2_in_progress", "dcgan_mnist_model", "v36_native", "v38_native_xlite", "v37_native_lite", "v31_image_variant"):
                if key in available:
                    return available[key], "Auto picked the DCGAN image model because the prompt explicitly mentions GAN, MNIST, CIFAR, or digit-grid generation."
        if wants_fast:
            for key in ("v38_native_xlite_fp16", "v38_native_xlite", "v37_native_lite", "v36_native"):
                if key in available:
                    return available[key], "Auto picked the smallest image-capable model for a quick image prompt."
        for key in ("v36_native", "v37_native_lite", "v38_native_xlite", "v31_image_variant"):
            if key in available:
                return available[key], "Auto picked an image-capable model because the prompt looks visual."

    if wants_math:
        for key in ("math_equation_micro_v1", "v33_final", "qwen_v28"):
            if key in available:
                return available[key], "Auto picked the math specialist because the prompt looks like an equation or symbolic math task."

    if wants_protein:
        for key in ("protein_folding_micro_v1", "v40_benchmax", "omni_collective_v6", "v33_final"):
            if key in available:
                return available[key], "Auto picked the protein-folding specialist because the prompt looks like protein structure or folding analysis."

    if wants_fast:
        for key in ("v30_lite", "qwen_v28", "v31_final"):
            if key in available:
                return available[key], "Auto picked the fastest local text model for a brief or quick-turn prompt."

    if EXPERIMENTAL_PROMPT_RE.search(prompt_text) and "v39_final" in available:
        return available["v39_final"], "Auto picked the newest experimental reasoning checkpoint."

    if CODE_PROMPT_RE.search(prompt_text) or ANALYTIC_PROMPT_RE.search(prompt_text):
        for key in ("v40_benchmax", "omni_collective_v6", "omni_collective_v5", "omni_collective_v4", "omni_collective_v3", "v33_final", "v35_final", "v34_final", "qwen_v28"):
            if key in available:
                return available[key], "Auto picked the strongest benchmarked reasoning/coding text model."

    if CREATIVE_PROMPT_RE.search(prompt_text):
        for key in ("qwen_v28", "omni_collective_v6", "omni_collective_v5", "v40_benchmax", "omni_collective_v4", "omni_collective_v3", "v33_final", "v31_final"):
            if key in available:
                return available[key], "Auto picked a more open-ended text model for a creative prompt."

    for key in ("v40_benchmax", "omni_collective_v6", "omni_collective_v5", "omni_collective_v4", "omni_collective_v3", "v33_final", "v35_final", "v34_final", "v31_final", "qwen_v28"):
        if key in available:
            return available[key], "Auto picked the default strongest local text model."
    return (text_models[0] if text_models else (image_models[0] if image_models else None), "Auto fell back to the first available local model.")


def models_to_json(records: Iterable[ModelRecord]) -> List[Dict[str, object]]:
    items = [record.to_dict() for record in records]
    items.insert(
        0,
        {
            "key": "auto",
            "label": "Auto",
            "family": "router",
            "kind": "auto",
            "capabilities": ["chat", "image", "vision"],
            "zip_path": "",
            "zip_name": "",
            "zip_size_bytes": 0,
            "common_row_key": "",
            "common_overall_exact": None,
            "recipe_eval_accuracy": None,
            "score_source": "router",
            "note": "Routes each prompt to the most appropriate local model.",
            "benchmark_hint": "Prompt-aware model routing.",
            "per_benchmark": {},
        },
    )
    return items
