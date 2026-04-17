from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn


THREE_D_REQUEST_MARKERS = (
    "current user request:",
    "original request:",
    "request:",
)
WORD_RE = re.compile(r"[a-z0-9][a-z0-9_\\-\\+]*", re.IGNORECASE)
THREE_D_PROMPT_RE = re.compile(
    r"(?:\b3d\b|\bopen(?:scad)?\b|\bmesh\b|\bobj\b|\bstl\b|\bcad\b|\bparametric\b|"
    r"\bphone stand\b|\bhollow box\b|\bcountersunk\b|\bmounting holes\b|\bprism\b|"
    r"\bpyramid\b|\btetrahedron\b|\bgrid plane\b|\bspokes\b|\bextrude\b|\bpolyhedron\b)",
    re.IGNORECASE,
)


THREE_D_GENERATION_SPECS: Tuple[Dict[str, Any], ...] = (
    {
        "concept": "hollow_box",
        "display_name": "hollow box",
        "keywords": ("hollow box", "box with walls", "shell box", "rectangular enclosure"),
        "answers": {
            "code": "module hollow_box(size=[40,30,20], wall=2){difference(){cube(size);translate([wall,wall,wall])cube([size[0]-2*wall,size[1]-2*wall,size[2]-2*wall]);}} hollow_box();",
            "explain_then_code": "Use `difference()` to subtract an inner cube from the outer shell so the wall thickness stays parametric.\n\nmodule hollow_box(size=[40,30,20], wall=2){difference(){cube(size);translate([wall,wall,wall])cube([size[0]-2*wall,size[1]-2*wall,size[2]-2*wall]);}} hollow_box();",
            "concise_answer": "module hollow_box(size=[40,30,20], wall=2){difference(){cube(size);translate([wall,wall,wall])cube([size[0]-2*wall,size[1]-2*wall,size[2]-2*wall]);}} hollow_box();",
        },
    },
    {
        "concept": "rounded_plate_holes",
        "display_name": "rounded plate with four mounting holes",
        "keywords": ("rounded plate", "mounting holes", "mount plate", "rounded rectangle"),
        "answers": {
            "code": "module rounded_plate(size=[70,40], r=6, h=4, hole_r=2, inset=8){difference(){linear_extrude(height=h)hull(){for(x=[-size[0]/2+r,size[0]/2-r], y=[-size[1]/2+r,size[1]/2-r])translate([x,y])circle(r=r,$fn=48);}for(x=[-size[0]/2+inset,size[0]/2-inset], y=[-size[1]/2+inset,size[1]/2-inset])translate([x,y,-1])cylinder(h=h+2,r=hole_r,$fn=36);}} rounded_plate();",
            "explain_then_code": "A simple way is to hull four corner circles into a rounded rectangle, then subtract four cylinders for the holes.\n\nmodule rounded_plate(size=[70,40], r=6, h=4, hole_r=2, inset=8){difference(){linear_extrude(height=h)hull(){for(x=[-size[0]/2+r,size[0]/2-r], y=[-size[1]/2+r,size[1]/2-r])translate([x,y])circle(r=r,$fn=48);}for(x=[-size[0]/2+inset,size[0]/2-inset], y=[-size[1]/2+inset,size[1]/2-inset])translate([x,y,-1])cylinder(h=h+2,r=hole_r,$fn=36);}} rounded_plate();",
            "concise_answer": "Hull four corner circles into a rounded plate, then subtract four cylinders at the mounting coordinates with `difference()`.",
        },
    },
    {
        "concept": "phone_stand",
        "display_name": "parametric phone stand",
        "keywords": ("phone stand", "stand", "desk stand", "device stand"),
        "answers": {
            "code": "angle=68; thickness=4; width=80; height=55; module phone_stand(){linear_extrude(height=thickness)polygon([[0,0],[width,0],[width*0.60,height],[18,height]]);} rotate([90-angle,0,0]) phone_stand();",
            "explain_then_code": "Keep the footprint and support angle as parameters so you can tune the geometry without rewriting the polygon.\n\nangle=68; thickness=4; width=80; height=55; module phone_stand(){linear_extrude(height=thickness)polygon([[0,0],[width,0],[width*0.60,height],[18,height]]);} rotate([90-angle,0,0]) phone_stand();",
            "concise_answer": "angle=68; thickness=4; width=80; height=55; module phone_stand(){linear_extrude(height=thickness)polygon([[0,0],[width,0],[width*0.60,height],[18,height]]);} rotate([90-angle,0,0]) phone_stand();",
        },
    },
    {
        "concept": "cylinder_center_hole",
        "display_name": "cylinder with a centered hole",
        "keywords": ("cylinder with hole", "centered hole", "centered cylinder", "through-hole", "tube", "ring spacer"),
        "answers": {
            "code": "difference(){cylinder(h=20, r=12, $fn=64); translate([0,0,-1]) cylinder(h=22, r=4, $fn=48);}",
            "explain_then_code": "Start with the outer cylinder and subtract a slightly taller inner cylinder so the hole cleanly passes through.\n\ndifference(){cylinder(h=20, r=12, $fn=64); translate([0,0,-1]) cylinder(h=22, r=4, $fn=48);}",
            "concise_answer": "difference(){cylinder(h=20, r=12, $fn=64); translate([0,0,-1]) cylinder(h=22, r=4, $fn=48);}",
        },
    },
    {
        "concept": "radial_spokes",
        "display_name": "radial spokes around a circle",
        "keywords": ("spokes", "radial", "repeat around a circle", "wheel spokes"),
        "answers": {
            "code": "radius=24; spoke_len=18; spoke_w=4; for(angle=[0:30:330]) rotate([0,0,angle]) translate([radius,0,0]) cube([spoke_len,spoke_w,4], center=true);",
            "explain_then_code": "Use a `for()` loop over angles, then rotate and translate the spoke geometry into place.\n\nradius=24; spoke_len=18; spoke_w=4; for(angle=[0:30:330]) rotate([0,0,angle]) translate([radius,0,0]) cube([spoke_len,spoke_w,4], center=true);",
            "concise_answer": "Loop over angles with `for(angle=[0:30:330]) rotate([0,0,angle]) translate([radius,0,0]) ...` to place each spoke evenly.",
        },
    },
    {
        "concept": "countersunk_hole",
        "display_name": "countersunk screw hole",
        "keywords": ("countersunk", "screw hole", "fastener hole", "screw head"),
        "answers": {
            "code": "module countersunk_hole(depth=8, shaft_r=1.7, head_r=3.6, head_h=2){cylinder(h=depth, r=shaft_r, $fn=36); cylinder(h=head_h, r1=head_r, r2=shaft_r, $fn=36);} countersunk_hole();",
            "explain_then_code": "Model the shaft as a straight cylinder and the countersink as a tapered cone frustum.\n\nmodule countersunk_hole(depth=8, shaft_r=1.7, head_r=3.6, head_h=2){cylinder(h=depth, r=shaft_r, $fn=36); cylinder(h=head_h, r1=head_r, r2=shaft_r, $fn=36);} countersunk_hole();",
            "concise_answer": "module countersunk_hole(depth=8, shaft_r=1.7, head_r=3.6, head_h=2){cylinder(h=depth, r=shaft_r, $fn=36); cylinder(h=head_h, r1=head_r, r2=shaft_r, $fn=36);} countersunk_hole();",
        },
    },
    {
        "concept": "mirrored_feature",
        "display_name": "mirrored feature on both sides of a part",
        "keywords": ("mirror", "mirrored feature", "symmetry", "both sides"),
        "answers": {
            "code": "module feature(){translate([18,0,0]) cylinder(h=8, r=3, $fn=32);} union(){feature(); mirror([1,0,0]) feature();}",
            "explain_then_code": "Wrap the feature in a module, place it once, and call it again through `mirror()` across the symmetry plane.\n\nmodule feature(){translate([18,0,0]) cylinder(h=8, r=3, $fn=32);} union(){feature(); mirror([1,0,0]) feature();}",
            "concise_answer": "Wrap the feature in a module and call it twice, once normally and once with `mirror([1,0,0])` or the axis that matches your symmetry plane.",
        },
    },
    {
        "concept": "decorative_star",
        "display_name": "gear-like decorative star",
        "keywords": ("star", "gear-like", "decorative star", "polygon star"),
        "answers": {
            "code": "points=[for(i=[0:11]) let(r=(i%2==0)?24:12,a=i*30) [r*cos(a), r*sin(a)]]; linear_extrude(height=4) polygon(points);",
            "explain_then_code": "Alternate outer and inner radii in a polar point list, then extrude the 2D polygon into a thin solid.\n\npoints=[for(i=[0:11]) let(r=(i%2==0)?24:12,a=i*30) [r*cos(a), r*sin(a)]]; linear_extrude(height=4) polygon(points);",
            "concise_answer": "points=[for(i=[0:11]) let(r=(i%2==0)?24:12,a=i*30) [r*cos(a), r*sin(a)]]; linear_extrude(height=4) polygon(points);",
        },
    },
    {
        "concept": "boolean_ops",
        "display_name": "OpenSCAD boolean operations",
        "keywords": ("union", "difference", "intersection", "boolean operations"),
        "answers": {
            "code": "`union()` combines solids, `difference()` subtracts later solids from the first, and `intersection()` keeps only the overlapping volume shared by the solids.",
            "explain_then_code": "`union()` combines shapes into one solid, `difference()` cuts later shapes out of the first one, and `intersection()` keeps only the overlap.",
            "concise_answer": "`union()` combines, `difference()` subtracts, and `intersection()` keeps only the shared overlap.",
        },
    },
    {
        "concept": "editable_models",
        "display_name": "editable parametric OpenSCAD models",
        "keywords": ("editable", "parametric", "best practice", "reusable modules"),
        "answers": {
            "code": "Use small reusable modules, expose the key dimensions as parameters, and separate primitive generation from transforms so edits stay local and predictable.",
            "explain_then_code": "The safest pattern is to isolate geometry in small modules, push dimensions into parameters, and keep transforms outside the primitive definitions so later edits do not cascade unpredictably.",
            "concise_answer": "Use small modules, parameters for the main dimensions, and separate transforms from primitive generation.",
        },
    },
    {
        "concept": "pyramid",
        "display_name": "square pyramid",
        "keywords": ("pyramid", "square pyramid", "polyhedron pyramid"),
        "answers": {
            "code": "polyhedron(points=[[-15,-15,0],[15,-15,0],[15,15,0],[-15,15,0],[0,0,24]], faces=[[0,1,2,3],[0,1,4],[1,2,4],[2,3,4],[3,0,4]]);",
            "explain_then_code": "A square pyramid is easiest as a `polyhedron()` with four base corners and one apex.\n\npolyhedron(points=[[-15,-15,0],[15,-15,0],[15,15,0],[-15,15,0],[0,0,24]], faces=[[0,1,2,3],[0,1,4],[1,2,4],[2,3,4],[3,0,4]]);",
            "concise_answer": "polyhedron(points=[[-15,-15,0],[15,-15,0],[15,15,0],[-15,15,0],[0,0,24]], faces=[[0,1,2,3],[0,1,4],[1,2,4],[2,3,4],[3,0,4]]);",
        },
    },
    {
        "concept": "triangular_prism",
        "display_name": "triangular prism",
        "keywords": ("triangular prism", "prism", "triangle prism"),
        "answers": {
            "code": "polyhedron(points=[[-16,-10,0],[16,-10,0],[0,12,0],[-16,-10,24],[16,-10,24],[0,12,24]], faces=[[0,1,2],[3,5,4],[0,3,4,1],[1,4,5,2],[2,5,3,0]]);",
            "explain_then_code": "Define one triangle at z=0, copy it upward, then connect the matching edges to form the prism walls.\n\npolyhedron(points=[[-16,-10,0],[16,-10,0],[0,12,0],[-16,-10,24],[16,-10,24],[0,12,24]], faces=[[0,1,2],[3,5,4],[0,3,4,1],[1,4,5,2],[2,5,3,0]]);",
            "concise_answer": "polyhedron(points=[[-16,-10,0],[16,-10,0],[0,12,0],[-16,-10,24],[16,-10,24],[0,12,24]], faces=[[0,1,2],[3,5,4],[0,3,4,1],[1,4,5,2],[2,5,3,0]]);",
        },
    },
    {
        "concept": "tetrahedron",
        "display_name": "tetrahedron",
        "keywords": ("tetrahedron", "four-sided polyhedron", "triangular pyramid"),
        "answers": {
            "code": "polyhedron(points=[[14,14,14],[-14,-14,14],[-14,14,-14],[14,-14,-14]], faces=[[0,1,2],[0,3,1],[0,2,3],[1,3,2]]);",
            "explain_then_code": "A tetrahedron only needs four non-coplanar points and four triangular faces.\n\npolyhedron(points=[[14,14,14],[-14,-14,14],[-14,14,-14],[14,-14,-14]], faces=[[0,1,2],[0,3,1],[0,2,3],[1,3,2]]);",
            "concise_answer": "polyhedron(points=[[14,14,14],[-14,-14,14],[-14,14,-14],[14,-14,-14]], faces=[[0,1,2],[0,3,1],[0,2,3],[1,3,2]]);",
        },
    },
    {
        "concept": "grid_plane",
        "display_name": "flat grid plane",
        "keywords": ("grid plane", "flat plane", "grid mesh", "floor grid"),
        "answers": {
            "code": "for(x=[-20:10:20]) translate([x,0,0]) cube([1,50,1], center=true); for(y=[-20:10:20]) translate([0,y,0]) cube([50,1,1], center=true);",
            "explain_then_code": "A quick printable grid plane is just two perpendicular line families laid out with loops.\n\nfor(x=[-20:10:20]) translate([x,0,0]) cube([1,50,1], center=true); for(y=[-20:10:20]) translate([0,y,0]) cube([50,1,1], center=true);",
            "concise_answer": "for(x=[-20:10:20]) translate([x,0,0]) cube([1,50,1], center=true); for(y=[-20:10:20]) translate([0,y,0]) cube([50,1,1], center=true);",
        },
    },
)


THREE_D_CONCEPT_LABELS: Tuple[str, ...] = tuple(str(item["concept"]) for item in THREE_D_GENERATION_SPECS)
THREE_D_CONCEPT_TO_INDEX = {label: index for index, label in enumerate(THREE_D_CONCEPT_LABELS)}
THREE_D_DISPLAY_NAMES: Dict[str, str] = {
    str(item["concept"]): str(item["display_name"]) for item in THREE_D_GENERATION_SPECS
}
THREE_D_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    str(item["concept"]): tuple(str(keyword) for keyword in item["keywords"]) for item in THREE_D_GENERATION_SPECS
}


def extract_request_text(prompt: str) -> str:
    cooked = str(prompt or "").strip()
    lowered = cooked.lower()
    for marker in THREE_D_REQUEST_MARKERS:
        idx = lowered.rfind(marker)
        if idx >= 0:
            return cooked[idx + len(marker):].strip()
    return cooked


def normalize_3d_text(text: str) -> str:
    cooked = str(text or "").strip()
    cooked = cooked.replace("Ã¢â‚¬â€œ", "-").replace("Ã¢â‚¬â€", "-").replace("Ã¢Ë†â€™", "-")
    cooked = re.sub(r"\s+", " ", cooked)
    return cooked


def build_vocab(texts: Sequence[str], *, min_frequency: int = 1) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for text in texts:
        for ch in normalize_3d_text(text).lower():
            counts[ch] = counts.get(ch, 0) + 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for ch, count in sorted(counts.items()):
        if count >= min_frequency and ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def encode_text(text: str, vocab: Dict[str, int], max_len: int) -> List[int]:
    cooked = normalize_3d_text(extract_request_text(text)).lower()
    ids = [vocab.get(ch, 1) for ch in cooked[:max_len]]
    if len(ids) < max_len:
        ids.extend([0] * (max_len - len(ids)))
    return ids


def tokenize_words(text: str, *, max_words: int) -> List[str]:
    cooked = normalize_3d_text(extract_request_text(text)).lower()
    return WORD_RE.findall(cooked)[:max_words]


def encode_words(text: str, *, word_buckets: int, max_words: int) -> List[int]:
    ids = [0] * max_words
    for index, token in enumerate(tokenize_words(text, max_words=max_words)):
        ids[index] = (sum(ord(ch) * (offset + 1) for offset, ch in enumerate(token)) % max(word_buckets - 1, 1)) + 1
    return ids


def vectorize_batch(
    texts: Sequence[str],
    vocab: Dict[str, int],
    *,
    max_len: int,
    word_buckets: int,
    max_words: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    char_rows = [encode_text(text, vocab, max_len) for text in texts]
    word_rows = [encode_words(text, word_buckets=word_buckets, max_words=max_words) for text in texts]
    return (
        torch.tensor(char_rows, dtype=torch.long),
        torch.tensor(word_rows, dtype=torch.long),
    )


def looks_like_3d_generation_prompt(prompt: str) -> bool:
    return bool(THREE_D_PROMPT_RE.search(str(prompt or "")))


def heuristic_3d_concept(prompt: str) -> Optional[str]:
    lowered = normalize_3d_text(extract_request_text(prompt)).lower()
    scores: Dict[str, int] = {}
    for concept, keywords in THREE_D_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword in lowered:
                score += max(1, len(keyword.split()))
        if score > 0:
            scores[concept] = score
    if not scores:
        return None
    return max(sorted(scores), key=lambda key: scores[key])


def choose_answer_variant(prompt: str) -> str:
    lowered = normalize_3d_text(prompt).lower()
    if any(token in lowered for token in ("best practice", "editable", "parametric", "maintainable")):
        return "explain_then_code"
    if any(token in lowered for token in ("what is", "difference between", "when should", "why use", "explain")):
        return "explain_then_code"
    if any(token in lowered for token in ("brief", "short", "one sentence", "concise")):
        return "concise_answer"
    if any(token in lowered for token in ("code only", "snippet", "generate", "write", "show", "example")):
        return "code"
    return "explain_then_code"


class ThreeDGenerationMiniNet(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        num_concepts: int,
        char_embed_dim: int = 28,
        conv_channels: int = 56,
        word_buckets: int = 512,
        word_embed_dim: int = 20,
        hidden_dim: int = 112,
    ) -> None:
        super().__init__()
        self.char_embedding = nn.Embedding(vocab_size, char_embed_dim, padding_idx=0)
        self.char_conv = nn.Conv1d(char_embed_dim, conv_channels, kernel_size=5, padding=2)
        self.word_embedding = nn.Embedding(word_buckets, word_embed_dim, padding_idx=0)
        self.norm = nn.LayerNorm(conv_channels * 2 + word_embed_dim)
        self.fc1 = nn.Linear(conv_channels * 2 + word_embed_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.16)
        self.fc2 = nn.Linear(hidden_dim, num_concepts)

    def forward(self, char_ids: torch.Tensor, word_ids: torch.Tensor) -> torch.Tensor:
        char_mask = char_ids.ne(0)
        char_embed = self.char_embedding(char_ids).transpose(1, 2)
        conv = torch.relu(self.char_conv(char_embed)).transpose(1, 2)
        pooled_avg = (conv * char_mask.unsqueeze(-1)).sum(dim=1) / char_mask.sum(dim=1, keepdim=True).clamp(min=1)
        masked_conv = conv.masked_fill(~char_mask.unsqueeze(-1), -1e4)
        pooled_max = masked_conv.max(dim=1).values
        pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max))

        word_mask = word_ids.ne(0)
        word_embed = self.word_embedding(word_ids)
        word_pool = (word_embed * word_mask.unsqueeze(-1)).sum(dim=1) / word_mask.sum(dim=1, keepdim=True).clamp(min=1)

        features = torch.cat([pooled_avg, pooled_max, word_pool], dim=1)
        hidden = torch.relu(self.fc1(self.dropout(self.norm(features))))
        return self.fc2(self.dropout(hidden))


@dataclass
class ThreeDPrediction:
    concept: str
    label: str
    confidence: float
    probabilities: Dict[str, float]
    top_predictions: List[Dict[str, float]]
    variant: str


class ThreeDGenerationEngine:
    def __init__(self, *, weights_path: Path, meta_path: Path, device: Optional[torch.device] = None) -> None:
        self.weights_path = Path(weights_path).resolve()
        self.meta_path = Path(meta_path).resolve()
        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.vocab = {str(key): int(value) for key, value in dict(self.meta.get("vocab") or {}).items()}
        self.labels = tuple(str(item) for item in (self.meta.get("labels") or THREE_D_CONCEPT_LABELS))
        self.answer_bank = {
            str(concept): {str(key): str(value) for key, value in dict(variants or {}).items()}
            for concept, variants in dict(self.meta.get("answer_bank") or {}).items()
        }
        self.display_names = {
            str(key): str(value)
            for key, value in dict(self.meta.get("display_names") or THREE_D_DISPLAY_NAMES).items()
        }
        self.max_len = int(self.meta.get("max_len") or 224)
        self.max_words = int(self.meta.get("max_words") or 28)
        self.word_buckets = int(self.meta.get("word_buckets") or 512)
        self.device = device or torch.device("cpu")
        self.model = ThreeDGenerationMiniNet(
            vocab_size=max(len(self.vocab), 2),
            num_concepts=len(self.labels),
            char_embed_dim=int(self.meta.get("char_embed_dim") or 28),
            conv_channels=int(self.meta.get("conv_channels") or 56),
            word_buckets=self.word_buckets,
            word_embed_dim=int(self.meta.get("word_embed_dim") or 20),
            hidden_dim=int(self.meta.get("hidden_dim") or 112),
        ).to(self.device)
        try:
            state = torch.load(self.weights_path, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(self.weights_path, map_location=self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def predict(self, prompt: str) -> ThreeDPrediction:
        char_tensor, word_tensor = vectorize_batch(
            [prompt],
            self.vocab,
            max_len=self.max_len,
            word_buckets=self.word_buckets,
            max_words=self.max_words,
        )
        char_tensor = char_tensor.to(self.device)
        word_tensor = word_tensor.to(self.device)
        with torch.inference_mode():
            logits = self.model(char_tensor, word_tensor)[0]
            probs = torch.softmax(logits, dim=0).detach().cpu().tolist()
        probabilities = {label: float(prob) for label, prob in zip(self.labels, probs)}
        predicted = max(probabilities, key=probabilities.get)
        confidence = float(probabilities[predicted])
        heuristic = heuristic_3d_concept(prompt)
        if heuristic in probabilities and confidence < 0.88:
            heuristic_score = float(probabilities.get(heuristic, 0.0))
            if heuristic_score >= 0.14 or confidence < 0.45:
                predicted = heuristic
                confidence = max(confidence, heuristic_score, 0.91 if confidence < 0.45 else confidence)
        ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        top_predictions = [
            {"concept": concept, "confidence": round(float(score), 4)}
            for concept, score in ranked[:3]
        ]
        return ThreeDPrediction(
            concept=predicted,
            label=self.display_names.get(predicted, predicted.replace("_", " ")),
            confidence=confidence,
            probabilities=probabilities,
            top_predictions=top_predictions,
            variant=choose_answer_variant(prompt),
        )

    def answer(self, prompt: str) -> str:
        prediction = self.predict(prompt)
        variants = self.answer_bank.get(prediction.concept, {})
        answer = (
            variants.get(prediction.variant)
            or variants.get("code")
            or variants.get("explain_then_code")
            or next(iter(variants.values()), f"This specialist matched {prediction.label}.")
        )
        if any(token in prompt.lower() for token in ("brief", "short", "one sentence", "concise")):
            return answer
        lines = [answer]
        if prediction.confidence < 0.52:
            lines.append(
                f"Best matched 3D concept: {prediction.label} with low confidence ({prediction.confidence:.2f}), so treat this as a grounded best guess."
            )
        elif any(token in prompt.lower() for token in ("why", "confidence", "match", "which concept")):
            lines.append(f"Matched 3D concept: {prediction.label} ({prediction.confidence:.2f} confidence).")
        return "\n".join(line for line in lines if line.strip())

    def status(self) -> Dict[str, Any]:
        return {
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "labels": list(self.labels),
            "max_len": self.max_len,
            "max_words": self.max_words,
            "device": str(self.device),
            "val_accuracy": self.meta.get("val_accuracy"),
            "train_accuracy": self.meta.get("train_accuracy"),
            "parameter_count": int(sum(parameter.numel() for parameter in self.model.parameters())),
        }


def format_3d_generation_response(response: str, prediction: ThreeDPrediction) -> str:
    trailer = f"\n\n[3D concept: {prediction.label.lower()} | confidence {prediction.confidence:.2f}]"
    return str(response or "").strip() + trailer
