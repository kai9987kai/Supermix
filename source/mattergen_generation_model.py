from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn


REQUEST_MARKERS = ("current user request:", "original request:", "request:")
WORD_RE = re.compile(r"[a-z0-9][a-z0-9_\-\+\.]*", re.IGNORECASE)
MATTERGEN_PROMPT_RE = re.compile(
    r"(?:\bmattergen\b|\bmaterials? design\b|\bcrystal(?:line)?\b|\bspace group\b|\blattice\b|"
    r"\bfractional coordinates\b|\bcif\b|\bperovskite\b|\bspinel\b|\bgarnet\b|\bthermoelectric\b|"
    r"\bband gap\b|\benergy above hull\b|\bionic conductivity\b|\bhard magnet\b|\bphotocatalyst\b|"
    r"\btransparent conductor\b|\b2d semiconductor\b|\bmof\b|\bmaterials discovery\b)",
    re.IGNORECASE,
)

PAPER_INSPIRATIONS: Tuple[Dict[str, str], ...] = (
    {
        "title": "MatterGen: a generative model for inorganic materials design",
        "url": "https://arxiv.org/abs/2312.03687",
    },
    {
        "title": "microsoft/mattergen",
        "url": "https://github.com/microsoft/mattergen",
    },
    {
        "title": "DiffCrysGen: A Score-Based Diffusion Model for Diverse Inorganic Crystalline Materials",
        "url": "https://arxiv.org/abs/2505.07442",
    },
)

MATTERGEN_CONCEPT_SPECS: Tuple[Dict[str, Any], ...] = (
    {
        "concept": "solid_electrolyte",
        "display_name": "garnet solid electrolyte",
        "keywords": ("solid electrolyte", "ionic conductivity", "garnet", "li conductor", "llzo"),
        "property_tags": ("ionic_conductivity", "stability", "electrochemical_window"),
        "prototype": {
            "name": "LLZO-like garnet seed",
            "formula": "Li7La3Zr2O12",
            "crystal_system": "cubic",
            "space_group": "Ia-3d",
            "lattice": {"a": 12.97, "b": 12.97, "c": 12.97, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
            "sites": (("La", "La1", 0.1250, 0.0000, 0.2500), ("Zr", "Zr1", 0.0000, 0.0000, 0.0000), ("Li", "Li1", 0.3750, 0.0000, 0.2500), ("O", "O1", 0.2800, 0.0950, 0.1950)),
            "screening_focus": "low energy above hull, wide electrochemical window, and percolating Li pathways",
            "synthesis_note": "Cubic LLZO often needs dopant-assisted stabilization.",
        },
    },
    {
        "concept": "battery_cathode",
        "display_name": "lithium-ion cathode",
        "keywords": ("cathode", "battery cathode", "olivine", "spinel cathode", "intercalation"),
        "property_tags": ("voltage", "stability", "diffusion"),
        "prototype": {
            "name": "LiFePO4 olivine seed",
            "formula": "LiFePO4",
            "crystal_system": "orthorhombic",
            "space_group": "Pnma",
            "lattice": {"a": 10.33, "b": 6.01, "c": 4.69, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
            "sites": (("Li", "Li1", 0.0000, 0.0000, 0.0000), ("Fe", "Fe1", 0.2800, 0.2500, 0.9750), ("P", "P1", 0.0950, 0.2500, 0.4200), ("O", "O1", 0.1650, 0.0450, 0.2850)),
            "screening_focus": "voltage window, antisite defect control, and Li diffusion topology",
            "synthesis_note": "Carbon coating and particle-size control often matter as much as bulk composition.",
        },
    },
    {
        "concept": "hard_magnet",
        "display_name": "rare-earth-lean hard magnet",
        "keywords": ("hard magnet", "magnetic density", "permanent magnet", "coercivity", "fe16n2"),
        "property_tags": ("magnetic_density", "anisotropy", "supply_risk"),
        "prototype": {
            "name": "Fe16N2-inspired seed",
            "formula": "Fe16N2",
            "crystal_system": "tetragonal",
            "space_group": "I4/mmm",
            "lattice": {"a": 5.72, "b": 5.72, "c": 6.29, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
            "sites": (("Fe", "Fe1", 0.0000, 0.0000, 0.0000), ("Fe", "Fe2", 0.5000, 0.5000, 0.2500), ("N", "N1", 0.0000, 0.0000, 0.5000)),
            "screening_focus": "high magnetic density with controllable anisotropy and low decomposition risk",
            "synthesis_note": "Pair magnetic targets with hull and phonon checks before trusting the candidate.",
        },
    },
    {
        "concept": "thermoelectric",
        "display_name": "thermoelectric chalcogenide",
        "keywords": ("thermoelectric", "seebeck", "zT", "snse", "low thermal conductivity"),
        "property_tags": ("seebeck", "thermal_conductivity", "carrier_tuning"),
        "prototype": {
            "name": "SnSe-like thermoelectric seed",
            "formula": "SnSe",
            "crystal_system": "orthorhombic",
            "space_group": "Pnma",
            "lattice": {"a": 11.50, "b": 4.16, "c": 4.44, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
            "sites": (("Sn", "Sn1", 0.1200, 0.2500, 0.4800), ("Se", "Se1", 0.8550, 0.2500, 0.0900)),
            "screening_focus": "high Seebeck coefficient, low lattice thermal conductivity, and dopability",
            "synthesis_note": "Anisotropy and off-stoichiometry sensitivity should be screened early.",
        },
    },
    {
        "concept": "wide_bandgap_oxide",
        "display_name": "wide-bandgap oxide semiconductor",
        "keywords": ("wide bandgap", "oxide semiconductor", "power electronics", "beta-ga2o3", "breakdown field"),
        "property_tags": ("band_gap", "breakdown_field", "mobility"),
        "prototype": {
            "name": "beta-Ga2O3-like seed",
            "formula": "Ga2O3",
            "crystal_system": "monoclinic",
            "space_group": "C2/m",
            "lattice": {"a": 12.23, "b": 3.04, "c": 5.80, "alpha": 90.0, "beta": 103.7, "gamma": 90.0},
            "sites": (("Ga", "Ga1", 0.0900, 0.0000, 0.7950), ("Ga", "Ga2", 0.2810, 0.0000, 0.1550), ("O", "O1", 0.1650, 0.0000, 0.5630)),
            "screening_focus": "band gap, dielectric strength, and defect-controlled carrier density",
            "synthesis_note": "Oxygen-vacancy screening is critical for grounded claims.",
        },
    },
    {
        "concept": "perovskite_photovoltaic",
        "display_name": "perovskite absorber",
        "keywords": ("perovskite", "photovoltaic", "solar absorber", "halide perovskite", "ba zrs3"),
        "property_tags": ("band_gap", "absorption", "defect_tolerance"),
        "prototype": {
            "name": "BaZrS3-like perovskite seed",
            "formula": "BaZrS3",
            "crystal_system": "orthorhombic",
            "space_group": "Pnma",
            "lattice": {"a": 7.06, "b": 9.96, "c": 6.98, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
            "sites": (("Ba", "Ba1", 0.0000, 0.2500, 0.9900), ("Zr", "Zr1", 0.5000, 0.0000, 0.5000), ("S", "S1", 0.2900, 0.0400, 0.7150)),
            "screening_focus": "optical absorption, band gap control, and defect tolerance",
            "synthesis_note": "Phase purity and sulfur activity need explicit checks.",
        },
    },
    {
        "concept": "photocatalyst_oxide",
        "display_name": "oxide photocatalyst",
        "keywords": ("photocatalyst", "water splitting", "bivo4", "visible light", "photoanode"),
        "property_tags": ("band_gap", "band_alignment", "carrier_separation"),
        "prototype": {
            "name": "BiVO4-like seed",
            "formula": "BiVO4",
            "crystal_system": "monoclinic",
            "space_group": "I2/b",
            "lattice": {"a": 5.20, "b": 5.09, "c": 11.70, "alpha": 90.0, "beta": 90.4, "gamma": 90.0},
            "sites": (("Bi", "Bi1", 0.2500, 0.0200, 0.1250), ("V", "V1", 0.2500, 0.5200, 0.1250), ("O", "O1", 0.0200, 0.2900, 0.0400)),
            "screening_focus": "visible-light band gap, favorable band edges, and carrier-separation pathways",
            "synthesis_note": "Surface cocatalysts often dominate practical activity after bulk selection.",
        },
    },
    {
        "concept": "transparent_conductor",
        "display_name": "transparent conducting oxide",
        "keywords": ("transparent conductor", "tco", "transparent conducting oxide", "basno3", "mobility"),
        "property_tags": ("mobility", "band_gap", "dopability"),
        "prototype": {
            "name": "BaSnO3-like TCO seed",
            "formula": "BaSnO3",
            "crystal_system": "cubic",
            "space_group": "Pm-3m",
            "lattice": {"a": 4.12, "b": 4.12, "c": 4.12, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
            "sites": (("Ba", "Ba1", 0.0000, 0.0000, 0.0000), ("Sn", "Sn1", 0.5000, 0.5000, 0.5000), ("O", "O1", 0.5000, 0.5000, 0.0000)),
            "screening_focus": "high mobility, large band gap, and dopant-compatible carrier generation",
            "synthesis_note": "Ground conductivity claims with optical absorption and mobility together.",
        },
    },
    {
        "concept": "two_d_semiconductor",
        "display_name": "2D semiconductor",
        "keywords": ("2d semiconductor", "monolayer", "mos2", "valley", "transition metal dichalcogenide"),
        "property_tags": ("band_gap", "exfoliation", "anisotropy"),
        "prototype": {
            "name": "MoS2-like layered seed",
            "formula": "MoS2",
            "crystal_system": "hexagonal",
            "space_group": "P63/mmc",
            "lattice": {"a": 3.16, "b": 3.16, "c": 12.30, "alpha": 90.0, "beta": 90.0, "gamma": 120.0},
            "sites": (("Mo", "Mo1", 0.3333, 0.6667, 0.2500), ("S", "S1", 0.3333, 0.6667, 0.6210), ("S", "S2", 0.6667, 0.3333, 0.8790)),
            "screening_focus": "band gap, exfoliation energy, and defect-tolerant layered growth",
            "synthesis_note": "Layered candidates still need substrate and interlayer checks.",
        },
    },
    {
        "concept": "porous_framework",
        "display_name": "porous crystalline framework",
        "keywords": ("porous framework", "mof", "zif", "gas capture", "surface area", "adsorption"),
        "property_tags": ("surface_area", "adsorption", "stability"),
        "prototype": {
            "name": "ZIF-8-like porous seed",
            "formula": "Zn(mIm)2",
            "crystal_system": "cubic",
            "space_group": "I-43m",
            "lattice": {"a": 16.99, "b": 16.99, "c": 16.99, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
            "sites": (("Zn", "Zn1", 0.0000, 0.0000, 0.0000), ("C", "C1", 0.3150, 0.1250, 0.2150), ("N", "N1", 0.2500, 0.0830, 0.1670)),
            "screening_focus": "adsorption capacity, pore accessibility, and moisture or thermal stability",
            "synthesis_note": "Framework activation and collapse risk must be screened before claiming usability.",
        },
    },
)

MATTERGEN_CONCEPT_LABELS: Tuple[str, ...] = tuple(str(item["concept"]) for item in MATTERGEN_CONCEPT_SPECS)
MATTERGEN_DISPLAY_NAMES: Dict[str, str] = {str(item["concept"]): str(item["display_name"]) for item in MATTERGEN_CONCEPT_SPECS}
MATTERGEN_CONCEPT_TO_INDEX = {label: index for index, label in enumerate(MATTERGEN_CONCEPT_LABELS)}
MATTERGEN_KEYWORDS: Dict[str, Tuple[str, ...]] = {str(item["concept"]): tuple(str(keyword) for keyword in item["keywords"]) for item in MATTERGEN_CONCEPT_SPECS}


def extract_request_text(prompt: str) -> str:
    cooked = str(prompt or "").strip()
    lowered = cooked.lower()
    for marker in REQUEST_MARKERS:
        idx = lowered.rfind(marker)
        if idx >= 0:
            return cooked[idx + len(marker):].strip()
    return cooked


def normalize_mattergen_text(text: str) -> str:
    cooked = str(text or "").strip()
    cooked = cooked.replace("Ã¢â‚¬â€œ", "-").replace("Ã¢â‚¬â€", "-").replace("Ã¢Ë†â€™", "-")
    cooked = cooked.replace("Å", " A ").replace("μ", " mu ")
    cooked = re.sub(r"\s+", " ", cooked)
    return cooked


def build_vocab(texts: Sequence[str], *, min_frequency: int = 1) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for text in texts:
        for ch in normalize_mattergen_text(text).lower():
            counts[ch] = counts.get(ch, 0) + 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for ch, count in sorted(counts.items()):
        if count >= min_frequency and ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def encode_text(text: str, vocab: Dict[str, int], max_len: int) -> List[int]:
    cooked = normalize_mattergen_text(extract_request_text(text)).lower()
    ids = [vocab.get(ch, 1) for ch in cooked[:max_len]]
    if len(ids) < max_len:
        ids.extend([0] * (max_len - len(ids)))
    return ids


def tokenize_words(text: str, *, max_words: int) -> List[str]:
    cooked = normalize_mattergen_text(extract_request_text(text)).lower()
    return WORD_RE.findall(cooked)[:max_words]


def encode_words(text: str, *, word_buckets: int, max_words: int) -> List[int]:
    ids = [0] * max_words
    for index, token in enumerate(tokenize_words(text, max_words=max_words)):
        ids[index] = (sum(ord(ch) * (offset + 1) for offset, ch in enumerate(token)) % max(word_buckets - 1, 1)) + 1
    return ids


def _extract_condition_features(prompt: str) -> List[float]:
    lowered = normalize_mattergen_text(extract_request_text(prompt)).lower()
    feature_patterns = (
        (r"\bband gap\b|\be_g\b|\bwide gap\b",),
        (r"\bhull\b|\bstable\b|\bformation energy\b|\bphonon\b",),
        (r"\bmagnet|\bcoerc|\banisotropy\b|\bmagnetic density\b",),
        (r"\bionic conductivity\b|\bli[- ]?ion\b|\bsolid electrolyte\b",),
        (r"\bthermoelectric\b|\bseebeck\b|\bz[tT]\b",),
        (r"\bphotovoltaic\b|\bphotocatal|\bvisible light\b|\btransparent\b",),
        (r"\b2d\b|\bmonolayer\b|\bporous\b|\bmof\b",),
        (r"\bgenerate\b|\bdesign\b|\bsuggest\b|\bcandidate\b|\bcif\b",),
    )
    return [1.0 if any(re.search(pattern, lowered) for pattern in patterns) else 0.0 for patterns in feature_patterns]


def vectorize_batch(
    texts: Sequence[str],
    vocab: Dict[str, int],
    *,
    max_len: int,
    word_buckets: int,
    max_words: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    char_rows = [encode_text(text, vocab, max_len) for text in texts]
    word_rows = [encode_words(text, word_buckets=word_buckets, max_words=max_words) for text in texts]
    cond_rows = [_extract_condition_features(text) for text in texts]
    return (
        torch.tensor(char_rows, dtype=torch.long),
        torch.tensor(word_rows, dtype=torch.long),
        torch.tensor(cond_rows, dtype=torch.float32),
    )


def looks_like_mattergen_prompt(prompt: str) -> bool:
    return bool(MATTERGEN_PROMPT_RE.search(str(prompt or "")))


def heuristic_mattergen_concept(prompt: str) -> Optional[str]:
    lowered = normalize_mattergen_text(extract_request_text(prompt)).lower()
    scores: Dict[str, int] = {}
    for concept, keywords in MATTERGEN_KEYWORDS.items():
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
    lowered = normalize_mattergen_text(prompt).lower()
    if any(token in lowered for token in ("cif", "fractional coordinate", "space group", "lattice parameter")):
        return "cif_seed"
    if any(token in lowered for token in ("validate", "screen", "phonon", "dft", "stability", "workflow")):
        return "validation_plan"
    if any(token in lowered for token in ("why", "tradeoff", "rationale", "explain", "reason")):
        return "design_rationale"
    if any(token in lowered for token in ("short", "brief", "concise", "one sentence")):
        return "concise_answer"
    return "candidate_card"


class MatterGenMicroNet(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        num_concepts: int,
        char_embed_dim: int = 30,
        conv_channels: int = 64,
        word_buckets: int = 512,
        word_embed_dim: int = 18,
        condition_dim: int = 8,
        hidden_dim: int = 120,
    ) -> None:
        super().__init__()
        self.char_embedding = nn.Embedding(vocab_size, char_embed_dim, padding_idx=0)
        self.char_conv = nn.Conv1d(char_embed_dim, conv_channels, kernel_size=5, padding=2)
        self.word_embedding = nn.Embedding(word_buckets, word_embed_dim, padding_idx=0)
        self.condition_proj = nn.Linear(condition_dim, 24)
        self.norm = nn.LayerNorm(conv_channels * 2 + word_embed_dim + 24)
        self.fc1 = nn.Linear(conv_channels * 2 + word_embed_dim + 24, hidden_dim)
        self.dropout = nn.Dropout(p=0.16)
        self.fc2 = nn.Linear(hidden_dim, num_concepts)

    def forward(self, char_ids: torch.Tensor, word_ids: torch.Tensor, cond_features: torch.Tensor) -> torch.Tensor:
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
        cond_hidden = torch.relu(self.condition_proj(cond_features))
        features = torch.cat([pooled_avg, pooled_max, word_pool, cond_hidden], dim=1)
        hidden = torch.relu(self.fc1(self.dropout(self.norm(features))))
        return self.fc2(self.dropout(hidden))


@dataclass
class MatterGenPrediction:
    concept: str
    label: str
    confidence: float
    probabilities: Dict[str, float]
    top_predictions: List[Dict[str, float]]
    variant: str


def _extract_float(prompt: str, patterns: Sequence[re.Pattern[str]]) -> Optional[float]:
    text = normalize_mattergen_text(prompt)
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            try:
                return float(match.group(1))
            except Exception:
                continue
    return None


_BAND_GAP_PATTERNS = (
    re.compile(r"band gap(?: of| around| near| target| >=| <=| >| <| =)?\s*([0-9]+(?:\.[0-9]+)?)\s*e?v", re.IGNORECASE),
    re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*e?v band gap", re.IGNORECASE),
)
_DENSITY_PATTERNS = (
    re.compile(r"density(?: of| around| near| <=| <| >=| >| =)?\s*([0-9]+(?:\.[0-9]+)?)\s*g/cm3", re.IGNORECASE),
    re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*g/cm3", re.IGNORECASE),
)
_HULL_PATTERNS = (
    re.compile(r"energy above hull(?: of| around| <=| <| >=| >| =)?\s*([0-9]+(?:\.[0-9]+)?)\s*mev/atom", re.IGNORECASE),
    re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*mev/atom", re.IGNORECASE),
)
_CONDUCTIVITY_PATTERNS = (
    re.compile(r"ionic conductivity(?: of| around| >=| <=| >| <| =)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:s/cm|ms/cm)", re.IGNORECASE),
    re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(?:s/cm|ms/cm)", re.IGNORECASE),
)
_MAG_DENSITY_PATTERNS = (
    re.compile(r"magnetic density(?: of| around| >=| <=| >| <| =)?\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
)


def extract_property_targets(prompt: str) -> Dict[str, float]:
    targets: Dict[str, float] = {}
    if (value := _extract_float(prompt, _BAND_GAP_PATTERNS)) is not None:
        targets["band_gap_eV"] = value
    if (value := _extract_float(prompt, _DENSITY_PATTERNS)) is not None:
        targets["density_g_cm3"] = value
    if (value := _extract_float(prompt, _HULL_PATTERNS)) is not None:
        targets["energy_above_hull_meV_atom"] = value
    if (value := _extract_float(prompt, _CONDUCTIVITY_PATTERNS)) is not None:
        targets["ionic_conductivity_value"] = value
    if (value := _extract_float(prompt, _MAG_DENSITY_PATTERNS)) is not None:
        targets["magnetic_density"] = value
    return targets


def _concept_spec(concept: str) -> Dict[str, Any]:
    for item in MATTERGEN_CONCEPT_SPECS:
        if item["concept"] == concept:
            return item
    raise KeyError(concept)


def _select_prototype(prompt: str, concept: str) -> Dict[str, Any]:
    _ = hashlib.sha1(normalize_mattergen_text(prompt).encode("utf-8")).hexdigest()
    return dict(_concept_spec(concept)["prototype"])


def _format_lattice(lattice: Dict[str, float]) -> str:
    return (
        f"a={lattice['a']:.2f} A, b={lattice['b']:.2f} A, c={lattice['c']:.2f} A, "
        f"alpha={lattice['alpha']:.1f}, beta={lattice['beta']:.1f}, gamma={lattice['gamma']:.1f}"
    )


def _build_cif_seed(prototype: Dict[str, Any], concept: str) -> str:
    lattice = dict(prototype["lattice"])
    lines = [
        f"data_supermix_{concept}",
        f"_symmetry_space_group_name_H-M '{prototype['space_group']}'",
        f"_cell_length_a {lattice['a']:.4f}",
        f"_cell_length_b {lattice['b']:.4f}",
        f"_cell_length_c {lattice['c']:.4f}",
        f"_cell_angle_alpha {lattice['alpha']:.2f}",
        f"_cell_angle_beta {lattice['beta']:.2f}",
        f"_cell_angle_gamma {lattice['gamma']:.2f}",
        "loop_",
        "_atom_site_type_symbol",
        "_atom_site_label",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
    ]
    for symbol, label, x, y, z in prototype["sites"]:
        lines.append(f"{symbol} {label} {x:.4f} {y:.4f} {z:.4f}")
    return "\n".join(lines)


def _format_targets(targets: Dict[str, float]) -> str:
    if not targets:
        return "No explicit numeric property target was detected, so this is a balanced seed candidate."
    parts: List[str] = []
    if "band_gap_eV" in targets:
        parts.append(f"band gap around {targets['band_gap_eV']:.2f} eV")
    if "density_g_cm3" in targets:
        parts.append(f"density near {targets['density_g_cm3']:.2f} g/cm3")
    if "energy_above_hull_meV_atom" in targets:
        parts.append(f"energy above hull capped near {targets['energy_above_hull_meV_atom']:.1f} meV/atom")
    if "ionic_conductivity_value" in targets:
        parts.append(f"ionic-conductivity target around {targets['ionic_conductivity_value']:.3g}")
    if "magnetic_density" in targets:
        parts.append(f"magnetic-density target around {targets['magnetic_density']:.2f}")
    return "Requested conditioning: " + ", ".join(parts) + "."


def render_mattergen_answer(prompt: str, prediction: MatterGenPrediction) -> str:
    prototype = _select_prototype(prompt, prediction.concept)
    targets = extract_property_targets(prompt)
    spec = _concept_spec(prediction.concept)
    lines = [
        f"MatterGen Micro candidate family: {prediction.label}",
        f"Prototype seed: {prototype['name']}",
        f"Formula: {prototype['formula']}",
        f"Crystal system: {prototype['crystal_system']}",
        f"Space group: {prototype['space_group']}",
        f"Lattice: {_format_lattice(dict(prototype['lattice']))}",
        _format_targets(targets),
        f"Screening focus: {prototype['screening_focus']}.",
        "Grounding note: this is a hypothesis seed, not a validated material. Relax it and run stability checks before trusting it.",
    ]
    if prediction.variant == "concise_answer":
        return f"Candidate: {prototype['formula']} ({prototype['space_group']}, {prototype['crystal_system']}) for {prediction.label}. Validate stability before trusting it."
    if prediction.variant == "design_rationale":
        return "\n".join(lines + [f"Why this family: it matches the prompt around {', '.join(spec.get('property_tags') or ())}.", f"Synthesis note: {prototype['synthesis_note']}"])
    if prediction.variant == "validation_plan":
        return "\n".join(
            lines
            + [
                "Validation plan:",
                "1. Relax the structure and recompute the lattice.",
                "2. Check energy above hull and requested property windows.",
                "3. Run phonon or force-constant screening for dynamic stability.",
                "4. Only then move to dopants, defects, or synthesis planning.",
            ]
        )
    return "\n".join(lines + ["Reduced CIF-style seed:", "```cif", _build_cif_seed(prototype, prediction.concept), "```", f"Synthesis note: {prototype['synthesis_note']}"])


class MatterGenMicroEngine:
    def __init__(self, *, weights_path: Path, meta_path: Path, device: Optional[torch.device] = None) -> None:
        self.weights_path = Path(weights_path).resolve()
        self.meta_path = Path(meta_path).resolve()
        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.vocab = {str(key): int(value) for key, value in dict(self.meta.get("vocab") or {}).items()}
        self.labels = tuple(str(item) for item in (self.meta.get("labels") or MATTERGEN_CONCEPT_LABELS))
        self.display_names = {str(key): str(value) for key, value in dict(self.meta.get("display_names") or MATTERGEN_DISPLAY_NAMES).items()}
        self.max_len = int(self.meta.get("max_len") or 256)
        self.max_words = int(self.meta.get("max_words") or 32)
        self.word_buckets = int(self.meta.get("word_buckets") or 512)
        self.device = device or torch.device("cpu")
        self.model = MatterGenMicroNet(
            vocab_size=max(len(self.vocab), 2),
            num_concepts=len(self.labels),
            char_embed_dim=int(self.meta.get("char_embed_dim") or 30),
            conv_channels=int(self.meta.get("conv_channels") or 64),
            word_buckets=self.word_buckets,
            word_embed_dim=int(self.meta.get("word_embed_dim") or 18),
            condition_dim=int(self.meta.get("condition_dim") or 8),
            hidden_dim=int(self.meta.get("hidden_dim") or 120),
        ).to(self.device)
        try:
            state = torch.load(self.weights_path, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(self.weights_path, map_location=self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def predict(self, prompt: str) -> MatterGenPrediction:
        char_tensor, word_tensor, cond_tensor = vectorize_batch(
            [prompt],
            self.vocab,
            max_len=self.max_len,
            word_buckets=self.word_buckets,
            max_words=self.max_words,
        )
        char_tensor = char_tensor.to(self.device)
        word_tensor = word_tensor.to(self.device)
        cond_tensor = cond_tensor.to(self.device)
        with torch.inference_mode():
            logits = self.model(char_tensor, word_tensor, cond_tensor)[0]
            probs = torch.softmax(logits, dim=0).detach().cpu().tolist()
        probabilities = {label: float(prob) for label, prob in zip(self.labels, probs)}
        predicted = max(probabilities, key=probabilities.get)
        confidence = float(probabilities[predicted])
        heuristic = heuristic_mattergen_concept(prompt)
        if heuristic in probabilities and confidence < 0.88:
            heuristic_score = float(probabilities.get(heuristic, 0.0))
            if heuristic_score >= 0.14 or confidence < 0.45:
                predicted = heuristic
                confidence = max(confidence, heuristic_score, 0.91 if confidence < 0.45 else confidence)
        ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        top_predictions = [{"concept": concept, "confidence": round(float(score), 4)} for concept, score in ranked[:3]]
        return MatterGenPrediction(
            concept=predicted,
            label=self.display_names.get(predicted, predicted.replace("_", " ")),
            confidence=confidence,
            probabilities=probabilities,
            top_predictions=top_predictions,
            variant=choose_answer_variant(prompt),
        )

    def answer(self, prompt: str) -> str:
        prediction = self.predict(prompt)
        answer = render_mattergen_answer(prompt, prediction)
        if any(token in prompt.lower() for token in ("brief", "short", "one sentence", "concise")):
            return answer
        if prediction.confidence < 0.52:
            answer += f"\n\nBest matched materials concept: {prediction.label} with low confidence ({prediction.confidence:.2f}), so treat this as a grounded best guess."
        elif any(token in prompt.lower() for token in ("confidence", "which concept", "matched concept")):
            answer += f"\n\nMatched materials concept: {prediction.label} ({prediction.confidence:.2f} confidence)."
        return answer

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
            "paper_inspirations": list(self.meta.get("paper_inspirations") or PAPER_INSPIRATIONS),
        }


def format_mattergen_response(response: str, prediction: MatterGenPrediction) -> str:
    trailer = f"\n\n[materials concept: {prediction.label.lower()} | confidence {prediction.confidence:.2f}]"
    return str(response or "").strip() + trailer
