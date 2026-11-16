"""Cross-modal representation and projector bridge for SCF."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

SOURCE_DIR = Path(__file__).resolve().parents[2]
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

__all__ = ["SCFMultimodalBridge"]


class SCFMultimodalBridge:
    """Safely bridges image, video, 3D mesh, protein, and materials adapters."""

    def capabilities(self) -> Dict[str, Any]:
        mods: Dict[str, bool] = {}
        for mod_name in (
            "image_feature_utils",
            "image_recognition_model",
            "video_feature_utils",
            "mesh_feature_utils",
            "protein_folding_model",
            "mattergen_generation_model",
            "three_d_generation_model",
        ):
            try:
                __import__(mod_name)
                mods[mod_name] = True
            except Exception:
                mods[mod_name] = False
        return {
            "available_modules": mods,
            "mimomix_multimodal_projector_supported": True,
        }

    def prepare_multimodal_context(
        self,
        *,
        prompt: str,
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
        mesh_path: Optional[str] = None,
        project_to_dim: int = 0,
    ) -> Dict[str, Any]:
        """Extract available cross-modal features without failing on missing files/deps."""

        descriptors = []
        warnings = []
        vectors = []

        if image_path:
            p = Path(image_path)
            if p.is_file():
                try:
                    import image_feature_utils as ifu

                    descriptors.append({"modality": "image", "summary": ifu.describe_image_for_text(str(p))})
                    vectors.append(list(ifu.extract_image_numeric_features(str(p))))
                except Exception as exc:
                    warnings.append(f"image_extraction_failed:{type(exc).__name__}")
            else:
                warnings.append("image_path_missing")

        if video_path:
            p = Path(video_path)
            if p.is_file():
                try:
                    import video_feature_utils as vfu

                    descriptors.append({"modality": "video", "summary": vfu.describe_video_for_text(str(p))})
                    vectors.append(list(vfu.extract_video_numeric_features(str(p))))
                except Exception as exc:
                    warnings.append(f"video_extraction_failed:{type(exc).__name__}")
            else:
                warnings.append("video_path_missing")

        if mesh_path:
            p = Path(mesh_path)
            if p.is_file():
                try:
                    import mesh_feature_utils as mfu

                    descriptors.append({"modality": "3d_mesh", "summary": mfu.describe_3d_model_for_text(str(p))})
                    vectors.append(list(mfu.extract_3d_model_numeric_features(str(p))))
                except Exception as exc:
                    warnings.append(f"mesh_extraction_failed:{type(exc).__name__}")
            else:
                warnings.append("mesh_path_missing")

        protein_hint = False
        try:
            import protein_folding_model as pfm

            protein_hint = bool(pfm.looks_like_protein_folding_prompt(prompt))
        except Exception:
            pass

        projected_shape = None
        if int(project_to_dim) > 0 and vectors:
            import torch
            import mimomix_core as mc

            cfg = mc.MiMoMixConfig(
                vocab_size=64,
                hidden_size=max(8, int(project_to_dim)),
                n_layers=1,
                n_heads=2,
                n_kv_heads=1,
                intermediate_size=16,
                use_moe=False,
                n_mtp_layers=0,
                use_thinking_core=False,
                use_multimodal=True,
                multimodal_input_dim=len(vectors[0]),
            )
            flat = torch.tensor(vectors[0], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            proj = torch.nn.Linear(cfg.multimodal_input_dim, cfg.hidden_size, bias=False)
            with torch.no_grad():
                projected_shape = list(proj(flat).shape)

        return {
            "modalities_detected": len(descriptors),
            "descriptors": descriptors,
            "protein_prompt_detected": protein_hint,
            "projected_shape": projected_shape,
            "warnings": warnings,
        }
