"""Discover known MiMoMix talk weights in source checkouts and model releases."""

from pathlib import Path


# v96 is the promoted default. The other entries retain their selection priority.
TALK_CHECKPOINTS = (
    "v96_physio_neurogenesis.selected.pt",
    "v100_scf_balanced.selected.pt",
    "v101_scf_cognitive.pt",
    "v102_scf_frontier.pt",
    "v103_scf_synthesis.pt",
    "v98_scf_predictive.selected.pt",
    "v99_scf_evolution.selected.pt",
    "v97_scf_connectome.selected.pt",
    "v93_neurogenesis.selected.pt",
    "v91_cns_connectome.selected.pt",
    "v89_corpus.pt",
    "v80_omni.pt",
    "v74_broad.pt",
    "v57_talk_v2.pt",
    "v88_corpus.selected.pt",
    "v91_cns_rewired.selected.pt",
    "v91_control.selected.pt",
)


def discover_talk_models(root: Path) -> dict[str, Path]:
    """Map model names to relative paths, preferring training outputs over copies."""
    models = {}
    for filename in TALK_CHECKPOINTS:
        name = filename.split(".", 1)[0]
        run_dir = "hf_v57_publish" if name == "v57_talk_v2" else name
        for path in (Path("output", run_dir, filename), Path("weights", filename)):
            if (root / path).is_file():
                models[name] = path
                break
    return models
