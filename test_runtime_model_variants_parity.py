import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def test_runtime_model_variants_snapshot_matches_source():
    completed = subprocess.run(
        [sys.executable, "source/sync_runtime_model_variants.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_runtime_model_variants_imports_without_source_tree(tmp_path: Path):
    isolated = tmp_path / "runtime_python"
    isolated.mkdir()
    for name in ("model_variants.py", "run.py"):
        (isolated / name).write_bytes((ROOT / "runtime_python" / name).read_bytes())

    script = f"""
import sys
sys.path.insert(0, {str(isolated)!r})
import model_variants
assert "cognitive_leap_ultra_expert" in model_variants.SUPPORTED_MODEL_SIZES
model = model_variants.build_model(model_size="cognitive_leap_ultra_expert")
assert model is not None
print(type(model).__name__)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_runtime_snapshot_builds_and_runs_v52_without_source_tree(tmp_path: Path):
    """A shipped runtime_python/ must build v52 on its own, controls included.

    This is the point of merging the v52 head into the generated snapshot: the
    packaged runtime has no source/ sibling on disk.
    """

    isolated = tmp_path / "runtime_python"
    isolated.mkdir()
    for name in ("model_variants.py", "run.py"):
        (isolated / name).write_bytes((ROOT / "runtime_python" / name).read_bytes())

    script = f"""
import sys
import torch
sys.path.insert(0, {str(isolated)!r})
import model_variants

assert "cognitive_leap_v52_expert" in model_variants.SUPPORTED_MODEL_SIZES

torch.manual_seed(0)
model = model_variants.build_model(model_size="cognitive_leap_v52_expert")
model.eval()
head = model.layers[10]
assert type(head).__name__ == "CognitiveLeapV52ExpertHead", type(head).__name__

x = torch.randn(2, 1, 128)
with torch.no_grad():
    dense = model(x)
    sparse = model(x, core_top_k=2)
    verified = model(x, verifier_adaptive_compute=True, verifier_continue_threshold=0.1)

assert dense.shape == sparse.shape == verified.shape
for attr in ("last_router_load_balance", "last_active_cores", "last_quality_score",
             "last_continue_probability", "last_calibrated_entropy"):
    assert hasattr(head, attr), attr
print("v52 ok:", type(head).__name__, float(head.last_active_cores))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "v52 ok: CognitiveLeapV52ExpertHead" in completed.stdout
