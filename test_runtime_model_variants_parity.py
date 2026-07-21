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
