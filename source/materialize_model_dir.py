import shutil
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python source/materialize_model_dir.py <source_dir> <dest_dir>")

    source_dir = Path(sys.argv[1]).expanduser().resolve()
    dest_dir = Path(sys.argv[2]).expanduser().resolve()

    if not source_dir.is_dir():
        raise SystemExit(f"source model directory does not exist: {source_dir}")

    if dest_dir.exists():
        shutil.rmtree(dest_dir)

    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, dest_dir, symlinks=False)


if __name__ == "__main__":
    main()
