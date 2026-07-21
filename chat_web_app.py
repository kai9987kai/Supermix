"""Compatibility entrypoint for the canonical source chat web app.

The implementation lives in ``source/chat_web_app.py``. Keeping this thin
wrapper at the repository root prevents imports from binding to an older copy
when tests or scripts import ``chat_web_app`` before adding ``source`` to
``sys.path``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parent
_SOURCE_DIR = _ROOT / "source"
_IMPL_PATH = _SOURCE_DIR / "chat_web_app.py"

if str(_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOURCE_DIR))

_SPEC = importlib.util.spec_from_file_location("_supermix_source_chat_web_app", _IMPL_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load chat web app implementation from {_IMPL_PATH}")

_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

__all__ = [name for name in vars(_IMPL) if not name.startswith("_")]

for _name in __all__:
    globals()[_name] = getattr(_IMPL, _name)

# Keep the root compatibility surface behaviorally identical for callers that
# validate CLI default inheritance after another test imported this wrapper
# first. The canonical helper remains private in the source implementation.
for _name in ("_runtime_compute_cli_overrides",):
    globals()[_name] = getattr(_IMPL, _name)


def main() -> None:
    _IMPL.main()


if __name__ == "__main__":
    main()
