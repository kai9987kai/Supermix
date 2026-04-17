"""Runtime proxy for shared ChampionNet model variant definitions.

This runtime bundle should use the same architecture registry as ``source/``
so expert variants and checkpoint detection stay in sync.
"""

from importlib.util import module_from_spec as _module_from_spec, spec_from_file_location as _spec_from_file_location
from pathlib import Path as _Path


_SOURCE_MODEL_VARIANTS = _Path(__file__).resolve().parents[1] / "source" / "model_variants.py"
_SPEC = _spec_from_file_location("_shared_source_model_variants", _SOURCE_MODEL_VARIANTS)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load shared model variants from {_SOURCE_MODEL_VARIANTS}")

_MODULE = _module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

__doc__ = getattr(_MODULE, "__doc__", __doc__)

for _name, _value in vars(_MODULE).items():
    if _name.startswith("__") and _name not in {"__all__", "__doc__"}:
        continue
    globals()[_name] = _value

__all__ = getattr(
    _MODULE,
    "__all__",
    tuple(name for name in globals() if not name.startswith("_")),
)
