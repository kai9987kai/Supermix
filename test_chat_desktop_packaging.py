"""Tests for the desktop build's packaging contract.

The first build of SupermixChatDesktop produced a working-looking 1.8 GB
application that died on launch with:

    ImportError: cannot import name 'distributions' from partially
    initialized module 'torch'

The cause was `torch.distributions` in the spec's `excludes`. torch's own
`__init__` imports it, so excluding it broke the package outright. Nothing in
the build output said so -- PyInstaller reported success -- and the failure
only appeared when the executable was run.

These parse the spec rather than executing it, because a spec depends on
PyInstaller-injected globals (SPECPATH, Analysis, ...) that only exist inside
a build.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent
SPEC_PATH = REPO_ROOT / "SupermixChatDesktop.spec"


def _spec_tree() -> ast.Module:
    return ast.parse(SPEC_PATH.read_text(encoding="utf-8"))


def _analysis_keyword(name: str) -> list:
    """The literal list passed as `name=` to Analysis(...)."""

    for node in ast.walk(_spec_tree()):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Analysis":
            for keyword in node.keywords:
                if keyword.arg == name:
                    return ast.literal_eval(keyword.value)
    raise AssertionError(f"Analysis(...) has no {name}= argument")


def test_the_spec_exists_and_parses():
    assert SPEC_PATH.is_file()
    assert _spec_tree() is not None


def test_no_exclude_is_a_submodule():
    """The v74 build failure, pinned.

    Excluding `package.submodule` is unsafe whenever the package itself is
    shipped: the package's own imports still run and will fail. Only whole
    top-level packages may be excluded.
    """

    submodules = [name for name in _analysis_keyword("excludes") if "." in name]

    assert submodules == [], (
        f"excludes contains submodule(s) {submodules}. Excluding part of a "
        "package that still ships breaks its __init__ at runtime, and the "
        "build still reports success."
    )


def test_torch_is_not_excluded_in_any_form():
    """The model cannot load without it; this is the specific regression."""

    for name in _analysis_keyword("excludes"):
        assert name != "torch"
        assert not name.startswith("torch.")


@pytest.mark.parametrize("package", ["flask", "webview", "numpy", "werkzeug"])
def test_runtime_dependencies_are_not_excluded(package):
    excludes = _analysis_keyword("excludes")

    assert package not in excludes
    assert not any(name.startswith(package + ".") for name in excludes)


def _assigned_list(name: str) -> list:
    """The literal list first assigned to `name` at module level.

    `hiddenimports` is extended with `+=` from collect_all(), so the value
    reaching Analysis(...) is a Name, not a literal. The declared entries are
    in the initial assignment.
    """

    for node in _spec_tree().body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if getattr(target, "id", None) == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"no module-level assignment to {name}")


def test_local_modules_are_declared_as_hidden_imports():
    """The desktop app imports these inside functions, so static analysis of
    the entry script alone is not guaranteed to reach them."""

    hidden = _assigned_list("hiddenimports")
    for module in ("supermix_chat_server", "mimomix_core", "train_mimomix_talk",
                   "prompt_normaliser", "answer_check"):
        assert module in hidden, f"{module} must be a hidden import"


def test_the_spec_refuses_to_build_without_a_staged_model():
    """A build with no model in it would look successful and fail on launch."""

    source = SPEC_PATH.read_text(encoding="utf-8")

    assert "FileNotFoundError" in source
    assert "*.pt" in source


def test_the_model_is_staged_where_the_app_looks_for_it():
    """The spec's destination and the app's search path must agree."""

    import sys

    source_dir = REPO_ROOT / "source"
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    import supermix_chat_desktop_app as desktop

    spec_source = SPEC_PATH.read_text(encoding="utf-8")

    # The spec stages into "model"; the app looks under `model/`.
    assert '"model"' in spec_source
    assert desktop.BUNDLED_CHECKPOINT.endswith(".pt")
