"""AST-based import-boundary CI gate.

Two rules, both ship in this file and run on every default ``pytest``
invocation:

1. **Qt-import allowlist** — Inside
   ``src/non_local_detector/visualization/interactive/``, no Python
   file may ``import``/``from`` ``pyqtgraph`` or ``PySide6`` outside
   the allowlist (``viewer/qt.py`` and ``panels/qt/**/*.py``). This
   keeps ``view_models/``, ``data_source.py``, ``viewer/core.py``,
   ``viewer/backend.py``, and ``app.py`` GUI-toolkit-free so v2's
   Panel/holoviews backend can re-use them unchanged.

2. **encoding_model_ allowlist** — Repo-wide. The only legitimate
   ``detector.encoding_model_[...]`` access lives in
   ``src/non_local_detector/analysis/place_fields.py`` (the helper
   module the static plot, the viewer slice panel, and Track A
   cache-builder all route through) and ``src/non_local_detector/
   models/base.py`` (the writer that builds the dict during fit).
   Any new consumer should go through the helpers — not reach into
   the dict directly.

Both rules are AST-based (not grep) because the relevant strings
appear legitimately in ``pyproject.toml``'s ``[viewer]`` extra and in
plan/doc prose.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

# __file__ → src/non_local_detector/tests/lint/test_import_boundary.py
# parents[2] → src/non_local_detector/ (the package root)
PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[2]
INTERACTIVE_ROOT = PACKAGE_ROOT / "visualization" / "interactive"

QT_BANNED = {"pyqtgraph", "PySide6"}
QT_ALLOWLIST_PREFIXES = ("viewer/qt.py", "panels/qt/")

ENCODING_MODEL_ALLOWLIST = {
    PACKAGE_ROOT / "analysis" / "place_fields.py",
    PACKAGE_ROOT / "models" / "base.py",
}


def _import_modules(node: ast.AST) -> list[str]:
    """Return the top-level module names imported by an AST node."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        return [node.module] if node.module else []
    return []


@pytest.mark.unit
def test_no_qt_imports_outside_allowlist() -> None:
    """Qt modules may only be imported under the Qt allowlist paths.

    AST-based scan — false-positives on string mentions are avoided
    because we only inspect ``Import`` / ``ImportFrom`` nodes.
    """
    if not INTERACTIVE_ROOT.is_dir():
        pytest.skip(
            f"Interactive sub-package not present at {INTERACTIVE_ROOT}; "
            "rule has nothing to enforce yet."
        )
    violations: list[str] = []
    for path in INTERACTIVE_ROOT.rglob("*.py"):
        rel = path.relative_to(INTERACTIVE_ROOT)
        if any(str(rel).startswith(p) for p in QT_ALLOWLIST_PREFIXES):
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError as exc:
            violations.append(f"{rel}: SyntaxError: {exc}")
            continue
        for node in ast.walk(tree):
            for module in _import_modules(node):
                top = (module or "").split(".")[0]
                if top in QT_BANNED:
                    violations.append(
                        f"{rel}: imports {module!r} but is not under "
                        f"the Qt allowlist ({list(QT_ALLOWLIST_PREFIXES)!r})."
                    )
    assert not violations, "Qt-import allowlist violations:\n" + "\n".join(violations)


def _is_encoding_model_subscript(node: ast.AST) -> bool:
    """Return True for ``<expr>.encoding_model_[<key>]`` subscripts."""
    if not isinstance(node, ast.Subscript):
        return False
    value = node.value
    return isinstance(value, ast.Attribute) and value.attr == "encoding_model_"


@pytest.mark.unit
def test_no_encoding_model_subscript_outside_allowlist() -> None:
    """Repo-wide: only the place-fields helper and the writer may
    subscript ``detector.encoding_model_[...]``.

    Catches new consumers regressing from the helper. Walks every
    ``*.py`` under ``src/non_local_detector/`` and looks for
    ``Subscript`` nodes whose ``value`` is an ``Attribute`` matching
    ``*.encoding_model_``. Tests are excluded — they hand-build
    ``encoding_model_`` payloads for fixtures and that's expected.
    """
    violations: list[str] = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        # Tests legitimately reach into encoding_model_ to construct
        # fixtures.
        if "/tests/" in str(path):
            continue
        if path in ENCODING_MODEL_ALLOWLIST:
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError as exc:
            violations.append(f"{path}: SyntaxError: {exc}")
            continue
        for node in ast.walk(tree):
            if _is_encoding_model_subscript(node):
                violations.append(
                    f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno}: subscripts "
                    f"`encoding_model_[...]` but is not in the allowlist "
                    f"({sorted(p.relative_to(PACKAGE_ROOT) for p in ENCODING_MODEL_ALLOWLIST)!r})."
                )
    assert not violations, "encoding_model_[...] allowlist violations:\n" + "\n".join(
        violations
    )
