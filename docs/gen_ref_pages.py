"""Generate one API reference page per public module of non_local_detector."""

import ast
from pathlib import Path

import mkdocs_gen_files

# The likelihood algorithms are internal; ``likelihoods.common`` keeps a
# non-empty ``__all__`` only because it re-exports the names of the module it
# replaced, so pickles and imports that name them still resolve.
INTERNAL_PACKAGES = ("non_local_detector.likelihoods",)


def read_dunder_all(path: Path) -> list[str] | None:
    """Return the names in a module's literal ``__all__``, or None if it has none."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            return list(ast.literal_eval(value))
    return None


nav = mkdocs_gen_files.Nav()

src_root = Path(__file__).parent.parent / "src"
package_root = src_root / "non_local_detector"

for path in sorted(package_root.rglob("*.py")):
    module_path = path.relative_to(src_root).with_suffix("")
    doc_path = path.relative_to(src_root).with_suffix(".md")
    full_doc_path = Path("api") / doc_path
    parts = tuple(module_path.parts)

    # Underscore modules (_version, _validation, ...) are private.
    if "__pycache__" in parts or any(
        part.startswith("_") and part != "__init__" for part in parts
    ):
        continue

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")

    ident = ".".join(parts)

    # Only modules that export names are public API; ``__all__ = []`` marks an
    # internal module, and a module without ``__all__`` is a bug that
    # tests/test_public_api.py reports.
    public_names = read_dunder_all(path)
    if not public_names or ident.startswith(INTERNAL_PACKAGES):
        continue

    nav[parts] = doc_path.as_posix()
    if path.name == "__init__.py":
        # A package page shows its docstring; its re-exported objects are
        # documented on their own modules' pages.
        option_lines = ["members: false"]
    else:
        # Document exactly the names the module lists in ``__all__``.
        option_lines = ["members:", *(f"  - {name}" for name in public_names)]
        if parts[:2] == ("non_local_detector", "models") and parts[-1] != "base":
            # The models' fit/predict/... are inherited from models.base.
            option_lines.append("inherited_members: true")
    options = "    options:\n" + "".join(f"      {line}\n" for line in option_lines)
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"# `{ident}`\n\n::: {ident}\n{options}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(src_root.parent))

with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
