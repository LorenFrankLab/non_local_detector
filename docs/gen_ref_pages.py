"""Generate one API reference page per public module of non_local_detector."""

import ast
from pathlib import Path

import mkdocs_gen_files

# Internal modules and packages, read from a file the public-API test shares.
INTERNAL_MODULES = tuple(
    f"non_local_detector.{line.strip()}"
    for line in (Path(__file__).parent / "internal_modules.txt")
    .read_text()
    .splitlines()
    if line.strip() and not line.startswith("#")
)


def is_internal(ident: str) -> bool:
    """Whether a dotted module name is, or is inside, an internal module."""
    return any(ident == m or ident.startswith(f"{m}.") for m in INTERNAL_MODULES)


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

    # Rule: a module is documented, with only its ``__all__`` names, when it
    # defines a non-empty ``__all__`` and is not an internal package's
    # ``__init__`` (``likelihoods.common`` re-exports names for pickles). An
    # internal module without ``__all__`` is skipped; a public one without
    # it is a bug that tests/test_public_api.py reports.
    public_names = read_dunder_all(path)
    if not public_names or (path.name == "__init__.py" and is_internal(ident)):
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
