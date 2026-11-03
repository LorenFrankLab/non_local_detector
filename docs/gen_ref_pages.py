"""Generate one API reference page per public module of non_local_detector."""

from pathlib import Path

import mkdocs_gen_files

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

    nav[parts] = doc_path.as_posix()

    ident = ".".join(parts)
    if path.name == "__init__.py":
        # A package page shows its docstring; its re-exported objects are
        # documented on their own modules' pages.
        options = "    options:\n      members: false\n"
    elif parts[:2] == ("non_local_detector", "models") and parts[-1] != "base":
        # The models' fit/predict/... are inherited from models.base.
        options = "    options:\n      inherited_members: true\n"
    else:
        options = ""
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"# `{ident}`\n\n::: {ident}\n{options}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(src_root.parent))

with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
