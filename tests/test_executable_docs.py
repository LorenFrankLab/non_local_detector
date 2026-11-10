"""Code blocks marked ``<!-- docs-test: run -->`` in the docs run as written."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
MARKED_BLOCK = re.compile(
    r"<!-- docs-test: run -->\s*\n```python\n(?P<code>.*?)\n```", re.DOTALL
)
pytestmark = pytest.mark.skipif(
    not DOCS.is_dir(), reason="docs/ is not part of this installation"
)
PAGES = sorted(
    path.relative_to(DOCS).as_posix()
    for path in DOCS.rglob("*.md")
    if MARKED_BLOCK.search(path.read_text())
)


def test_the_marker_finds_pages():
    """A typo in the marker would otherwise skip every page silently."""
    assert "getting-started/quickstart.md" in PAGES


@pytest.mark.integration
@pytest.mark.parametrize("page", PAGES)
def test_marked_blocks_run(page, tmp_path, monkeypatch):
    """Run a page's marked blocks in order, in one namespace, as a reader would."""
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {"__name__": "__main__"}
    try:
        for match in MARKED_BLOCK.finditer((DOCS / page).read_text()):
            exec(compile(match["code"], f"{page}", "exec"), namespace)
    finally:
        plt.close("all")
