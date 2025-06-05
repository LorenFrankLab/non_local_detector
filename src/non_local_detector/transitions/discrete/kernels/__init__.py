from importlib import import_module
from pathlib import Path

# auto-import every .py file in this directory except __init__.py
for path in Path(__file__).parent.glob("*.py"):
    if path.stem != "__init__":
        import_module(f".{path.stem}", package=__name__)
