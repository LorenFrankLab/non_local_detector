import inspect
from typing import Any, Dict, List, Type

from non_local_detector.environment.layout.base import LayoutEngine
from non_local_detector.environment.layout.engines.graph import GraphLayout
from non_local_detector.environment.layout.engines.hexagonal import HexagonalLayout
from non_local_detector.environment.layout.engines.image_mask import ImageMaskLayout
from non_local_detector.environment.layout.engines.masked_grid import MaskedGridLayout
from non_local_detector.environment.layout.engines.regular_grid import RegularGridLayout
from non_local_detector.environment.layout.engines.shapely_polygon import (
    ShapelyPolygonLayout,
)
from non_local_detector.environment.layout.engines.triangular_mesh import (
    TriangularMeshLayout,
)

_LAYOUT_MAP: Dict[str, Type[LayoutEngine]] = {
    "RegularGrid": RegularGridLayout,
    "MaskedGrid": MaskedGridLayout,
    "ImageMask": ImageMaskLayout,
    "Hexagonal": HexagonalLayout,
    "Graph": GraphLayout,
    "TriangularMesh": TriangularMeshLayout,
    "ShapelyPolygon": ShapelyPolygonLayout,
}


def _normalize_name(name: str) -> str:
    """
    Normalize a layout name by removing non-alphanumeric characters and
    converting to lowercase.

    Parameters
    ----------
    name : str
        The layout name to normalize.

    Returns
    -------
    str
        The normalized name.
    """
    return "".join(filter(str.isalnum, name)).lower()


def list_available_layouts() -> List[str]:
    """
    List user-friendly type strings for all available layout engines.

    Returns
    -------
    List[str]
        A sorted list of unique string identifiers for available
        `LayoutEngine` types (e.g., "RegularGrid", "Hexagonal").
    """
    unique_options: List[str] = []
    processed_normalized_options: set[str] = set()
    for opt in _LAYOUT_MAP.keys():
        norm_opt = _normalize_name(opt)
        if norm_opt not in processed_normalized_options:
            unique_options.append(opt)
            processed_normalized_options.add(norm_opt)
    return sorted(unique_options)


def get_layout_parameters(layout_type: str) -> Dict[str, Dict[str, Any]]:
    """
    Retrieve expected build parameters for a specified layout engine type.

    Inspects the `build` method signature of the specified `LayoutEngine`
    class to determine its required and optional parameters.

    Parameters
    ----------
    layout_type : str
        The string identifier of the layout engine type (case-insensitive,
        ignores non-alphanumeric characters).

    Returns
    -------
    Dict[str, Dict[str, Any]]
        A dictionary where keys are parameter names for the `build` method.
        Each value is another dictionary containing:
        - 'annotation': The type annotation of the parameter.
        - 'default': The default value, or `None` if no default.
        - 'kind': The parameter kind (e.g., 'keyword-only').

    Raises
    ------
    ValueError
        If `layout_type` is unknown.
    """
    normalized_kind_query = _normalize_name(layout_type)
    found_key = next(
        (
            name
            for name in _LAYOUT_MAP
            if _normalize_name(name) == normalized_kind_query
        ),
        None,
    )
    if not found_key:
        raise ValueError(
            f"Unknown engine kind '{layout_type}'. Available: {list_available_layouts()}"
        )
    engine_class = _LAYOUT_MAP[found_key]
    sig = inspect.signature(engine_class.build)
    params_info: Dict[str, Dict[str, Any]] = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        params_info[name] = {
            "annotation": (
                param.annotation
                if param.annotation is not inspect.Parameter.empty
                else None
            ),
            "default": (
                param.default if param.default is not inspect.Parameter.empty else None
            ),
            "kind": param.kind.description,
        }
    return params_info


def create_layout(kind: str, **kwargs) -> LayoutEngine:
    """
    Factory for creating and building a spatial-layout engine.

    Parameters
    ----------
    kind : str
        Case-insensitive name of the layout engine to create
        (e.g., "RegularGrid", "Hexagonal", "Graph", etc.).
    **kwargs : any
        Parameters passed to the chosen engine’s `build(...)` method.

    Returns
    -------
    LayoutEngine
        A fully constructed layout engine.

    Raises
    ------
    ValueError
        - If `kind` is not one of the available layouts.
        - If any unexpected keyword arguments are passed to `build`.
    """
    # 1) Normalize user input and find matching key
    norm_query = "".join(ch for ch in kind if ch.isalnum()).lower()
    found_key = next(
        (
            name
            for name in _LAYOUT_MAP
            if "".join(ch for ch in name if ch.isalnum()).lower() == norm_query
        ),
        None,
    )
    if found_key is None:
        suggestions = ", ".join(list_available_layouts())
        raise ValueError(f"Unknown layout kind '{kind}'. Available: {suggestions}")

    # 2) Instantiate the class
    engine_cls = _LAYOUT_MAP[found_key]
    engine = engine_cls()

    # 3) Validate `kwargs` against `build(...)` signature
    sig = inspect.signature(engine.build)
    allowed = {param for param in sig.parameters if param != "self"}
    unexpected = set(kwargs) - allowed
    if unexpected:
        raise ValueError(f"Unexpected arguments for {found_key}.build(): {unexpected}")

    # 4) Call `build(...)` with validated params
    engine.build(**{k: kwargs[k] for k in allowed if k in kwargs})
    return engine
