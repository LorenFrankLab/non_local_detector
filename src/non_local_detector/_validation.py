"""Internal validation utilities for parameter checking.

This module provides reusable validation functions following Raymond Hettinger's
principle: "Functions are better than classes when you don't need state."

These functions are for internal use only (note the leading underscore in module name).
"""

from typing import Any

import numpy as np

from non_local_detector.exceptions import DataError, ValidationError


def ensure_probability_distribution(
    arr: np.ndarray, name: str, tolerance: float = 1e-6
) -> None:
    """Verify array is a valid probability distribution (sums to 1).

    Parameters
    ----------
    arr : np.ndarray
        Array that should sum to 1.0
    name : str
        Name of the array for error messages
    tolerance : float, optional
        Tolerance for sum check, by default 1e-6

    Raises
    ------
    ValidationError
        If array does not sum to 1.0 within tolerance

    Examples
    --------
    >>> ensure_probability_distribution(np.array([0.5, 0.5]), "probs")  # OK
    >>> ensure_probability_distribution(np.array([0.6, 0.6]), "probs")  # Raises
    """
    arr_sum = np.sum(arr)
    if not np.isclose(arr_sum, 1.0, atol=tolerance):
        raise ValidationError(
            f"{name} must be a valid probability distribution",
            expected="sum = 1.0",
            got=f"sum = {arr_sum:.6f}",
            hint=f"Normalize the array: {name} = {name} / {name}.sum()",
            example=f"    {name} = np.array([0.25, 0.25, 0.25, 0.25])  # sums to 1.0",
        )


def ensure_positive_scalar(
    value: float, name: str, minimum: float = 0.0, strict: bool = True
) -> None:
    """Verify value is positive (or non-negative).

    Parameters
    ----------
    value : float
        Value to check
    name : str
        Name of the parameter for error messages
    minimum : float, optional
        Minimum allowed value, by default 0.0
    strict : bool, optional
        If True, value must be > minimum. If False, value must be >= minimum.
        By default True.

    Raises
    ------
    ValidationError
        If value does not meet the constraint

    Examples
    --------
    >>> ensure_positive_scalar(0.5, "concentration")  # OK
    >>> ensure_positive_scalar(0.0, "concentration")  # Raises (strict=True)
    >>> ensure_positive_scalar(0.0, "regularization", strict=False)  # OK
    """
    if strict:
        condition = value > minimum
        expected_str = f"{name} > {minimum}"
    else:
        condition = value >= minimum
        expected_str = f"{name} >= {minimum}"

    if not condition:
        raise ValidationError(
            f"Invalid value for {name}",
            expected=expected_str,
            got=f"{name} = {value}",
            hint=f"Use a {'positive' if strict else 'non-negative'} value",
            example=f"    {name} = {1.0 if strict else 0.0}",
        )


def ensure_array_1d(arr: np.ndarray, name: str) -> None:
    """Verify array is 1-dimensional.

    Parameters
    ----------
    arr : np.ndarray
        Array to check
    name : str
        Name of the array for error messages

    Raises
    ------
    ValidationError
        If array is not 1D

    Examples
    --------
    >>> ensure_array_1d(np.array([1, 2, 3]), "probs")  # OK
    >>> ensure_array_1d(np.array([[1, 2]]), "probs")  # Raises
    """
    if arr.ndim != 1:
        raise ValidationError(
            f"{name} must be a 1-dimensional array",
            expected="array with shape (n,)",
            got=f"array with shape {arr.shape}",
            hint="Use array.flatten() or array.ravel() to convert to 1D",
            example=f"    {name} = np.array([0.25, 0.25, 0.25, 0.25])",
        )


def ensure_all_finite(arr: np.ndarray, name: str) -> None:
    """Verify all array elements are finite (no NaN or Inf).

    Parameters
    ----------
    arr : np.ndarray
        Array to check
    name : str
        Name of the array for error messages

    Raises
    ------
    DataError
        If array contains NaN or Inf values

    Examples
    --------
    >>> ensure_all_finite(np.array([1.0, 2.0]), "data")  # OK
    >>> ensure_all_finite(np.array([1.0, np.nan]), "data")  # Raises
    """
    if not np.all(np.isfinite(arr)):
        n_nan = np.sum(np.isnan(arr))
        n_inf = np.sum(np.isinf(arr))
        raise DataError(
            f"Found non-finite values in {name}",
            data_name=name,
            hint=f"Array contains {n_nan} NaN value(s) and {n_inf} Inf value(s). Check your data for missing or invalid values.",
        )


def ensure_all_non_negative(arr: np.ndarray, name: str) -> None:
    """Verify all array elements are non-negative.

    Parameters
    ----------
    arr : np.ndarray
        Array to check
    name : str
        Name of the array for error messages

    Raises
    ------
    ValidationError
        If array contains negative values

    Examples
    --------
    >>> ensure_all_non_negative(np.array([0.0, 1.0]), "probs")  # OK
    >>> ensure_all_non_negative(np.array([-0.1, 1.0]), "probs")  # Raises
    """
    if np.any(arr < 0):
        n_negative = np.sum(arr < 0)
        min_val = np.min(arr)
        raise ValidationError(
            f"{name} must contain only non-negative values",
            expected="all values >= 0",
            got=f"{n_negative} negative value(s), minimum = {min_val:.6f}",
            hint="Check your data for errors or use np.abs() if negative values should be positive",
        )


def ensure_in_range(arr: np.ndarray, name: str, low: float, high: float) -> None:
    """Verify all array elements are within a range [low, high].

    Parameters
    ----------
    arr : np.ndarray
        Array to check
    name : str
        Name of the array for error messages
    low : float
        Minimum allowed value (inclusive)
    high : float
        Maximum allowed value (inclusive)

    Raises
    ------
    ValidationError
        If array contains values outside the range

    Examples
    --------
    >>> ensure_in_range(np.array([0.5, 0.8]), "probs", 0.0, 1.0)  # OK
    >>> ensure_in_range(np.array([1.5]), "probs", 0.0, 1.0)  # Raises
    """
    if np.any(arr < low) or np.any(arr > high):
        min_val = np.min(arr)
        max_val = np.max(arr)
        raise ValidationError(
            f"{name} values must be in range [{low}, {high}]",
            expected=f"all values in [{low}, {high}]",
            got=f"values in [{min_val:.6f}, {max_val:.6f}]",
            hint=f"Use np.clip({name}, {low}, {high}) to clamp values to valid range",
        )


def ensure_square_matrix(matrix: np.ndarray, name: str) -> None:
    """Verify matrix is square (n x n).

    Parameters
    ----------
    matrix : np.ndarray
        Matrix to check
    name : str
        Name of the matrix for error messages

    Raises
    ------
    ValidationError
        If matrix is not square

    Examples
    --------
    >>> ensure_square_matrix(np.eye(3), "transition_matrix")  # OK
    >>> ensure_square_matrix(np.zeros((3, 4)), "transition_matrix")  # Raises
    """
    if matrix.ndim != 2:
        raise ValidationError(
            f"{name} must be a 2-dimensional matrix",
            expected="matrix with shape (n, n)",
            got=f"array with shape {matrix.shape}",
            hint="Transition matrices must be 2D and square",
        )

    if matrix.shape[0] != matrix.shape[1]:
        raise ValidationError(
            f"{name} must be square",
            expected=f"matrix with shape ({matrix.shape[0]}, {matrix.shape[0]})",
            got=f"matrix with shape {matrix.shape}",
            hint="Transition matrix needs one row and one column per state",
        )


def ensure_stochastic_matrix(
    matrix: np.ndarray, name: str, tolerance: float = 1e-6
) -> None:
    """Verify matrix is row-stochastic (each row sums to 1).

    Parameters
    ----------
    matrix : np.ndarray
        Matrix to check
    name : str
        Name of the matrix for error messages
    tolerance : float, optional
        Tolerance for row sum check, by default 1e-6

    Raises
    ------
    ValidationError
        If any row does not sum to 1.0 within tolerance

    Examples
    --------
    >>> ensure_stochastic_matrix(np.eye(3), "transition_matrix")  # OK
    >>> bad_matrix = np.array([[0.5, 0.5], [0.6, 0.3]])  # Row 2 sums to 0.9
    >>> ensure_stochastic_matrix(bad_matrix, "transition_matrix")  # Raises
    """
    row_sums = np.sum(matrix, axis=1)
    bad_rows = ~np.isclose(row_sums, 1.0, atol=tolerance)

    if np.any(bad_rows):
        bad_indices = np.where(bad_rows)[0]
        bad_sums = row_sums[bad_rows]
        raise ValidationError(
            f"{name} must be row-stochastic (each row sums to 1)",
            expected="all row sums = 1.0",
            got=f"{len(bad_indices)} row(s) with invalid sums: rows {bad_indices.tolist()}, sums {bad_sums.tolist()}",
            hint=f"Normalize each row: {name} = {name} / {name}.sum(axis=1, keepdims=True)",
            example=f"    # Row 0 sums to {row_sums[0]:.6f}{'  ✓' if not bad_rows[0] else '  ✗'}",
        )


def ensure_ndarray(value: Any, name: str) -> None:
    """Verify value is a numpy array.

    Parameters
    ----------
    value : Any
        Value to check
    name : str
        Name of the parameter for error messages

    Raises
    ------
    ValidationError
        If value is not a numpy array

    Examples
    --------
    >>> ensure_ndarray(np.array([1, 2]), "position")  # OK
    >>> ensure_ndarray([1, 2], "position")  # Raises
    """
    if not isinstance(value, np.ndarray):
        raise ValidationError(
            f"{name} must be a numpy array",
            expected="np.ndarray",
            got=f"{type(value).__name__}",
            hint=f"Convert to numpy array: {name} = np.array({name})",
            example=f"    {name} = np.array([1.0, 2.0, 3.0])",
        )


def ensure_monotonic_increasing(
    arr: np.ndarray, name: str, strict: bool = False
) -> None:
    """Verify array is monotonically increasing.

    Parameters
    ----------
    arr : np.ndarray
        Array to check
    name : str
        Name of the array for error messages
    strict : bool, optional
        If True, require strictly increasing (no equal consecutive values).
        By default False.

    Raises
    ------
    DataError
        If array is not monotonically increasing

    Examples
    --------
    >>> ensure_monotonic_increasing(np.array([1, 2, 3]), "time")  # OK
    >>> ensure_monotonic_increasing(np.array([1, 3, 2]), "time")  # Raises
    """
    diffs = np.diff(arr)

    if strict:
        condition = np.all(diffs > 0)
        expected_str = "strictly increasing (each value > previous)"
    else:
        condition = np.all(diffs >= 0)
        expected_str = "monotonically increasing (each value >= previous)"

    if not condition:
        # Find first violation
        if strict:
            bad_indices = np.where(diffs <= 0)[0]
        else:
            bad_indices = np.where(diffs < 0)[0]

        if len(bad_indices) > 0:
            idx = bad_indices[0]
            raise DataError(
                f"{name} must be {expected_str}",
                data_name=name,
                hint=f"Found violation at index {idx}: {name}[{idx}] = {arr[idx]:.6f}, {name}[{idx + 1}] = {arr[idx + 1]:.6f}. Check for duplicate or out-of-order timestamps.",
            )


def ensure_matching_lengths(arr1, arr2, name1: str, name2: str) -> None:
    """Verify two arrays have matching lengths.

    Parameters
    ----------
    arr1 : array-like
        First array
    arr2 : array-like
        Second array
    name1 : str
        Name of first array for error messages
    name2 : str
        Name of second array for error messages

    Raises
    ------
    ValidationError
        If arrays have different lengths

    Examples
    --------
    >>> ensure_matching_lengths(np.array([1, 2, 3]), np.array([4, 5, 6]), "pos", "time")  # OK
    >>> ensure_matching_lengths(np.array([1, 2]), np.array([4, 5, 6]), "pos", "time")  # Raises
    """
    len1 = len(arr1)
    len2 = len(arr2)

    if len1 != len2:
        raise ValidationError(
            f"Length mismatch between {name1} and {name2}",
            expected=f"{name2} with length {len1}",
            got=f"{name2} with length {len2}",
            hint=f"Ensure {name1} and {name2} have the same number of time points",
            example=f"    # Both should have same length\n    len({name1}) = {len1}\n    len({name2}) = {len2}  # Mismatch!",
        )


def ensure_networkx_graph(value: Any, name: str) -> None:
    """Verify value is a NetworkX Graph.

    Parameters
    ----------
    value : Any
        Value to check
    name : str
        Name of the parameter for error messages

    Raises
    ------
    ValidationError
        If value is not a NetworkX Graph

    Examples
    --------
    >>> import networkx as nx
    >>> ensure_networkx_graph(nx.Graph(), "track_graph")  # OK
    >>> ensure_networkx_graph({"a": 1}, "track_graph")  # Raises
    """
    try:
        import networkx as nx
    except ImportError as err:
        raise ValidationError(
            "NetworkX is required but not installed",
            hint="Install NetworkX: pip install networkx",
        ) from err

    if not isinstance(value, nx.Graph):
        raise ValidationError(
            f"{name} must be a NetworkX Graph",
            expected="networkx.Graph",
            got=f"{type(value).__name__}",
            hint=f"Create a NetworkX graph: {name} = nx.Graph()",
            example=f"    import networkx as nx\n    {name} = nx.Graph()\n    {name}.add_edge(0, 1)",
        )


def ensure_graph_has_nodes(graph: Any, name: str, min_nodes: int = 2) -> None:
    """Verify NetworkX graph has sufficient nodes.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to check
    name : str
        Name of the parameter for error messages
    min_nodes : int, optional
        Minimum number of nodes required, by default 2

    Raises
    ------
    ValidationError
        If graph has too few nodes

    Examples
    --------
    >>> import networkx as nx
    >>> g = nx.Graph()
    >>> g.add_edge(0, 1)
    >>> ensure_graph_has_nodes(g, "track_graph", min_nodes=2)  # OK
    >>> ensure_graph_has_nodes(nx.Graph(), "track_graph")  # Raises
    """
    n_nodes = graph.number_of_nodes()

    if n_nodes < min_nodes:
        raise ValidationError(
            f"{name} has too few nodes",
            expected=f"at least {min_nodes} node(s)",
            got=f"{n_nodes} node(s)",
            hint=f"Add more nodes to {name} using .add_node() or .add_edge()",
            example=f"    {name}.add_edge(0, 1)  # Creates nodes 0 and 1 with an edge",
        )


def ensure_graph_has_edges(graph: Any, name: str, min_edges: int = 1) -> None:
    """Verify NetworkX graph has sufficient edges.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to check
    name : str
        Name of the parameter for error messages
    min_edges : int, optional
        Minimum number of edges required, by default 1

    Raises
    ------
    ValidationError
        If graph has too few edges

    Examples
    --------
    >>> import networkx as nx
    >>> g = nx.Graph()
    >>> g.add_edge(0, 1)
    >>> ensure_graph_has_edges(g, "track_graph")  # OK
    >>> ensure_graph_has_edges(nx.Graph(), "track_graph")  # Raises
    """
    n_edges = graph.number_of_edges()

    if n_edges < min_edges:
        raise ValidationError(
            f"{name} has too few edges",
            expected=f"at least {min_edges} edge(s)",
            got=f"{n_edges} edge(s)",
            hint=f"Add edges to {name} using .add_edge(node1, node2)",
            example=f"    {name}.add_edge(0, 1)  # Creates an edge between nodes 0 and 1",
        )


def ensure_graph_nodes_have_attribute(
    graph: Any, name: str, attribute: str, sample_value_type: type | None = None
) -> None:
    """Verify all nodes in NetworkX graph have required attribute.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to check
    name : str
        Name of the parameter for error messages
    attribute : str
        Required attribute name
    sample_value_type : type, optional
        Expected type of attribute value, by default None (no type checking)

    Raises
    ------
    ValidationError
        If any node is missing the required attribute or has wrong type

    Examples
    --------
    >>> import networkx as nx
    >>> g = nx.Graph()
    >>> g.add_node(0, pos=(0, 0))
    >>> g.add_node(1, pos=(1, 1))
    >>> ensure_graph_nodes_have_attribute(g, "track_graph", "pos")  # OK
    >>> g.add_node(2)  # Missing pos
    >>> ensure_graph_nodes_have_attribute(g, "track_graph", "pos")  # Raises
    """
    nodes_without_attribute = []
    nodes_with_wrong_type = []

    for node in graph.nodes():
        if attribute not in graph.nodes[node]:
            nodes_without_attribute.append(node)
        elif sample_value_type is not None:
            value = graph.nodes[node][attribute]
            if not isinstance(value, sample_value_type):
                nodes_with_wrong_type.append((node, type(value).__name__))

    if nodes_without_attribute:
        n_missing = len(nodes_without_attribute)
        sample_nodes = nodes_without_attribute[:3]
        sample_str = ", ".join(str(n) for n in sample_nodes)
        if n_missing > 3:
            sample_str += f", ... ({n_missing - 3} more)"

        raise ValidationError(
            f"{name} nodes missing required attribute '{attribute}'",
            expected=f"all {graph.number_of_nodes()} node(s) to have '{attribute}' attribute",
            got=f"{n_missing} node(s) missing '{attribute}': {sample_str}",
            hint=f"Add '{attribute}' to all nodes: {name}.nodes[node]['{attribute}'] = value",
            example=f"    {name}.add_node(0, {attribute}=(0, 0))\n    # or\n    {name}.nodes[0]['{attribute}'] = (0, 0)",
        )

    if nodes_with_wrong_type:
        sample_node, sample_type = nodes_with_wrong_type[0]

        raise ValidationError(
            f"{name} node attribute '{attribute}' has wrong type",
            expected=f"attribute type: {sample_value_type.__name__}",
            got=f"node {sample_node} has type {sample_type}",
            hint=f"Ensure all '{attribute}' values are {sample_value_type.__name__}",
            example=f"    {name}.nodes[{sample_node}]['{attribute}'] = {sample_value_type.__name__}(...)",
        )


def ensure_graph_edges_have_attribute(
    graph: Any, name: str, attribute: str, sample_value_type: type | None = None
) -> None:
    """Verify all edges in NetworkX graph have required attribute.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to check
    name : str
        Name of the parameter for error messages
    attribute : str
        Required attribute name
    sample_value_type : type, optional
        Expected type of attribute value, by default None (no type checking)

    Raises
    ------
    ValidationError
        If any edge is missing the required attribute or has wrong type

    Examples
    --------
    >>> import networkx as nx
    >>> g = nx.Graph()
    >>> g.add_edge(0, 1, distance=10.0)
    >>> ensure_graph_edges_have_attribute(g, "track_graph", "distance")  # OK
    >>> g.add_edge(1, 2)  # Missing distance
    >>> ensure_graph_edges_have_attribute(g, "track_graph", "distance")  # Raises
    """
    edges_without_attribute = []
    edges_with_wrong_type = []

    for edge in graph.edges():
        if attribute not in graph.edges[edge]:
            edges_without_attribute.append(edge)
        elif sample_value_type is not None:
            value = graph.edges[edge][attribute]
            if not isinstance(value, sample_value_type):
                edges_with_wrong_type.append((edge, type(value).__name__))

    if edges_without_attribute:
        n_missing = len(edges_without_attribute)
        sample_edges = edges_without_attribute[:3]
        sample_str = ", ".join(str(e) for e in sample_edges)
        if n_missing > 3:
            sample_str += f", ... ({n_missing - 3} more)"

        raise ValidationError(
            f"{name} edges missing required attribute '{attribute}'",
            expected=f"all {graph.number_of_edges()} edge(s) to have '{attribute}' attribute",
            got=f"{n_missing} edge(s) missing '{attribute}': {sample_str}",
            hint=f"Add '{attribute}' to all edges: {name}.edges[edge]['{attribute}'] = value",
            example=f"    {name}.add_edge(0, 1, {attribute}=10.0)\n    # or\n    {name}.edges[(0, 1)]['{attribute}'] = 10.0",
        )

    if edges_with_wrong_type:
        sample_edge, sample_type = edges_with_wrong_type[0]

        raise ValidationError(
            f"{name} edge attribute '{attribute}' has wrong type",
            expected=f"attribute type: {sample_value_type.__name__}",
            got=f"edge {sample_edge} has type {sample_type}",
            hint=f"Ensure all '{attribute}' values are {sample_value_type.__name__}",
            example=f"    {name}.edges[{sample_edge}]['{attribute}'] = {sample_value_type.__name__}(...)",
        )
