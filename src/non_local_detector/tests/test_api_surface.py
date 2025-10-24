"""API surface monitoring tests.

These tests ensure backward compatibility by tracking all public APIs.
Any change to function signatures, classes, or exports triggers a failure
with a detailed report of what changed.
"""

import inspect
import json
from pathlib import Path
from typing import Any

import pytest

import non_local_detector as nld

# Path to API snapshots
API_SNAPSHOT_DIR = Path(__file__).parent / "api_snapshots"
API_SNAPSHOT_FILE = API_SNAPSHOT_DIR / "public_api.json"


def extract_api_surface(module) -> dict[str, Any]:
    """Extract all public classes, functions, and their signatures.

    Parameters
    ----------
    module : module
        Python module to extract API from

    Returns
    -------
    dict
        Dictionary containing classes, functions, and constants
    """
    api = {"classes": {}, "functions": {}, "constants": {}}

    for name in dir(module):
        if name.startswith("_"):
            continue

        obj = getattr(module, name)

        if inspect.isclass(obj):
            # Capture class and its public methods
            methods = {}
            for method_name in dir(obj):
                if method_name.startswith("_") and method_name != "__init__":
                    continue
                method_obj = getattr(obj, method_name)
                if callable(method_obj):
                    try:
                        sig = str(inspect.signature(method_obj))
                        methods[method_name] = sig
                    except (ValueError, TypeError):
                        # Some built-in methods don't have signatures
                        methods[method_name] = "<builtin>"

            api["classes"][name] = {
                "module": obj.__module__,
                "methods": methods,
            }

        elif inspect.isfunction(obj):
            try:
                api["functions"][name] = {
                    "signature": str(inspect.signature(obj)),
                    "module": obj.__module__,
                }
            except (ValueError, TypeError):
                api["functions"][name] = {
                    "signature": "<builtin>",
                    "module": getattr(obj, "__module__", "unknown"),
                }
        else:
            # Constants and other exports
            api["constants"][name] = type(obj).__name__

    return api


def save_api_snapshot(api_surface: dict[str, Any]) -> None:
    """Save API surface snapshot to disk.

    Parameters
    ----------
    api_surface : dict
        API surface dictionary from extract_api_surface
    """
    API_SNAPSHOT_DIR.mkdir(exist_ok=True)
    with open(API_SNAPSHOT_FILE, "w") as f:
        json.dump(api_surface, f, indent=2, sort_keys=True)


def load_api_snapshot() -> dict[str, Any]:
    """Load API surface snapshot from disk.

    Returns
    -------
    dict
        Previously saved API surface
    """
    with open(API_SNAPSHOT_FILE) as f:
        return json.load(f)


def compare_api_surfaces(
    baseline: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    """Compare two API surfaces and identify changes.

    Parameters
    ----------
    baseline : dict
        Baseline API surface
    current : dict
        Current API surface

    Returns
    -------
    dict
        Dictionary with keys: breaking_changes, additions, removals, modifications
    """
    diff = {
        "breaking_changes": [],
        "additions": [],
        "removals": [],
        "modifications": [],
    }

    # Check classes
    baseline_classes = set(baseline.get("classes", {}).keys())
    current_classes = set(current.get("classes", {}).keys())

    # Removed classes are breaking
    for removed in baseline_classes - current_classes:
        diff["breaking_changes"].append(f"Class removed: {removed}")
        diff["removals"].append(f"class {removed}")

    # Added classes are fine
    for added in current_classes - baseline_classes:
        diff["additions"].append(f"class {added}")

    # Check for changed class methods
    for cls_name in baseline_classes & current_classes:
        baseline_methods = baseline["classes"][cls_name].get("methods", {})
        current_methods = current["classes"][cls_name].get("methods", {})

        baseline_method_names = set(baseline_methods.keys())
        current_method_names = set(current_methods.keys())

        # Removed methods are breaking
        for removed_method in baseline_method_names - current_method_names:
            diff["breaking_changes"].append(
                f"Method removed: {cls_name}.{removed_method}"
            )

        # Check signature changes
        for method_name in baseline_method_names & current_method_names:
            if baseline_methods[method_name] != current_methods[method_name]:
                diff["breaking_changes"].append(
                    f"Signature changed: {cls_name}.{method_name}\n"
                    f"  Old: {baseline_methods[method_name]}\n"
                    f"  New: {current_methods[method_name]}"
                )

    # Check functions
    baseline_functions = set(baseline.get("functions", {}).keys())
    current_functions = set(current.get("functions", {}).keys())

    # Removed functions are breaking
    for removed in baseline_functions - current_functions:
        diff["breaking_changes"].append(f"Function removed: {removed}")
        diff["removals"].append(f"function {removed}")

    # Added functions are fine
    for added in current_functions - baseline_functions:
        diff["additions"].append(f"function {added}")

    # Check signature changes
    for func_name in baseline_functions & current_functions:
        baseline_sig = baseline["functions"][func_name].get("signature")
        current_sig = current["functions"][func_name].get("signature")
        if baseline_sig != current_sig:
            diff["breaking_changes"].append(
                f"Signature changed: {func_name}\n"
                f"  Old: {baseline_sig}\n"
                f"  New: {current_sig}"
            )

    # Check constants
    baseline_constants = set(baseline.get("constants", {}).keys())
    current_constants = set(current.get("constants", {}).keys())

    for removed in baseline_constants - current_constants:
        diff["removals"].append(f"constant {removed}")

    for added in current_constants - baseline_constants:
        diff["additions"].append(f"constant {added}")

    return diff


def format_api_diff_report(diff: dict[str, Any]) -> str:
    """Format API diff into human-readable report.

    Parameters
    ----------
    diff : dict
        Diff dictionary from compare_api_surfaces

    Returns
    -------
    str
        Formatted report
    """
    lines = []

    if diff["breaking_changes"]:
        lines.append("\n❌ BREAKING CHANGES:")
        for change in diff["breaking_changes"]:
            lines.append(f"  - {change}")

    if diff["additions"]:
        lines.append("\n➕ ADDITIONS (non-breaking):")
        for addition in diff["additions"]:
            lines.append(f"  - {addition}")

    if diff["removals"]:
        lines.append("\n🗑️  REMOVALS:")
        for removal in diff["removals"]:
            lines.append(f"  - {removal}")

    return "\n".join(lines) if lines else "No API changes detected"


@pytest.mark.unit
def test_public_api_compatibility():
    """Ensure public API hasn't changed without approval.

    This test fails if:
    - Public classes are removed
    - Public functions are removed
    - Function/method signatures change
    - Public class methods are removed

    To update the baseline after intentional API changes:
    1. Review the diff carefully
    2. Approve the changes
    3. Delete api_snapshots/public_api.json
    4. Re-run this test to create new baseline
    """
    current_api = extract_api_surface(nld)

    if not API_SNAPSHOT_FILE.exists():
        # First run: save baseline
        save_api_snapshot(current_api)
        pytest.skip("API baseline created, skipping comparison")

    # Load baseline and compare
    baseline_api = load_api_snapshot()
    diff = compare_api_surfaces(baseline_api, current_api)

    # Fail if any changes detected
    has_changes = diff["breaking_changes"] or diff["additions"] or diff["removals"]

    if has_changes:
        report = format_api_diff_report(diff)
        pytest.fail(
            f"API surface changed:\n{report}\n\n"
            f"If this change is intentional:\n"
            f"1. Review the changes carefully\n"
            f"2. Delete {API_SNAPSHOT_FILE}\n"
            f"3. Re-run test to create new baseline"
        )
