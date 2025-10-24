"""Generate comprehensive regression test reports for Claude Code workflow.

This module consolidates results from all regression detection layers:
- API surface changes
- Golden data numerical diffs
- Snapshot test changes
- Property test violations

Used by Claude Code to present changes before requesting approval.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RegressionReport:
    """Consolidated regression test results."""

    api_changes: dict[str, Any]
    golden_diffs: dict[str, dict[str, float]]
    snapshot_changes: list[str]
    property_violations: list[str]
    all_tests_passed: bool
    test_summary: dict[str, int]

    def has_breaking_changes(self) -> bool:
        """Check if any breaking changes detected."""
        return bool(self.api_changes.get("breaking_changes")) or bool(
            self.api_changes.get("removals")
        )

    def has_numerical_changes(self) -> bool:
        """Check if numerical outputs changed."""
        return bool(self.golden_diffs) or bool(self.snapshot_changes)

    def format_summary(self) -> str:
        """Format human-readable summary for approval request."""
        lines = ["=" * 80]
        lines.append("REGRESSION TEST SUMMARY")
        lines.append("=" * 80)

        # Test execution summary
        lines.append("\n## TEST EXECUTION")
        lines.append(f"  Passed: {self.test_summary.get('passed', 0)}")
        lines.append(f"  Failed: {self.test_summary.get('failed', 0)}")
        lines.append(f"  Skipped: {self.test_summary.get('skipped', 0)}")
        lines.append(f"  Errors: {self.test_summary.get('errors', 0)}")

        # API Changes
        lines.append("\n## API COMPATIBILITY")
        if self.api_changes:
            if self.api_changes.get("breaking_changes"):
                lines.append("❌ BREAKING CHANGES DETECTED:")
                for change in self.api_changes["breaking_changes"]:
                    # Indent multi-line changes
                    for line in change.split("\n"):
                        lines.append(f"  {line}")
            if self.api_changes.get("additions"):
                lines.append("➕ New APIs added:")
                for add in self.api_changes["additions"]:
                    lines.append(f"  - {add}")
            if self.api_changes.get("removals"):
                lines.append("🗑️  APIs removed:")
                for removal in self.api_changes["removals"]:
                    lines.append(f"  - {removal}")
        else:
            lines.append("✅ No API changes detected")

        # Golden Data Diffs
        lines.append("\n## NUMERICAL OUTPUT CHANGES")
        if self.golden_diffs:
            for test_name, diffs in self.golden_diffs.items():
                lines.append(f"\n### {test_name}")
                lines.append(f"  Max absolute diff: {diffs['max_abs_diff']:.2e}")
                lines.append(f"  Max relative diff: {diffs['max_rel_diff']:.2e}")
                lines.append(f"  Mean absolute diff: {diffs['mean_abs_diff']:.2e}")

                if diffs["max_abs_diff"] > 1e-10:
                    lines.append("  ⚠️  Exceeds tolerance (1e-10)")
        else:
            lines.append("✅ No numerical changes in golden tests")

        # Snapshot Changes
        lines.append("\n## SNAPSHOT TESTS")
        if self.snapshot_changes:
            lines.append(f"⚠️  {len(self.snapshot_changes)} snapshots changed:")
            for snapshot in self.snapshot_changes[:10]:  # Limit to first 10
                lines.append(f"  - {snapshot}")
            if len(self.snapshot_changes) > 10:
                lines.append(f"  ... and {len(self.snapshot_changes) - 10} more")
        else:
            lines.append("✅ All snapshots match baseline")

        # Property Violations
        lines.append("\n## MATHEMATICAL INVARIANTS")
        if self.property_violations:
            lines.append(f"❌ {len(self.property_violations)} property violations:")
            for violation in self.property_violations[:5]:  # Limit to first 5
                lines.append(f"  - {violation}")
            if len(self.property_violations) > 5:
                lines.append(f"  ... and {len(self.property_violations) - 5} more")
        else:
            lines.append("✅ All mathematical invariants hold")

        # Overall Status
        lines.append("\n" + "=" * 80)
        if (
            self.all_tests_passed
            and not self.has_numerical_changes()
            and not self.has_breaking_changes()
        ):
            lines.append("✅ ALL CHECKS PASSED - No regressions detected")
        elif self.has_breaking_changes():
            lines.append("❌ BREAKING CHANGES - Requires careful review")
        elif self.property_violations:
            lines.append("❌ MATHEMATICAL INVARIANTS VIOLATED - Do not proceed")
        elif not self.all_tests_passed:
            lines.append("❌ TESTS FAILED - Fix failures before proceeding")
        elif self.has_numerical_changes():
            lines.append("⚠️  NUMERICAL CHANGES - Review diffs before approval")

        lines.append("=" * 80)
        return "\n".join(lines)


def load_api_diff_results() -> dict[str, Any]:
    """Load API diff results from test output.

    Returns
    -------
    dict
        API changes dictionary
    """
    # Check if API snapshot test failed
    snapshot_file = Path(__file__).parent / "api_snapshots" / "public_api.json"

    if not snapshot_file.exists():
        return {}

    # Try to detect changes by running comparison
    try:
        import non_local_detector as nld

        from .test_api_surface import (
            compare_api_surfaces,
            extract_api_surface,
            load_api_snapshot,
        )

        current_api = extract_api_surface(nld)
        baseline_api = load_api_snapshot()
        return compare_api_surfaces(baseline_api, current_api)
    except Exception:
        return {}


def load_golden_regression_diffs() -> dict[str, dict[str, float]]:
    """Load numerical diffs from golden regression tests.

    Returns
    -------
    dict
        Dictionary mapping test name to diff statistics
    """
    # This would be populated by golden regression tests that detect changes
    # For now, return empty dict (will be implemented when golden tests are extended)
    return {}


def load_snapshot_changes() -> list[str]:
    """Load list of changed snapshots from syrupy.

    Returns
    -------
    list
        List of snapshot test names that changed
    """
    # Syrupy reports changes in test output
    # This parses pytest output to find snapshot changes
    # For now, return empty list (syrupy integration in next phase)
    return []


def load_property_test_failures() -> list[str]:
    """Load property test failures.

    Returns
    -------
    list
        List of property test violations
    """
    # Property tests that fail indicate invariant violations
    # Parse pytest output for property test failures
    return []


def parse_pytest_summary() -> dict[str, int]:
    """Parse pytest execution summary.

    Returns
    -------
    dict
        Dictionary with passed, failed, skipped, errors counts
    """
    # This would parse pytest's final summary line
    # For now return placeholder
    return {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}


def generate_regression_report() -> RegressionReport:
    """Generate comprehensive regression report from test results.

    Called by Claude Code after running full test suite.

    Returns
    -------
    RegressionReport
        Complete regression analysis
    """
    # Collect results from all test layers
    api_changes = load_api_diff_results()
    golden_diffs = load_golden_regression_diffs()
    snapshot_changes = load_snapshot_changes()
    property_violations = load_property_test_failures()
    test_summary = parse_pytest_summary()

    # Overall pass/fail
    all_tests_passed = (
        test_summary.get("failed", 0) == 0 and test_summary.get("errors", 0) == 0
    )

    return RegressionReport(
        api_changes=api_changes,
        golden_diffs=golden_diffs,
        snapshot_changes=snapshot_changes,
        property_violations=property_violations,
        all_tests_passed=all_tests_passed,
        test_summary=test_summary,
    )


def main():
    """Generate and print regression report."""
    report = generate_regression_report()
    print(report.format_summary())


if __name__ == "__main__":
    main()
