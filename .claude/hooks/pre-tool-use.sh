#!/bin/bash
# Pre-tool-use hook for Claude Code
# Enforces conda environment and test-before-commit requirements

# Source utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/env_check.sh"
source "$SCRIPT_DIR/lib/numerical_validation.sh"

# Get the tool name and command from arguments
TOOL_NAME="${CLAUDE_TOOL_NAME:-unknown}"
TOOL_COMMAND="${CLAUDE_TOOL_COMMAND:-}"

# Only process Bash tool invocations
if [ "$TOOL_NAME" != "Bash" ]; then
    exit 0
fi

# Extract base command (first word)
BASE_CMD=$(echo "$TOOL_COMMAND" | awk '{print $1}')

# Check 1: Environment enforcement for Python commands
if needs_conda "$BASE_CMD"; then
    if ! check_conda_env; then
        echo "❌ Wrong conda environment detected!"
        echo "Required: non_local_detector"
        echo "Current: ${CONDA_DEFAULT_ENV:-none}"
        echo ""
        echo "Run: $(get_activation_cmd)"
        echo ""
        echo "Or commands will be auto-prepended with activation."

        # Don't block - just warn (auto-prepend will handle it)
        exit 0
    fi
fi

# Check 2: Test-before-commit enforcement
if [[ "$TOOL_COMMAND" =~ ^git[[:space:]]+commit ]]; then
    # Check if tests have run recently (within last 5 minutes)
    if [ -d ".pytest_cache" ]; then
        CACHE_AGE=$(find .pytest_cache -name "*.pytest_cache" -mmin -5 2>/dev/null | wc -l)
        if [ "$CACHE_AGE" -eq 0 ]; then
            echo "⚠️  No recent test runs detected"
            echo "Recommendation: Run tests before committing"
            echo ""
            echo "Run: /Users/edeno/miniconda3/envs/non_local_detector/bin/pytest -v"
            echo ""
            echo "Proceeding with commit anyway (warning only)..."
        fi
    fi
fi

# Check 3: Snapshot update protection
if [[ "$TOOL_COMMAND" =~ --snapshot-update ]]; then
    if ! has_snapshot_approval; then
        echo "❌ Snapshot update attempted without approval!"
        echo ""
        echo "Snapshot updates require explicit approval."
        echo "Claude must provide full analysis first:"
        echo "  1. Diff: What changed in snapshots"
        echo "  2. Explanation: Why the change occurred"
        echo "  3. Validation: Mathematical properties still hold"
        echo "  4. Test case: Demonstrate correctness"
        echo ""
        echo "After approval, user must set approval flag."

        # Block this command
        exit 1
    fi

    # Clear approval flag after use
    clear_snapshot_approval
fi

# All checks passed
exit 0
