#!/bin/bash
# User-prompt-submit hook for Claude Code
# Runs after user submits a prompt

# Source utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/numerical_validation.sh"

# Check for snapshot changes after test runs
if has_snapshot_changes; then
    echo ""
    echo "📸 Snapshot changes detected!"
    echo ""
    echo "Modified snapshot files:"
    git status --porcelain | grep "\.ambr$" | sed 's/^/  /'
    echo ""
    echo "Before updating snapshots, Claude must provide:"
    echo "  1. Diff showing what changed"
    echo "  2. Explanation of why it changed"
    echo "  3. Validation that mathematical properties hold"
    echo "  4. Test case demonstrating correctness"
    echo ""
    echo "User must approve before --snapshot-update is allowed."
    echo ""
fi

exit 0
