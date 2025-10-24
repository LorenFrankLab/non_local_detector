#!/bin/bash
# Numerical validation utilities for scientific code

# Check if mathematical invariants hold
check_invariants() {
    local test_output="$1"

    # Look for common numerical issues in test output
    if echo "$test_output" | grep -q "NaN\|Inf\|-Inf"; then
        echo "❌ Numerical issue detected: NaN or Inf in outputs"
        return 1
    fi

    return 0
}

# Detect snapshot changes
has_snapshot_changes() {
    # Check if any snapshot files have been modified
    if git status --porcelain | grep -q "\.ambr$"; then
        return 0
    fi

    # Check syrupy snapshot directory
    if [ -d ".pytest_cache/v/cache/snapshot" ]; then
        if [ -n "$(find .pytest_cache/v/cache/snapshot -mmin -5 2>/dev/null)" ]; then
            return 0
        fi
    fi

    return 1
}

# Check if approval flag is set
has_snapshot_approval() {
    [ -f ".claude/snapshot_update_approved" ]
}

# Set approval flag
set_snapshot_approval() {
    mkdir -p .claude
    touch .claude/snapshot_update_approved
    echo "Snapshot update approved at $(date)" > .claude/snapshot_update_approved
}

# Clear approval flag
clear_snapshot_approval() {
    rm -f .claude/snapshot_update_approved
}
