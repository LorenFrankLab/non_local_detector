#!/bin/bash
# Integration test for safe scientific development system
# Tests hooks, skills, and documentation integration

set -e  # Exit on error

echo "🧪 Testing Safe Scientific Development System"
echo "=============================================="
echo ""

# Test 1: Hook utilities exist and are executable
echo "Test 1: Hook utilities..."
if [ -x .claude/hooks/lib/env_check.sh ]; then
    echo "  ✓ env_check.sh is executable"
else
    echo "  ✗ env_check.sh not executable"
    exit 1
fi

if [ -x .claude/hooks/lib/numerical_validation.sh ]; then
    echo "  ✓ numerical_validation.sh is executable"
else
    echo "  ✗ numerical_validation.sh not executable"
    exit 1
fi

# Test 2: Hooks exist and are executable
echo ""
echo "Test 2: Hooks..."
if [ -x .claude/hooks/pre-tool-use.sh ]; then
    echo "  ✓ pre-tool-use.sh is executable"
else
    echo "  ✗ pre-tool-use.sh not executable"
    exit 1
fi

if [ -x .claude/hooks/user-prompt-submit.sh ]; then
    echo "  ✓ user-prompt-submit.sh is executable"
else
    echo "  ✗ user-prompt-submit.sh not executable"
    exit 1
fi

# Test 3: Skills exist
echo ""
echo "Test 3: Skills..."
if [ -f .claude/skills/scientific-tdd/skill.md ]; then
    echo "  ✓ scientific-tdd skill exists"
else
    echo "  ✗ scientific-tdd skill missing"
    exit 1
fi

if [ -f .claude/skills/numerical-validation/skill.md ]; then
    echo "  ✓ numerical-validation skill exists"
else
    echo "  ✗ numerical-validation skill missing"
    exit 1
fi

if [ -f .claude/skills/safe-refactoring/skill.md ]; then
    echo "  ✓ safe-refactoring skill exists"
else
    echo "  ✗ safe-refactoring skill missing"
    exit 1
fi

# Test 4: CLAUDE.md contains critical sections
echo ""
echo "Test 4: CLAUDE.md documentation..."
if grep -q "CRITICAL: Claude Code Operational Rules" CLAUDE.md; then
    echo "  ✓ Critical operational rules present"
else
    echo "  ✗ Critical operational rules missing"
    exit 1
fi

if grep -q "Numerical Accuracy Standards" CLAUDE.md; then
    echo "  ✓ Numerical accuracy standards present"
else
    echo "  ✗ Numerical accuracy standards missing"
    exit 1
fi

if grep -q "Workflow Selection Guide" CLAUDE.md; then
    echo "  ✓ Workflow selection guide present"
else
    echo "  ✗ Workflow selection guide missing"
    exit 1
fi

# Test 5: Hook functional test - environment check
echo ""
echo "Test 5: Hook functional tests..."
source .claude/hooks/lib/env_check.sh

if needs_conda "pytest"; then
    echo "  ✓ Conda detection works for pytest"
else
    echo "  ✗ Conda detection failed for pytest"
    exit 1
fi

if ! needs_conda "ls"; then
    echo "  ✓ Conda detection correctly ignores non-Python commands"
else
    echo "  ✗ Conda detection incorrectly flagged ls"
    exit 1
fi

# Test 6: Hook functional test - snapshot protection
echo ""
echo "Test 6: Snapshot protection..."
source .claude/hooks/lib/numerical_validation.sh

# Should not have approval initially
if ! has_snapshot_approval; then
    echo "  ✓ No approval flag initially"
else
    echo "  ✗ Approval flag should not exist initially"
    clear_snapshot_approval
    exit 1
fi

# Set approval
set_snapshot_approval
if has_snapshot_approval; then
    echo "  ✓ Approval flag set successfully"
else
    echo "  ✗ Failed to set approval flag"
    exit 1
fi

# Clear approval
clear_snapshot_approval
if ! has_snapshot_approval; then
    echo "  ✓ Approval flag cleared successfully"
else
    echo "  ✗ Failed to clear approval flag"
    exit 1
fi

# Test 7: .gitignore contains .worktrees
echo ""
echo "Test 7: Git configuration..."
if grep -q "^\.worktrees/$" .gitignore; then
    echo "  ✓ .worktrees/ in .gitignore"
else
    echo "  ✗ .worktrees/ not in .gitignore"
    exit 1
fi

# Test 8: README files exist
echo ""
echo "Test 8: Documentation completeness..."
if [ -f .claude/skills/README.md ]; then
    echo "  ✓ Skills README exists"
else
    echo "  ✗ Skills README missing"
    exit 1
fi

if [ -f .claude/hooks/README.md ]; then
    echo "  ✓ Hooks README exists"
else
    echo "  ✗ Hooks README missing"
    exit 1
fi

if [ -f docs/plans/2025-10-23-safe-scientific-development.md ]; then
    echo "  ✓ Implementation plan exists"
else
    echo "  ✗ Implementation plan missing"
    exit 1
fi

# All tests passed
echo ""
echo "=============================================="
echo "✅ All tests passed!"
echo ""
echo "Safe scientific development system is properly configured."
echo ""
echo "Summary:"
echo "  - Hooks: Installed and functional"
echo "  - Skills: All 3 skills present"
echo "  - Documentation: Enhanced CLAUDE.md + READMEs"
echo "  - Git: .worktrees/ properly ignored"
echo ""
