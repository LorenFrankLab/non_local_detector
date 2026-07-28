#!/bin/bash
# Test script for the safe scientific development system
# Tests skills and documentation integration

set -e

echo "🧪 Testing Safe Scientific Development System"
echo "=============================================="
echo ""

# Test 1: Skills are present and named so they load on case-sensitive filesystems
echo "Test 1: Skills..."
for skill in scientific-tdd numerical-validation safe-refactoring; do
    if [ -f ".claude/skills/$skill/SKILL.md" ]; then
        echo "  ✓ $skill skill exists"
    else
        echo "  ✗ $skill skill missing (expected .claude/skills/$skill/SKILL.md)"
        exit 1
    fi
done

# Test 2: Skills invoke the project's package manager, not a hardcoded interpreter
echo ""
echo "Test 2: Skill commands use uv..."
if grep -rq 'miniconda3\|/bin/pytest' .claude/skills/; then
    echo "  ✗ Hardcoded interpreter paths found in skills"
    grep -rn 'miniconda3\|/bin/pytest' .claude/skills/
    exit 1
else
    echo "  ✓ No hardcoded interpreter paths"
fi

# Test 3: CLAUDE.md contains critical sections
echo ""
echo "Test 3: CLAUDE.md documentation..."
for section in "CRITICAL: Claude Code Operational Rules" \
               "Numerical Accuracy Standards" \
               "Mandatory Skills Usage"; do
    if grep -q "$section" CLAUDE.md; then
        echo "  ✓ CLAUDE.md has \"$section\""
    else
        echo "  ✗ CLAUDE.md missing \"$section\""
        exit 1
    fi
done

# Test 4: CLAUDE.md pointers into skills resolve
echo ""
echo "Test 4: CLAUDE.md skill references resolve..."
missing=0
while read -r ref; do
    if [ ! -f "$ref" ]; then
        echo "  ✗ CLAUDE.md references missing file: $ref"
        missing=1
    fi
done < <(grep -o '\.claude/skills/[A-Za-z0-9_-]*/SKILL\.md' CLAUDE.md | sort -u)
if [ "$missing" -eq 0 ]; then
    echo "  ✓ All referenced skill files exist"
else
    exit 1
fi

# Test 5: Git configuration
echo ""
echo "Test 5: Git configuration..."
if grep -q "^\.worktrees/$" .gitignore; then
    echo "  ✓ .worktrees/ in .gitignore"
else
    echo "  ✗ .worktrees/ not in .gitignore"
    exit 1
fi

# Test 6: Documentation completeness
echo ""
echo "Test 6: Documentation completeness..."
if [ -f .claude/skills/README.md ]; then
    echo "  ✓ Skills README exists"
else
    echo "  ✗ Skills README missing"
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
echo "✅ All checks passed"
