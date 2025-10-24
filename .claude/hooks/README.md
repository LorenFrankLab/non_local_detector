# Claude Code Hooks for Safe Scientific Development

This directory contains hooks that provide technical enforcement of safety requirements for the `non_local_detector` project.

## Hook Overview

Hooks automatically intercept commands and verify safety requirements before execution. They provide the first layer of defense in the three-layer system (hooks + skills + documentation).

## Available Hooks

### pre-tool-use.sh
**Triggers:** Before every Bash tool execution

**Enforces:**
1. **Conda environment validation**
   - Warns if Python commands run outside `non_local_detector` environment
   - Suggests activation command
   - Auto-prepends activation to commands

2. **Test-before-commit**
   - Checks if tests have run recently before allowing commits
   - Warns if no test cache found
   - Recommends running test suite

3. **Snapshot update protection**
   - Blocks `--snapshot-update` without approval flag
   - Requires full analysis before approval
   - Clears approval flag after use

**Exit codes:**
- `0`: Command allowed (may include warnings)
- `1`: Command blocked (e.g., snapshot update without approval)

---

### user-prompt-submit.sh
**Triggers:** After user submits a prompt

**Detects:**
- Snapshot file changes (`.ambr` files)
- Recent snapshot test runs

**Actions:**
- Reports modified snapshot files
- Reminds Claude to provide full analysis
- Lists requirements for snapshot update approval

**Exit codes:**
- Always `0` (informational only, never blocks)

---

## Utility Libraries

### lib/env_check.sh
**Functions:**
- `check_conda_env()`: Verify conda environment active
- `get_activation_cmd()`: Return activation command string
- `prepend_activation()`: Auto-prepend activation to command
- `needs_conda()`: Check if command requires conda environment

### lib/numerical_validation.sh
**Functions:**
- `check_invariants()`: Scan test output for NaN/Inf
- `has_snapshot_changes()`: Detect modified snapshot files
- `has_snapshot_approval()`: Check if approval flag set
- `set_snapshot_approval()`: Create approval flag file
- `clear_snapshot_approval()`: Remove approval flag file

---

## Hook Environment Variables

Hooks receive these environment variables from Claude Code:

- `CLAUDE_TOOL_NAME`: Name of tool being invoked (e.g., "Bash")
- `CLAUDE_TOOL_COMMAND`: Full command string being executed
- `CONDA_DEFAULT_ENV`: Current conda environment name

---

## Testing Hooks

### Manual testing

```bash
# Test environment check
export CLAUDE_TOOL_NAME="Bash"
export CLAUDE_TOOL_COMMAND="pytest --version"
.claude/hooks/pre-tool-use.sh
echo "Exit code: $?"

# Test snapshot protection (should block)
export CLAUDE_TOOL_COMMAND="pytest --snapshot-update"
.claude/hooks/pre-tool-use.sh
echo "Exit code: $?"  # Should be 1

# Test with approval
source .claude/hooks/lib/numerical_validation.sh
set_snapshot_approval
.claude/hooks/pre-tool-use.sh
echo "Exit code: $?"  # Should be 0
```

### Expected behavior

| Scenario | Hook | Expected Result |
|----------|------|-----------------|
| Python command, wrong env | pre-tool-use | Warning (exit 0) |
| Commit without recent tests | pre-tool-use | Warning (exit 0) |
| Snapshot update, no approval | pre-tool-use | Blocked (exit 1) |
| Snapshot update, with approval | pre-tool-use | Allowed, flag cleared (exit 0) |
| Snapshot files modified | user-prompt-submit | Info message (exit 0) |

---

## Integration with Skills

Hooks work together with skills:

1. **Skills** (layer 2) guide Claude through workflows
2. **Hooks** (layer 1) enforce critical requirements
3. **Documentation** (layer 3) explains the "why"

Example flow:
```
Claude uses scientific-tdd skill
  ↓
Skill says: "Run tests"
  ↓
Claude executes: pytest command
  ↓
pre-tool-use hook checks conda env
  ↓
Hook passes or warns
  ↓
Command executes
```

---

## Debugging Hooks

If hooks misbehave:

1. **Check hook permissions:**
   ```bash
   ls -l .claude/hooks/*.sh
   # Should show -rwxr-xr-x
   ```

2. **Test utilities directly:**
   ```bash
   source .claude/hooks/lib/env_check.sh
   check_conda_env && echo "OK" || echo "FAIL"
   ```

3. **Run hook with debug output:**
   ```bash
   bash -x .claude/hooks/pre-tool-use.sh
   ```

4. **Check environment variables:**
   ```bash
   echo "Tool: $CLAUDE_TOOL_NAME"
   echo "Command: $CLAUDE_TOOL_COMMAND"
   echo "Conda: $CONDA_DEFAULT_ENV"
   ```

---

## Maintenance

### When to update hooks:

- Conda environment name changes
- New tools need environment checking
- Validation requirements change
- New safety checks needed

### Best practices:

- Keep hooks fast (< 100ms)
- Always exit with appropriate code (0 or 1)
- Provide helpful error messages
- Source utilities from `lib/` directory
- Test thoroughly before deployment

---

## Related Documentation

- **CLAUDE.md**: Project operational rules
- **Skills**: Workflow enforcement
- **Testing Plan**: Coverage improvement strategy
