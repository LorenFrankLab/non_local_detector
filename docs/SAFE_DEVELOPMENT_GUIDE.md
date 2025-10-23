# Safe Scientific Development with Claude Code - User Guide

This guide explains how to work with Claude Code on the `non_local_detector` project with confidence that changes won't introduce regressions or break numerical accuracy.

## Quick Start

### For Users

When you ask Claude to make changes, the system automatically:

1. **Enforces correct conda environment** (hooks check this)
2. **Follows appropriate workflow** (skills guide Claude)
3. **Validates numerical accuracy** (for mathematical code)
4. **Requires your approval** for critical actions

You don't need to remember all the rules - the system guides both you and Claude.

### For Claude

Read CLAUDE.md at session start. The system will:
- Tell you which skill to use
- Enforce environment requirements via hooks
- Block dangerous operations
- Guide you through validation steps

---

## The Three-Layer System

### Layer 1: Hooks (Automatic Enforcement)

**What they do:**
- Check conda environment before Python commands
- Warn before commits without tests
- Block snapshot updates without approval

**You'll see:**
- Warnings like "⚠️ Wrong conda environment"
- Errors like "❌ Snapshot update requires approval"

**User action:** Usually none - hooks guide Claude automatically

---

### Layer 2: Skills (Workflow Guidance)

**What they do:**
- Guide Claude through complex workflows step-by-step
- Create todo lists for tracking progress
- Enforce best practices (TDD, numerical validation, safe refactoring)

**You'll see:**
- Claude announcing: "I'm using the scientific-tdd skill..."
- Todo lists showing progress through workflow steps

**User action:** Monitor progress, approve at decision points

---

### Layer 3: Documentation (Context & Rules)

**What it does:**
- Explains when to use which skill
- Defines numerical tolerances
- Provides decision trees
- Documents the "why" behind rules

**You'll see:**
- Claude following patterns from CLAUDE.md
- References to specific sections

**User action:** Update CLAUDE.md if needs change

---

## Common Workflows

### Workflow 1: Adding a New Feature

**You:** "Add a new likelihood model based on Gaussian mixtures"

**System does:**
1. Claude announces: "Using scientific-tdd skill"
2. Claude writes failing test first (RED)
3. Claude implements minimal code (GREEN)
4. Claude runs full test suite
5. Claude runs numerical validation
6. Claude presents analysis
7. **YOU APPROVE** commit

**Your checkpoints:**
- Review test (does it test the right thing?)
- Review implementation (makes sense?)
- Review numerical validation (no unexpected changes?)
- Approve commit

---

### Workflow 2: Fixing a Bug

**You:** "Fix the bug where transition matrix isn't normalized"

**System does:**
1. If existing tests cover bug: Claude fixes directly, runs tests
2. If no coverage: Claude uses scientific-tdd to add test first
3. Claude runs full test suite
4. Claude runs numerical validation (bug fix likely changes outputs)
5. Claude provides full analysis of numerical changes
6. **YOU APPROVE** commit after reviewing analysis

**Your checkpoints:**
- Does fix make sense?
- Are numerical changes expected?
- Do invariants still hold?

---

### Workflow 3: Refactoring Code

**You:** "Refactor the HMM filtering to use JAX scan instead of loops"

**System does:**
1. Claude announces: "Using safe-refactoring skill"
2. Claude captures test baseline
3. Claude captures snapshot baseline
4. Claude performs refactoring
5. Claude verifies EXACT match (zero tolerance)
6. Claude runs numerical validation (must be < 1e-14 difference)
7. **YOU APPROVE** commit

**Your checkpoints:**
- No test changes? (must be exactly same)
- No snapshot changes? (must be exactly same)
- Numerical differences < 1e-14? (floating-point noise only)

---

### Workflow 4: Optimizing JAX Code

**You:** "Optimize the likelihood calculation for better performance"

**System does:**
1. Claude announces: "Using jax + numerical-validation skills"
2. Claude captures performance baseline
3. Claude uses jax skill for optimization approach
4. Claude runs numerical validation
5. Claude compares performance (before/after)
6. Claude verifies numerical equivalence (< 1e-10)
7. Claude provides analysis (speedup + numerical validation)
8. **YOU APPROVE** commit

**Your checkpoints:**
- Performance actually improved?
- Numerical differences acceptable?
- No degradation in accuracy?

---

## Approval Gates Explained

### When Claude Asks for Approval

Claude MUST ask before:

1. **Updating snapshots** (`--snapshot-update`)
2. **Committing changes** (`git commit`)
3. **Pushing to remote** (`git push`)

### Snapshot Update Approval

When Claude asks to update snapshots, you'll receive:

```
Snapshot Analysis:

1. DIFF:
   [Exact changes shown]

2. EXPLANATION:
   [Why it changed]

3. VALIDATION:
   [Proof invariants hold]

4. TEST CASE:
   [Before/after comparison]

Approve snapshot update?
```

**Review checklist:**
- [ ] Do I understand why it changed?
- [ ] Are mathematical properties preserved?
- [ ] Is the magnitude of change acceptable?
- [ ] Does the test case demonstrate correctness?

**If yes:** Reply "Yes" or "Approved"
**If no:** Ask for clarification or reject

### Commit Approval

Claude will present:
- What changed (files + summary)
- What tests verified the change
- Numerical validation results (if applicable)
- Proposed commit message

**Review checklist:**
- [ ] Changes make sense?
- [ ] Tests cover the changes?
- [ ] No unexpected side effects?
- [ ] Commit message descriptive?

**If yes:** Approve
**If no:** Request changes

---

## Numerical Tolerance Guide

Understanding when to approve numerical changes:

### ✅ Safe to Approve (< 1e-14)

**Scenario:** Pure refactoring, code restructuring
**Example:** "Changed for loop to JAX scan"
**Tolerance:** < 1e-14 (floating-point noise)
**Action:** Approve automatically

### ⚠️ Review Carefully (1e-14 to 1e-10)

**Scenario:** Algorithm tweaks, optimization
**Example:** "Changed optimizer tolerance from 1e-6 to 1e-8"
**Tolerance:** 1e-14 to 1e-10
**Action:** Review explanation, verify invariants, then approve

### 🚨 Scrutinize (> 1e-10)

**Scenario:** Significant algorithm changes
**Example:** "Switched from EM to gradient descent"
**Tolerance:** > 1e-10
**Action:** Demand strong justification, verify with domain expert if needed

---

## Troubleshooting

### "❌ Wrong conda environment detected"

**Cause:** Claude trying to run Python without conda activated

**Solution:** Hook auto-prepends activation, no action needed

**Alternative:** Tell Claude: "Activate the conda environment first"

---

### "❌ Snapshot update requires approval"

**Cause:** Claude tried `--snapshot-update` without showing analysis

**Solution:** Ask Claude: "Show me the full snapshot analysis first"

**Claude should then provide the 4-part analysis**

---

### "Tests are failing after change"

**Cause:** Change introduced regression

**Solution:**
1. Ask Claude to show which tests failed
2. Review the failures
3. Decide: Fix the bug OR revert the change

**Safety:** Hooks remind to run tests before commits

---

### "Numerical differences are larger than expected"

**Cause:** Change affected algorithm behavior more than anticipated

**Solution:**
1. Ask Claude for detailed numerical analysis
2. Check if mathematical properties still hold
3. Verify against golden regression tests
4. Decide if magnitude is scientifically acceptable

**Don't approve if:**
- You don't understand why it changed
- Invariants are violated
- Magnitude seems too large

---

## Customizing the System

### Adjusting Tolerances

Edit `CLAUDE.md` section "Numerical Accuracy Standards":

```markdown
### Tolerance Specifications

| Change Type | Max Acceptable Difference | Approval Required |
|-------------|---------------------------|-------------------|
| Pure refactoring | 1e-14 | No |
| Code optimization | 1e-12 | Yes |  # Changed from 1e-10
...
```

Claude will follow updated tolerances.

---

### Adding New Skills

If you develop new workflow patterns:

1. Create `.claude/skills/your-skill/skill.md`
2. Follow existing skill format
3. Add to CLAUDE.md "Workflow Selection Guide"
4. Update `.claude/skills/README.md`

---

### Modifying Hooks

If environment changes (e.g., new conda env name):

1. Edit `.claude/hooks/lib/env_check.sh`
2. Update `required_env` variable
3. Test with `tests/test_safe_dev_system.sh`

---

## Best Practices

### For Users

1. **Start with clear requests**: "Add feature X" vs "Fix the code"
2. **Review analyses carefully**: Don't blindly approve
3. **Ask questions**: If unsure, ask Claude to explain
4. **Trust but verify**: System is good, but you're the expert

### For Claude

1. **Announce skill usage**: Always say which skill you're using
2. **Show your work**: Display test outputs, not just summaries
3. **Ask before approval gates**: Never assume approval
4. **Provide complete analyses**: All 4 parts for snapshot updates

---

## FAQ

**Q: Can I skip the workflow for small changes?**
A: Skills are designed to be fast. Even "small" changes benefit from the safety checks. But for truly trivial changes (typos in comments), Claude can skip TDD.

**Q: What if hooks block something that should be allowed?**
A: Check `.claude/hooks/README.md` for debugging. You can temporarily disable hooks by making them non-executable, but this defeats the safety system.

**Q: How do I know if Claude is following the skills?**
A: Claude should announce skill usage and create TodoWrite todos. Check the chat for these indicators.

**Q: Can I use this system with other projects?**
A: Yes! Copy the `.claude/` directory and adapt CLAUDE.md for your project's needs. Adjust conda environment names and tolerance values.

**Q: What if I disagree with a numerical validation result?**
A: You're the domain expert. If Claude says "differences are acceptable" but you disagree, reject the change and ask Claude to investigate further or revert.

---

## Support

**System not working?**
1. Run `./tests/test_safe_dev_system.sh` to verify installation
2. Check hook permissions: `ls -l .claude/hooks/*.sh`
3. Review recent changes to CLAUDE.md

**Need help?**
- Check `.claude/skills/README.md` for skill details
- Check `.claude/hooks/README.md` for hook troubleshooting
- Review this guide's troubleshooting section

---

## Summary

The safe scientific development system provides three layers of protection:

1. **Hooks** prevent catastrophic failures automatically
2. **Skills** guide Claude through validated workflows
3. **Documentation** provides context and decision criteria

**Your role:**
- Review analyses at approval gates
- Verify numerical changes make sense
- Trust the system, but verify the results

**Claude's role:**
- Follow appropriate skills
- Provide complete analyses
- Ask for approval at gates
- Respect your decisions

Together, this enables confident AI-assisted development of scientific software.
