# Claude Code Skills for Safe Scientific Development

This directory contains custom skills that enforce best practices for scientific software development with the `non_local_detector` project.

## Available Skills

### 1. scientific-tdd
**Purpose:** Pragmatic test-driven development for scientific code

**Use when:**
- Implementing new features or algorithms
- Making complex modifications
- Adding new mathematical models

**Workflow:**
1. Write failing test (RED)
2. Implement minimal code (GREEN)
3. Run full test suite (check regressions)
4. Run numerical validation (if mathematical)
5. Refactor and commit

**Invocation:** Claude automatically uses this skill for new features

---

### 2. numerical-validation
**Purpose:** Verify mathematical correctness and numerical accuracy

**Use when:**
- Modifying HMM algorithms (`core.py`)
- Changing likelihood models
- Updating state transitions
- Refactoring mathematical code

**Workflow:**
1. Capture baseline outputs
2. Make code change
3. Capture new outputs
4. Compare numerically (check tolerances)
5. Verify mathematical invariants
6. Run property and golden regression tests
7. Generate full analysis report
8. Request approval if needed

**Tolerances:**
- Refactoring: < 1e-14 (floating-point noise)
- Algorithm changes: < 1e-10 (must justify)

---

### 3. safe-refactoring
**Purpose:** Change code structure without changing behavior

**Use when:**
- Improving code readability
- Extracting reusable functions
- Reorganizing code structure
- Performance optimization (numerically equivalent)

**Workflow:**
1. Establish test baseline
2. Establish snapshot baseline
3. Perform refactoring
4. Verify exact match (zero tolerance)
5. Verify coverage maintained
6. Run quality checks
7. Commit

**Zero tolerance for:**
- Any test status changes
- Any snapshot differences
- Decreased coverage
- Behavioral changes

---

## How Skills Work

1. **Automatic invocation**: Claude reads CLAUDE.md and invokes appropriate skill based on task
2. **Checklist generation**: Skills create TodoWrite todos for tracking progress
3. **Enforcement**: Skills provide step-by-step procedures that must be followed
4. **Integration**: Skills work together (e.g., scientific-tdd + numerical-validation)

## Skill Development

These skills are specific to the `non_local_detector` project and its scientific computing requirements. They enforce:

- Conda environment usage
- Numerical accuracy verification
- Mathematical invariant checking
- Test-driven development
- Zero-regression tolerance

## Related Documentation

- **CLAUDE.md**: Project-wide guidance and operational rules
- **Hooks** (`.claude/hooks/`): Technical enforcement of environment and testing
- **Testing Plan** (`TESTING_PLAN.md`): Coverage improvement roadmap

## Maintenance

Skills should be updated when:
- New testing patterns emerge
- Tolerance specifications change
- Workflow improvements identified
- New validation requirements added
