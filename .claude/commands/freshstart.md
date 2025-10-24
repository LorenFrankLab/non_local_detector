I'm working on the non_local_detector project, implementing a comprehensive regression detection system.

Start now by reading the files and telling me which task you'll work on first.

Your workflow MUST be:

1. First, read these files IN ORDER:
   - CLAUDE.md (implementation guide and operational rules)
   - docs/plans/2025-10-23-comprehensive-regression-detection.md (detailed implementation plan)
   - docs/TASKS.md (current task checklist)

2. Find the FIRST unchecked [ ] task in TASKS.md

3. For EVERY feature/test file you create, follow TDD:
   a. Create the TEST file first (or add test to existing file)
   b. Run the test and verify it FAILS with expected error
   c. Only then create the implementation
   d. Run test until it PASSES
   e. Apply review agents (code-reviewer if significant code)
   f. Refactor for clarity and efficiency based on feedback
   g. Add/Update docstrings and types

4. Update TASKS.md checkboxes as you complete items:
   - Change `- [ ]` to `- [x]` for completed tasks
   - Commit TASKS.md updates with the feature commit

5. For regression detection work specifically:
   - ALWAYS use full conda paths: `/Users/edeno/miniconda3/envs/non_local_detector/bin/pytest`
   - For property tests: use hypothesis with `@given` decorators
   - For golden tests: follow existing pattern in test_golden_regression.py
   - For API tests: track function signatures and class methods
   - Run numerical validation after mathematical code changes

6. Commit frequently with descriptive messages:
   - "test: add property test for transition matrix stochastic property"
   - "feat: add API surface monitoring infrastructure"
   - "fix: enable nonlocal_detector golden regression test"
   - "docs: update TASKS.md - complete task 1.1"

7. When you complete a task:
   - Mark it complete in TASKS.md with `- [x]`
   - Run the verification step from the plan
   - Show me what you completed
   - Ask if you should continue to next task

8. CRITICAL RULES from CLAUDE.md:
   - Must use conda environment: `non_local_detector`
   - Must ask permission before: updating snapshots, committing, pushing, modifying golden data
   - Must provide full analysis before requesting snapshot updates (see CLAUDE.md approval process)
   - Property test failures = mathematical invariant violations = DO NOT PROCEED

Do not change tests or skip tests to match broken code. Do not modify numerical tolerances without justification. Ask permission to change requirements if needed.

Start by reading the three files above, then tell me which specific task you'll work on first and your plan for implementing it.
