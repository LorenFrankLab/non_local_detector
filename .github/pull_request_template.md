## Summary

<!-- What does this change do, and why? Link related issues (e.g. "Fixes #123"). -->

## Tests

<!-- Which tests cover the change? Paste the targeted pytest command you ran. -->

## Checklist

- [ ] Tests added or updated for the change
- [ ] `uv run pre-commit run --all-files` passes (ruff, strict mypy, codespell)
- [ ] `CHANGELOG.md` has an entry under `[Unreleased]` for any user-visible change
- [ ] Docstrings (numpydoc, with array shapes) and documentation are updated

## Numerical changes

<!-- Delete this section if outputs are unchanged. -->

- [ ] This change alters numerical outputs (posteriors, likelihoods, fitted parameters, golden regression or snapshot data)

If checked: describe the difference (size, which outputs), why it is correct or an improvement, and which invariants you verified (normalization, finite values, stochastic transition matrices). Golden regression and snapshot data are updated only after this analysis is reviewed.
