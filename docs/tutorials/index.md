# Tutorials

Each tutorial is a complete analysis of a simulated recording session, from the
raw inputs to the decoded results. They run on every documentation build, so
the code and the figures you see match the current version of the package.

Each page links to its source, a [jupytext](https://jupytext.readthedocs.io)
percent-format Python script. Run it as a script, or open it as a notebook
with `jupytext --to notebook <file>.py`.

- [Detecting non-local events across a session](05_detecting_non_local_events.py):
  fit on running, decode a whole session, extract replay events and describe
  each one.
