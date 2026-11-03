# Tutorials

Each tutorial is a complete analysis of a simulated recording session, from the
raw inputs to the decoded results. They run on every documentation build, so
the code and the figures you see match the current version of the package.

Each page links to its source, a [jupytext](https://jupytext.readthedocs.io)
percent-format Python script. Run it as a script, or open it as a notebook
with `jupytext --to notebook <file>.py`.

- [Classifying replay dynamics](04_classifying_replay_dynamics.py): fit a
  classifier with continuous, fragmented and No-Spike states on running, decode
  a whole session, and classify simulated replay events of each type.
