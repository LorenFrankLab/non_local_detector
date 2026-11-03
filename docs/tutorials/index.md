# Tutorials

Each tutorial is a complete analysis of a simulated recording session, from the
raw inputs to the decoded results. They run on every documentation build, so
the code and the figures you see match the current version of the package.

Each page links to its source, a [jupytext](https://jupytext.readthedocs.io)
percent-format Python script. Run it as a script, or open it as a notebook
with `jupytext --to notebook <file>.py`.

- [Decoding with clusterless spikes](03_decoding_clusterless.py): decode position
  from unsorted spikes and their waveform features with `ClusterlessDecoder`,
  fit on running and applied to a whole session.
- [Data format and environments](01_data_and_environments.py): what the
  inputs look like (position, sorted and clusterless spikes, decode bins) and
  how to describe space with open-field and track-graph environments.
