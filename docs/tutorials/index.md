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
- [Classifying replay dynamics](04_classifying_replay_dynamics.py): fit a
  classifier with continuous, fragmented and No-Spike states on running, decode
  a whole session, and classify simulated replay events of each type.
- [Decoding with clusterless spikes](03_decoding_clusterless.py): decode position
  from unsorted spikes and their waveform features with `ClusterlessDecoder`,
  fit on running and applied to a whole session.
- [Data format and environments](01_data_and_environments.py): what the
  inputs look like (position, sorted and clusterless spikes, decode bins) and
  how to describe space with open-field and track-graph environments.
