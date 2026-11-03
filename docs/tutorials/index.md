# Tutorials

Each tutorial is a complete analysis of a simulated recording session, from the
raw inputs to the decoded results. They run on every documentation build, so
the code and the figures you see match the current version of the package.

Each page links to its source, a [jupytext](https://jupytext.readthedocs.io)
percent-format Python script. Run it as a script, or open it as a notebook
with `jupytext --to notebook <file>.py`.

- [Decoding a session with sorted spikes](02_decoding_sorted_spikes.py): fit
  a `SortedSpikesDecoder` on the running periods, decode a whole session,
  measure the decoding error and save the results, on a linear track, a
  circular track and in a 2-D open field.
