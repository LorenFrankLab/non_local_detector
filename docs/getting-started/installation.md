# Installation

`non_local_detector` supports Python 3.10 and later. It runs on CPU everywhere
and on NVIDIA GPUs through JAX.

!!! note "These pages describe the development version"
    The documentation follows the `main` branch. The latest release on PyPI,
    0.6.9, predates the explicit time-bin API used throughout these pages
    (see [Time bins and rates](../time_grid_migration.md)). Until the next
    release, install from GitHub to follow along.

## From GitHub (development version)

```bash
pip install "non_local_detector @ git+https://github.com/LorenFrankLab/non_local_detector"
```

## From PyPI (latest release)

```bash
pip install non_local_detector
```

## GPU

JAX runs the decoding on an NVIDIA GPU when a CUDA build of JAX is installed.
The `gpu` extra installs one from PyPI:

```bash
pip install "non_local_detector[gpu]"
```

Check that JAX sees the GPU:

```python
import jax

print(jax.devices())  # e.g. [CudaDevice(id=0)]
```

See [Settings for your hardware](../hardware_settings.md) for memory settings on
24 GB GPUs, A100s and CPU-only machines.

## Optional extras

| Extra | Installs | Needed for |
| --- | --- | --- |
| `gpu` | `jax[cuda13]` (`jax[cuda12]` on Python 3.10) | decoding on an NVIDIA GPU (Linux; NVIDIA driver 580 or newer, 525 for CUDA 12) |
| `viz` | `sortingview` | interactive FigURL views in `non_local_detector.visualization` |

CUDA 13 needs a GPU of compute capability 7.5 (Turing) or newer. On older
GPUs (Pascal, Volta), install `jax[cuda12]` instead of the `gpu` extra.

## For development

Clone the repository and let [uv](https://docs.astral.sh/uv/) create the
environment from the lock file:

```bash
git clone https://github.com/LorenFrankLab/non_local_detector.git
cd non_local_detector
uv sync
uv run pytest tests/test_version_import.py
```

See [CONTRIBUTING.md](https://github.com/LorenFrankLab/non_local_detector/blob/main/CONTRIBUTING.md)
for the checks a pull request runs.
