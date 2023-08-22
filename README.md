# non_local_detector

## Installation

## GPU Installation

In order to use GPU with the conda or mamba environment, you need to install the following packages:

Using mamba or conda (mamba recommended):

```bash
mamba install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
mamba install non_local_detector -c edeno
```

or from the environment file:

```bash
mamba create env -f environment_gpu.yml
```

or via pip:

```bash
pip install non_local_detector[gpu]
```

## CPU Installation

```bash
mamba create env -f environment.yml
```

or via pip:

```bash
pip install non_local_detector
```
