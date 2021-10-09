# Holistic Rare Disease

## Setup

In order to compute the SHAP explanations with a CUDA-enabled GPU install the `shap` change the `CUDAHOME` environment variable. Note that the `CUDAHOME` variable has already been specified in the conda environment file, see the conda version requirement.

The GPU version has been tested on GNU/Linux x64 4.15 with cuda 10.2 and conda >= 4.3 . Use the following command in order to create the python environment needed to run the experiments.

```
conda env create -p ./.venv -f environment.yml
```
