[![DOI](https://zenodo.org/badge/362395439.svg)](https://zenodo.org/badge/latestdoi/362395439) [![PyPI version](https://badge.fury.io/py/drexml.svg)](https://badge.fury.io/py/drexml) [![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm.fming.dev)

# Drug REpurposing using eXplainable Machine Learning and Mechanistic Models of signal transduction

Repository for the `drexml` python package: (DRExM³L) Drug REpurposing using eXplainable Machine Learning and Mechanistic Models of signal transduction

## Setup

To install the `drexml` package use the following:

```
conda create -n drexml python=3.10
conda activate drexml
pip install drexml
```

If a CUDA~10.2/11.x (< 12) compatible device is available use:

```
conda create -n drexml --override-channels -c "nvidia/label/cuda-11.8.0" -c conda-forge cuda cuda-nvcc cuda-toolkit gxx=11.2 python=3.10
conda activate drexml
pip install --no-cache-dir --no-binary=shap drexml
```

To install `drexml` in an existing environment, activate it and use:

```
pip install drexml
```

Note that by default the `setup` will try to compile the `CUDA` modules, if not possible it will use the `CPU` modules.

## Run

To run the program for a disease map that uses circuits from the preprocessed `KEGG` pathways and the `KDT` standard list, construct an environment file (e.g. `disease.env`):

- using the following template if you have a set of seed genes (comma-separated):

```
seed_genes=2175,2176,2189
```

- using the following template if you want to use the DisGeNET [1] curated gene-disease associations as seeds.

```
disease_id="C0015625"
```

- using the following template if you know which circuits to include (the disease map):

```
circuits=circuits.tsv.gz
```

The `TSV` file `circuits.tsv` has the following format (tab delimited):

```
index	in_disease
P-hsa03320-37	0
P-hsa03320-61	0
P-hsa03320-46	0
P-hsa03320-57	0
P-hsa03320-64	0
P-hsa03320-47	0
P-hsa03320-65	0
P-hsa03320-55	0
P-hsa03320-56	0
P-hsa03320-33	0
P-hsa03320-58	0
P-hsa03320-59	0
P-hsa03320-63	0
P-hsa03320-44	0
P-hsa03320-36	0
P-hsa03320-30	0
P-hsa03320-28	1
```

where:

- `index`: Hipathia circuit id
- `in_disease`: (boolean) True/1 if a given circuit is part of the disease

Note that in all cases you can restrict the circuits to the physiological list by setting `use_physio=true` in the `env` file.

To run the experiment using 10 CPU cores and 0 GPUs, run the following command within an activated environment:

```
drexml run --n-gpus 0 --n-cpus 10 $DISEASE_PATH
```

where:

- `--n-gpus` indicates the number of gpu devices to use in parallel (-1 -> all) (0 -> None)
- `--n-cpus` indicates the number of cpu devices to use in parallel (-1 -> all) 8
- `DISEASE_PATH` indicates the path to the disease env file (e.g. `/path/to/disease/folder/disease.env`)

Use the `--debug` option for testing that everything works using a few iterations.

Note that the first time that the full program is run, it will take longer as it downloads the latest versions of each background dataset from Zenodo:

https://doi.org/10.5281/zenodo.6020480

## Contribute to development

The recommended setup is:

- setup `pipx`
- setup `miniforge`
- use `pipx` to install `pdm`
- ensure that `pdm` is version >=2.1, otherwise update with `pipx`
- use `pipx` to inject pdm-bump into `pdm`
- use `pipx` to install `nox`
- run `pdm config venv.backend conda`
- run `make`, if you want to use a CUDA enabled GPU, use `make gpu=1`
- (Recommended): For GPU development, clear the cache using `pdm clean cache` first

## Documentation

The documentation can be found here:

https://loucerac.github.io/drexml/


## References
[1] Janet Piñero, Juan Manuel Ramírez-Anguita, Josep Saüch-Pitarch, Francesco Ronzano, Emilio Centeno, Ferran Sanz, Laura I Furlong. The DisGeNET knowledge platform for disease genomics: 2019 update. Nucl. Acids Res. (2019) doi:10.1093/nar/gkz1021
