[![DOI](https://zenodo.org/badge/362395439.svg)](https://zenodo.org/badge/latestdoi/362395439)

# Drug REpurposing using eXplainable Machine Learning and Mechanistic Models of signal transduction

Repository for the `drexml` python package: (DRExM³L) Drug REpurposing using eXplainable Machine Learning and Mechanistic Models of signal transduction

## Setup

To install the `drexml` package use the following:

```
conda create -n drexml python=3.10
conda run -n drexml pip install git+https://github.com/loucerac/drexml.git@master
```

If a CUDA~11 compatible device is available use:

```
conda create -n drexml --override-channels -c "nvidia/label/cuda-11.8.0" -c conda-forge cuda cuda-nvcc cuda-toolkit gxx=11.2 python=3.10
conda run -n drexml pip install git+https://github.com/loucerac/drexml.git@master
```

To install `drexml` in an existing environment, activate it and use:

```
pip install git+https://github.com/loucerac/drexml.git@master
```

Note that by default the `setup` will try to compile the `CUDA` modules, if not possible it will use the `CPU` modules.

To install the development version use `@develop` instead of `@master`.

## Run

To run the program for a disease map that uses circuits from the preprocessed `KEGG` pathways and the `KDT` standard list, construct an environment file (e.g. `disease.env`):

- using the following template if you have a set of seed genes (comma-separated):

```
seed_genes=2175,2176,2189
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
- use `pipx` to install `poetry`
- use `pipx` to install `nox` and inject `nox-poetry` into `nox`
- run `make`, if you want to use a CUDA enabled GPU, use `make gpu=1`

## Documentation

The documentation can be found here:

https://loucerac.github.io/drexml/
