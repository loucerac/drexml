# Drug REpurposing using Mechanistic Models of signal transduction and -eXplainable Machine Learning

Repository for the `drexml` python package: (DRExM³L) Drug REpurposing using Mechanistic Models of signal transduction and eXplainable Machine Learning 

## Setup

To install the `drexml` package use the following:

```
conda create -p ./.venv python=3.8
conda run -p ./.venv git+ssh://git@github.com:loucerac/drexml.git
```

## Run

To run the program for a disease map that uses circuits from the preprocessed `KEGG` pathways and the `KDT` standard list, construct an environment file (e.g. `disease.env`) using the following template:

```
gene_exp=$default$
pathvals=$default$
circuits=circuits.tsv.gz
circuits_column=in_disease
genes=$default$
genes_column=approved_targets
```

Where `circuits.tsv` has the following format (tab delimited):
```
index	in_disease
P-hsa03320-37	False
P-hsa03320-61	False
P-hsa03320-46	False
P-hsa03320-57	False
P-hsa03320-64	False
P-hsa03320-47	False
P-hsa03320-65	False
P-hsa03320-55	False
P-hsa03320-56	False
P-hsa03320-33	False
P-hsa03320-58	False
P-hsa03320-59	False
P-hsa03320-63	False
P-hsa03320-44	False
P-hsa03320-36	False
P-hsa03320-30	False
P-hsa03320-28	True
```

where:
* `index`: Hipathia circuit id
* `in_disease`: boolean if a given circuit is part of the disease

```
conda run -p ./.venv drexml run --n-gpus 0 --n-cpus 10 disease.env
```

where:
* `--n-gpus` indicates the number of gpu devices to use in parallel (-1 -> all) (0 -> None)
* `--n-cpus` indicates the number of cpu devices to use in parallel (-1 -> all) 8

## Documentation
The documentation can be found here:

https://loucerac.github.io/drexml/
