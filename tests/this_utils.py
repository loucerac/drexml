# -*- coding: utf-8 -*-
"""
Test utilities module.
"""

import shutil
import tempfile
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).absolute().parent

DATA_DICT_NONAN = {
    "circuits": "circuits.tsv.gz",
    "gene_exp": "gene_exp.tsv.gz",
    "pathvals": "pathvals.tsv.gz",
}

DATA_DICT_WITHNAN = {
    "circuits": "circuits.tsv.gz",
    "gene_exp": "gene_exp_withnan.tsv.gz",
    "pathvals": "pathvals.tsv.gz",
}


def read_test_circuits():
    """Read test circuits."""
    return pd.read_csv(THIS_DIR.joinpath("circuits.tsv.gz"), sep="\t", index_col=0)


def read_test_gene_exp():
    """Read test gene expression."""
    return pd.read_csv(THIS_DIR.joinpath("gene_exp.tsv.gz"), sep="\t", index_col=0)


def read_test_pathvals():
    """Read test pathvals."""
    return pd.read_csv(THIS_DIR.joinpath("pathvals.tsv.gz"), sep="\t", index_col=0)


def make_disease_config(use_seeds=True, update=False, use_physio=True, impute=False):
    """Prepare fake disease data folder."""
    tmp_dir = Path(tempfile.mkdtemp())

    if impute:
        experiment_name = "experiment_withnan.env"
        data_dict = DATA_DICT_WITHNAN
    else:
        experiment_name = "experiment.env"
        data_dict = DATA_DICT_NONAN

    disease_path_out = tmp_dir.joinpath(experiment_name)
    with open(disease_path_out, mode="w", encoding="utf8") as this_file:
        pass

    with open(disease_path_out, "w", encoding="utf8") as this_file:
        for key, file_name in data_dict.items():
            if use_seeds:
                if key == "circuits":
                    continue
            else:
                if key == "circuits":
                    this_file.write(f"{key}=./{file_name}\n")

            if not update:
                this_file.write(f"{key}=./{file_name}\n")

        this_file.write(f"use_physio={use_physio}\n")

        if use_seeds:
            this_file.write("seed_genes=2180\n")

    for _, file_name in data_dict.items():
        shutil.copy(
            THIS_DIR.joinpath(file_name).as_posix(),
            tmp_dir.joinpath(file_name).as_posix(),
        )

    return disease_path_out
