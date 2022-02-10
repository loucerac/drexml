#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Carlos Loucera <carlos.loucera@juntadeandalucia.es>
Author: Maria Pena Chilet <maria.pena.chilet.ext@juntadeandalucia.es>
Author: Marina Esteban <marina.estebanm@gmail.com>

IO module for HORD multi-task framework.
"""
import os
import pathlib

import dotenv
import pandas as pd
from pandas.errors import ParserError
from zenodo_client import Zenodo

DEFAULT_STR = "$default$"
DEBUG_NAMES = {
    "gene_exp": "gene_exp.tsv.gz",
    "pathvals": "patvals.tsv.gz",
    "circuits": "circuits.tsv.gz",
    "genes": "genes.tsv.gz",
}

PRODUCTION_NAMES = {
    "gene_exp": "expreset_Hinorm_gtexV8.rds.feather",
    "pathvals": "expreset_pathvals_gtexV8.rds.feather",
    "genes": "genes01072021.rds.feather",
}

NAMES = {True: DEBUG_NAMES, False: PRODUCTION_NAMES}

RECORD_ID = "6020481"


def fetch_file(disease, key, version="latest", debug=False):
    """Retrieve data."""
    experiment_env_path = pathlib.Path(disease)
    dotenv.load_dotenv(experiment_env_path)
    if os.getenv(key) == DEFAULT_STR:
        if version == "latest":
            zenodo = Zenodo()
            path = zenodo.download_latest(RECORD_ID, NAMES[debug][key], force=False)
    else:
        data_path = pathlib.Path(os.getenv("data_path"))
        if data_path == DEFAULT_STR:
            data_path = experiment_env_path.parent
        path = data_path.joinpath(os.getenv(key))

    return load_df(path)


def fetch_data(zenodo=None, version="latest", debug=False):
    """[summary]

    Parameters
    ----------
    data_home : [type], optional
        [description], by default None
    download_if_missing : bool, optional
        [description], by default True
    """
    if zenodo is None:
        zenodo = Zenodo()

    record_id = "6020481"
    if debug:
        fname_lst = [
            "gene_exp.tsv.gz",
            "pathvals.tsv.gz",
            "genes.tsv.gz",
        ]

    else:
        fname_lst = [
            "expreset_Hinorm_gtexV8.rds.feather",
            "expreset_pathvals_gtexV8.rds.feather",
            "genes01072021.rds.feather",
        ]

    for fname in fname_lst:
        if version == "latest":
            this_path = zenodo.download_latest(record_id, fname, force=False)

    this_path = pathlib.Path(this_path)

    return this_path.parent


def load_df(path):
    """[summary]

    Parameters
    ----------
    path : path-like
        Data file path.

    Returns
    -------
    pd.DataFrame
        Project data

    Raises
    ------
    NotImplementedError
        Data for mat not implemented.
    """
    errors = {}
    try:
        # tsv, and compressed tsv
        return pd.read_csv(path, sep="\t").set_index("index", drop=True)
    except ParserError as err:
        errors["tsv"] = err
    except KeyError as key_err:
        errors["key"] = key_err

    try:
        # feather
        return pd.read_feather(path).set_index("index", drop=True)
    except ParserError as err:
        errors["feather"] = err
    except KeyError as key_err:
        errors["key"] = key_err

    print("The following exceptions have been catched:")
    for key, value in errors:
        print(f"{key}: {value}")
    raise NotImplementedError("Format not implemented yet.")


def get_disease_data(disease, debug):
    """[summary]

    Parameters
    ----------
    disease : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """

    # Load data
    experiment_env_path = pathlib.Path(disease)
    dotenv.load_dotenv(experiment_env_path)
    # experiment_folder = experiment_env_path.parent
    # check the cache, download if different, return storage folder
    # zenodo_path = fetch_data(debug=debug)
    # data_path = pathlib.Path(os.getenv("data_path"))

    genes_column = os.getenv("genes_column")
    circuits_column = os.getenv("circuits_column")

    # gene_exp_fname = os.getenv("gene_exp")
    gene_exp = fetch_file(disease, key="gene_exp", version="latest", debug=debug)

    # gene_exp = load_df(data_path.joinpath(os.getenv("gene_exp")))
    gene_exp.columns = gene_exp.columns.str.replace("X", "")

    # pathvals = load_df(data_path.joinpath(os.getenv("pathvals")))
    pathvals = fetch_file(disease, key="pathvals", version="latest", debug=debug)
    pathvals.columns = pathvals.columns.str.replace("-", ".").str.replace(" ", ".")

    # if no data folder is provided, the circuits should be in the same folder as
    # the env file.
    # if data_path == zenodo_path:
    #     circuits_path = experiment_folder.joinpath(os.getenv("circuits"))
    # else:
    #     circuits_path = data_path.joinpath(os.getenv("circuits"))

    # circuits = load_df(circuits_path)
    circuits = fetch_file(disease, key="circuits", version="latest", debug=debug)
    circuits.index = circuits.index.str.replace("-", ".").str.replace(" ", ".")

    # genes = load_df(data_path.joinpath(os.getenv("genes")))
    genes = fetch_file(disease, key="genes", version="latest", debug=debug)
    genes.index = genes.index.astype(str)

    gene_exp = gene_exp[genes.index[genes[genes_column]]]
    pathvals = pathvals[circuits.index[circuits[circuits_column]]]

    return gene_exp, pathvals, circuits, genes
