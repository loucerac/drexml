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


def get_disease_data(disease):
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
    experiment_data_path = pathlib.Path(os.getenv("data_path"))

    genes_column = os.getenv("genes_column")
    circuits_column = os.getenv("circuits_column")

    gene_exp = load_df(experiment_data_path.joinpath(os.getenv("gene_exp")))
    gene_exp.columns = gene_exp.columns.str.replace("X", "")

    pathvals = load_df(experiment_data_path.joinpath(os.getenv("pathvals")))
    pathvals.columns = pathvals.columns.str.replace("-", ".").str.replace(" ", ".")

    circuits = load_df(experiment_data_path.joinpath(os.getenv("circuits")))
    circuits.index = circuits.index.str.replace("-", ".").str.replace(" ", ".")

    genes = load_df(experiment_data_path.joinpath(os.getenv("genes")))
    genes.index = genes.index.astype(str)

    gene_exp = gene_exp[genes.index[genes[genes_column]]]
    pathvals = pathvals[circuits.index[circuits[circuits_column]]]

    return gene_exp, pathvals, circuits, genes


def fetch_data(version="latest"):
    """[summary]

    Parameters
    ----------
    data_home : [type], optional
        [description], by default None
    download_if_missing : bool, optional
        [description], by default True
    """

    zenodo = Zenodo()

    record_id = "6020481"

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
