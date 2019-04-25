#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Carlos Loucera <carlos.loucera@juntadeandalucia.es>
Author: Maria Pena Chilet <maria.pena.chilet.ext@juntadeandalucia.es>
Author: Marina Esteban <marina.estebanm@gmail.com>

Basic IO functionallity for HORD project.
"""

import sys
import os
import multiprocessing
import itertools
from pathlib import Path

import pandas as pd
import numpy as np

try:
    from feather import read_dataframe as read_feather
except:
    from pyarrow.feather import read_feather

import dotenv


dotenv_file_path = Path(dotenv.find_dotenv())
project_path = dotenv_file_path.parent

# LOAD USER ENVIRONMENT VARIABLES
dotenv.load_dotenv(dotenv_file_path)
DATA_PATH = Path(os.environ.get("DATA_PATH"))
NUM_CPUS = int(os.environ.get("NUM_CPUS"))
USE_GPU = bool(os.environ.get("USE_GPU"))

genes_fname = "genes.rds.feather"
expression_fname = "expreset_Hinorm.rds.feather"
clinical_info_fname = "clinical_info_gtex.rds.feather"


def get_circuits_fname(disease):
    """Get circuits metadata file name based on the disease under study.

    Parameters
    ----------
    disease : str
        The disease under study.

    Returns
    -------
    str
        Circuits metadata file name.
    """
    if disease == "fanconi":
        circuits_fname = "circuits_FA.rds.feather"
    else:
        circuits_fname = "circuits.rds.feather"

    return circuits_fname

def get_pathvals_fname(disease):
    """Get pathvals file name based on the disease under study.

    Parameters
    ----------
    disease : str
        The disease under study.

    Returns
    -------
    str
        Pathvals file name.
    """
    if disease == "fanconi":
        pathvals_fname = "expreset_pathvals_FA.rds.feather"
    else:
        pathvals_fname = "expreset_pathvals.rds.feather"

    return pathvals_fname

def load_circuits(disease):
    """Load ciruicts metadata for the disease under study.

    Parameters
    ----------
    disease : str
        The disease under study.

    Returns
    -------
    DataFrame
        Ciruits DataFrame.
    """
    circuits_fname = get_circuits_fname(disease)

    circuits = read_feather(DATA_PATH.joinpath(circuits_fname))
    circuits.set_index("index", drop=True, inplace=True)
    circuits.index = circuits.index.astype(str)
    circuits.replace({"FALSE": False, "TRUE": True}, inplace=True)
    circuits.index = circuits.index.str.replace("-", ".").str.replace(" ", ".")

    return circuits

def load_genes():
    """Load gene metadata dataset.

    Returns
    -------
    DataFrame, [n_features, ]
        Gene metadata dataset.
    """

    genes = read_feather(DATA_PATH.joinpath(genes_fname))
    genes.set_index("index", drop=True, inplace=True)
    genes.index = genes.index.astype(str)
    genes.replace({"FALSE": False, "TRUE": True}, inplace=True)

    return genes

def load_expression():
    """Load gene expression dataset.

    Returns
    -------
    DataFrame, [n_samples, n_features]
        Gene expression dataset (Gtex).
    """
    expression = read_feather(DATA_PATH.joinpath(expression_fname))
    expression.columns = expression.columns.str.replace("X", "")
    expression.set_index("index", drop=True, inplace=True)
    expression.index = expression.index.astype(str)

    return expression

def load_pathvals(disease):
    """Load pathvals dataset.

    Returns
    -------
    DataFrame, [n_samples, n_features]
        Pathvals dataset (Gtex).
    """
    pathvals_fname = get_pathvals_fname(disease)
    pathvals = read_feather(DATA_PATH.joinpath(pathvals_fname))
    pathvals.set_index("index", drop=True, inplace=True)
    pathvals.index = pathvals.index.astype(str)
    pathvals.columns = (
        pathvals.columns
        .str.replace("-", ".")
        .str.replace(" ", "."))

    return pathvals

def get_disease_data(disease, pathways=None):
    """Load all datasets for a given dataset.

    Parameters
    ----------
    disease : str
        Disease under study.
    pathways : array like, str, [n_pathways, ]
        Pathways to use as ML targets, by default None refers to circuits given
        by in_disease column in ciruits metadata dataset.

    Returns
    -------
    DataFrame, DataFrame, DataFrame, DataFrame, DataFrame
        Gene expression, pathvals, circuit metadata, gene metadata, clinical
        metadata
    """
    # Load data
    gene_exp = load_expression()
    pathvals = load_pathvals(disease)
    path_metadata = load_circuits(disease)
    gene_metadata = load_genes()
    clinical_info = load_clinical_data()

    # test integrity and reorder by sample index

    gene_exp, pathvals = test_integrity(
        gene_exp,
        pathvals,
        "Gene expr. and Pathvals")
    gene_exp, clinical_info = test_integrity(
        gene_exp,
        clinical_info,
        "Gene expr. and Clinical data")

    # Filter data
    target_gene_ids = gene_metadata.index[gene_metadata.approved_targets]
    gene_exp = gene_exp[target_gene_ids]

    if pathways is None:
        disease_circuits = path_metadata.loc[path_metadata.in_disease].index
        pathvals = pathvals.loc[:, disease_circuits]
    elif len(pathways) > 1:
        pathvals = pathvals.filter(axis=1, regex="|".join(pathways))
    else:
        pathvals = pathvals.filter(axis=1, regex=pathways[0])

    return gene_exp, pathvals, path_metadata, gene_metadata, clinical_info

def load_clinical_data():
    """Load clinical metadata dataset.

    Returns
    -------
    DataFrame, [n_samples, n_features]
        Clinical metadata dataset (Gtex).
    """
    clinical_info_path = DATA_PATH.joinpath(clinical_info_fname)
    clinical_info = read_feather(clinical_info_path)
    clinical_info.set_index("index", drop=True, inplace=True)
    clinical_info.index = clinical_info.index.astype(str)

    return clinical_info

def test_integrity(x, y, msg):
    """Test combined dataset integrity.

    Parameters
    ----------
    x : DataFrame
        DataFrame to comapre.
    y : DataFrame
        DataFrame to comapre.
    msg : str
        Message to be printed.

    Returns
    -------
    DataFrame, DataFrame
        DataFrames reordered by a common index.
    """
    if not np.array_equal(x.index.shape, y.index.shape):
        print(x.index.shape, y.index.shape)
        sys.exit(msg + " sample index differ in shape.")

    y = y.loc[x.index, :]

    if not x.index.equals(y.index):
        sys.exit(msg + " different sample indexes.")

    return x, y
