#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Carlos Loucera <carlos.loucera@juntadeandalucia.es>
Author: Maria Pena Chilet <maria.pena.chilet.ext@juntadeandalucia.es>
Author: Marina Esteban <marina.estebanm@gmail.com>

Basic IO functionallity for HORD project.
"""

import os
import sys
from pathlib import Path

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


def load_circuits(disease, from_env=None):
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

    if from_env is None:
        circuits_fname = get_circuits_fname(disease)
        circuits_fpath = DATA_PATH.joinpath(circuits_fname)
    else:
        circuits_fpath = from_env

    circuits = read_feather(circuits_fpath)
    circuits.set_index("index", drop=True, inplace=True)
    circuits.index = circuits.index.astype(str)
    circuits.replace({"FALSE": False, "TRUE": True}, inplace=True)
    circuits.index = circuits.index.str.replace("-", ".").str.replace(" ", ".")

    return circuits


def load_genes(path=None):
    """Load gene metadata dataset.

    Returns
    -------
    DataFrame, [n_features, ]
        Gene metadata dataset.
    """

    if path is None:
        genes_fpath = DATA_PATH.joinpath(genes_fname)
    else:
        genes_fpath = path

    genes = read_feather(genes_fpath)
    genes.set_index("index", drop=True, inplace=True)
    genes.index = genes.index.astype(str)

    try:
        genes.replace({"FALSE": False, "TRUE": True}, inplace=True)
    except:
        print("no str data")

    return genes


def load_expression(path=None):
    """Load gene expression dataset.

    Returns
    -------
    DataFrame, [n_samples, n_features]
        Gene expression dataset (Gtex).
    """

    if path is None:
        fpath = DATA_PATH.joinpath(expression_fname)
    else:
        fpath = path

    expression = read_feather(fpath)
    expression.columns = expression.columns.str.replace("X", "")
    expression.set_index("index", drop=True, inplace=True)
    expression.index = expression.index.astype(str)

    return expression


def load_pathvals(disease, from_env=None):
    """Load pathvals dataset.

    Returns
    -------
    DataFrame, [n_samples, n_features]
        Pathvals dataset (Gtex).
    """

    if from_env is not None:
        pathvals_fpath = from_env
    else:
        pathvals_fname = get_pathvals_fname(disease)
        pathvals_fpath = DATA_PATH.joinpath(pathvals_fname)

    pathvals = read_feather(pathvals_fpath)
    pathvals.set_index("index", drop=True, inplace=True)
    pathvals.index = pathvals.index.astype(str)
    pathvals.columns = (
        pathvals.columns
            .str.replace("-", ".")
            .str.replace(" ", "."))

    return pathvals

def get_disease_data_new(disease):

    # Load data
    experiment_env_path = Path(disease)
    dotenv.load_dotenv(experiment_env_path)
    experiment_data_path = Path(os.getenv("data_path"))

    gene_exp_fname = os.getenv("gene_exp")
    gene_exp_fpath = experiment_data_path.joinpath(gene_exp_fname)
    gene_exp = load_expression(gene_exp_fpath)

    pathvals_fname = os.getenv("pathvals")
    pathvals_fpath = experiment_data_path.joinpath(pathvals_fname)
    pathvals = load_pathvals(None, from_env=pathvals_fpath)

    circuits_fname = os.getenv("circuits")
    circuits_fpath = experiment_data_path.joinpath(circuits_fname)
    path_metadata = load_circuits(None, from_env=circuits_fpath)

    genes_fname = os.getenv("genes")
    genes_fpath = experiment_data_path.joinpath(genes_fname)
    genes_metadata = load_genes(genes_fpath)

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
    genes_inuse = os.getenv("genes_column")
    genes_query = genes_metadata.entrezs[genes_metadata[genes_inuse]]
    gene_exp = gene_exp.loc[:, genes_query].copy()

    circuits_inuse = os.getenv("circuits_column")
    circuits_query = path_metadata.index[path_metadata[circuits_inuse]]
    pathvals = pathvals.loc[:, circuits_query].copy()

    return gene_exp, pathvals, path_metadata, gene_metadata, clinical_info


def get_disease_data(disease, pathways=None, gset="all"):
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

    env_possible = Path(disease)

    if env_possible.exists() and (env_possible.suffix == ".env"):
        gene_exp, pathvals, path_metadata, gene_metadata, clinical_info = get_disease_data_new(disease)
    else:
        gene_exp, pathvals, path_metadata, gene_metadata, clinical_info = get_disease_data_old(disease, pathways, gset)

    return gene_exp, pathvals, path_metadata, gene_metadata, clinical_info


def get_disease_data_old(disease, pathways=None, gset="all"):
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
    if gset == "target":
        query_gene_ids = gene_metadata.index[gene_metadata.approved_targets]
    elif gset == "all":
        query = gene_metadata.approved_targets | gene_metadata.in_FAext
        query_gene_ids = gene_metadata.index[query]
        # query_gene_ids = np.random.choice(gene_metadata.index.values, size=100, replace=False)
    gene_exp = gene_exp[query_gene_ids]

    if not len(pathways):
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
