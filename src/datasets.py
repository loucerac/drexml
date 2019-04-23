#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Carlos Loucera <carlos.loucera@juntadeandalucia.es>
Author: Maria Pena Chilet <maria.pena.chilet.ext@juntadeandalucia.es>
Author: Marina Esteban <marina.estebanm@gmail.com>

Basic IO functionallity for Achilles cell line predictions.
"""

import sys
import os
import multiprocessing
import itertools
from pathlib import Path

import pandas as pd
import numpy as np

import feather

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

    if disease == "fanconi":
        circuits_fname = "circuits_FA.rds.feather"
    else:
        circuits_fname = "circuits.rds.feather"

    return circuits_fname

def get_pathvals_fname(disease):

    if disease == "fanconi":
        pathvals_fname = "expreset_pathvals_FA.rds.feather"
    else:
        pathvals_fname = "expreset_pathvals.rds.feather"

    return pathvals_fname

def load_circuits(disease):

    circuits_fname = get_circuits_fname(disease)

    circuits = feather.read_dataframe(DATA_PATH.joinpath(circuits_fname))
    circuits.set_index("index", drop=True, inplace=True)
    circuits.index = circuits.index.astype(str)
    circuits.replace({"FALSE": False, "TRUE": True}, inplace=True)
    circuits.index = circuits.index.str.replace("-", ".").str.replace(" ", ".")

    return circuits

def load_genes():

    genes = feather.read_dataframe(DATA_PATH.joinpath(genes_fname))
    genes.set_index("index", drop=True, inplace=True)
    genes.index = genes.index.astype(str)
    genes.replace({"FALSE": False, "TRUE": True}, inplace=True)

    return genes

def load_expression():

    expression = feather.read_dataframe(DATA_PATH.joinpath(expression_fname))
    expression.columns = expression.columns.str.replace("X", "")
    expression.set_index("index", drop=True, inplace=True)
    expression.index = expression.index.astype(str)

    return expression

def load_pathvals(disease):

    pathvals_fname = get_pathvals_fname(disease)
    pathvals = feather.read_dataframe(DATA_PATH.joinpath(pathvals_fname))
    pathvals.set_index("index", drop=True, inplace=True)
    pathvals.index = pathvals.index.astype(str)
    pathvals.columns = (
        pathvals.columns
        .str.replace("-", ".")
        .str.replace(" ", "."))

    return pathvals

def get_disease_data(disease, pathways=None):

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
    clinical_info_path = DATA_PATH.joinpath(clinical_info_fname)
    clinical_info = feather.read_dataframe(clinical_info_path)
    clinical_info.set_index("index", drop=True, inplace=True)
    clinical_info.index = clinical_info.index.astype(str)

    return clinical_info

def test_integrity(x, y, msg):
    if not np.array_equal(x.index.shape, y.index.shape):
        print(x.index.shape, y.index.shape)
        sys.exit(msg + " sample index differ in shape.")

    y = y.loc[x.index, :]

    if not x.index.equals(y.index):
        sys.exit(msg + " different sample indexes.")

    return x, y

