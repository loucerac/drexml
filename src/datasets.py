#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Carlos Loucera <carlos.loucera@juntadeandalucia.es>
Author: Maria Pena Chilet <maria.pena.chilet.ext@juntadeandalucia.es>
Author: Marina Esteban <marina.estebanm@gmail.com>

Basic IO functionallity for Achilles cell line predictions.
"""

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

# LOAD USER ENVIRONMET VARIABLES
dotenv.load_dotenv(dotenv_file_path)
DATA_PATH = Path(os.environ.get("DATA_PATH"))
NUM_CPUS = int(os.environ.get("NUM_CPUS"))
USE_GPU = bool(os.environ.get("USE_GPU"))

genes_fname = "genes.rds.feather"
pathvals_fname = "expreset_pathvals.rds.feather"
expression_fname = "expreset_Hinorm.rds.feather"
circuits_fname = "circuits.rds.feather"


def load_circuits():

    circuits = feather.read_dataframe(DATA_PATH.joinpath(circuits_fname))
    circuits.set_index("index", drop=True, inplace=True)
    circuits.index = circuits.index.astype(str)
    circuits.replace({"FALSE": False, "TRUE": True}, inplace=True)

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

    return expression

def load_pathvals():

    pathvals = feather.read_dataframe(DATA_PATH.joinpath(pathvals_fname))
    pathvals.set_index("index", drop=True, inplace=True)

    return pathvals

def get_disease_data(disease):

    gene_expression, path_vals, metadata = [], [], []

    return gene_expression, path_vals, metadata