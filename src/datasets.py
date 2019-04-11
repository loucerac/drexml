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


def load_metadata():
    """Load HORD metadata as a DataFrame.

    Returns
    -------
    pandas.DataFrame
        [n_samples x n_features] metadata DataFrame.
    """

    file_name = "sample_info.csv"
    fpath = os.path.join(RAW_PATH, file_name)
    metadata = pd.read_csv(fpath, index_col=0)

    return metadata


def get_disease_data(disease):

    gene_expression, path_vals, metadata = [], [], []

    return gene_expression, path_vals, metadata