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

import pandas as pd
import numpy as np

import dotenv


dotenv_filepath = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_filepath)
project_path = os.path.dirname(dotenv_filepath)

DATA_PATH = os.getenv("DATA_PATH")
NUM_CPUS = int(os.getenv("NUM_CPUS"))
USE_GPU = bool(os.getenv("USE_GPU"))


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