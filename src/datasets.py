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

    gene_exp_fname = os.getenv("gene_exp")
    gene_exp_fpath = experiment_data_path.joinpath(gene_exp_fname)
    gene_exp = pd.read_feather(gene_exp_fpath)
    gene_exp = gene_exp.set_index("index", drop=True).replace("X", "")
    gene_exp.columns = gene_exp.columns.str.replace("X", "")

    pathvals_fname = os.getenv("pathvals")
    pathvals_fpath = experiment_data_path.joinpath(pathvals_fname)
    pathvals = pd.read_feather(pathvals_fpath)
    pathvals = pathvals.set_index("index", drop=True)
    pathvals.columns = pathvals.columns.str.replace("-", ".").str.replace(" ", ".")

    circuits_fname = os.getenv("circuits")
    circuits_fpath = experiment_data_path.joinpath(circuits_fname)
    circuits = pd.read_feather(circuits_fpath)
    circuits = circuits.set_index("index", drop=True)
    circuits.index = circuits.index.str.replace("-", ".").str.replace(" ", ".")

    genes_fname = os.getenv("genes")
    genes_fpath = experiment_data_path.joinpath(genes_fname)
    genes = pd.read_feather(genes_fpath)
    genes = genes.set_index("entrezs", drop=True)
    genes = genes.drop("index", axis=1)

    gene_exp = gene_exp[genes.index[genes["approved_targets"]]]
    pathvals = pathvals[circuits.index[circuits["in_disease"]]]

    return gene_exp, pathvals, circuits, genes
