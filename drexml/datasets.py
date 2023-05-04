# -*- coding: utf-8 -*-
"""
IO module for DREXML.
"""
import pathlib

import pandas as pd
from dotenv.main import dotenv_values
from pandas.errors import ParserError
from requests.exceptions import ConnectTimeout
from zenodo_client import Zenodo

DEFAULT_STR = "$default$"
DEBUG_NAMES = {
    "gene_exp": "gene_exp.tsv.gz",
    "pathvals": "pathvals.tsv.gz",
    "circuits": "circuits.tsv.gz",
    "genes": "genes.tsv.gz",
}

PRODUCTION_NAMES = {
    "gene_exp": "expreset_Hinorm_gtexV8.rds.feather",
    "pathvals": "expreset_pathvals_gtexV8.rds.feather",
    "genes": "genes.tsv.gz",
}

NAMES = {True: DEBUG_NAMES, False: PRODUCTION_NAMES}

RECORD_ID = "7737166"


def fetch_file(disease, key, version="latest", debug=False):
    """Retrieve data."""
    print(f"Retrieving {key}")
    experiment_env_path = pathlib.Path(disease)
    env = dotenv_values(experiment_env_path)
    if env[key].lower() == DEFAULT_STR:
        if version == "latest":
            try:
                zenodo = Zenodo()
                path = zenodo.download_latest(RECORD_ID, NAMES[debug][key], force=False)
            except (ConnectTimeout) as err:
                print(err)
                path = pathlib.Path.home().joinpath(
                    ".data", "zenodo", RECORD_ID, "20230315"
                )
    else:
        data_path = pathlib.Path(env["data_path"]).absolute()
        print(data_path)
        if data_path.name.lower() == DEFAULT_STR:
            print(disease, env[key], data_path)
            data_path = experiment_env_path.parent
        path = data_path.joinpath(env[key])

    return load_df(path, key)


def load_df(path, key=None):
    """Load dataframe from file. At the moment: stv, tsv crompressed or feather.

    Parameters
    ----------
    path : pathlib.Path
        Path to file.

    Returns
    -------
    pandas.DataFrame
        Dataframe.

    Raises
    ------
    NotImplementedError
        Not implemented yet.
    """
    try:
        # tsv, and compressed tsv
        res = pd.read_csv(path, sep="\t")
        if "index" in res.columns:
            res = res.set_index("index", drop=True)
    except (ParserError, KeyError, UnicodeDecodeError) as err:
        print("Error found while trying to load a TSV or compressed TSV.")
        print(err)
        res = pd.DataFrame()

        try:
            # feather
            res = pd.read_feather(path).set_index("index", drop=True)
        except (ParserError, KeyError) as new_err:
            print("Error found while trying to load a Feather.")
            print(new_err)
            res = pd.DataFrame()

    if res.shape[0] == 0:
        raise NotImplementedError("Format not implemented yet.")

    if key is not None:
        index_name_options = get_index_name_options(key)

        for name in index_name_options:
            if name in res.columns:
                res = res.set_index(name, drop=True)
        res.index = res.index.astype(str)

    return res


def get_index_name_options(key):

    if key == "circuits":
        return ["hipathia_id", "hipathia", "circuits_id", "index"]
    elif key == "genes":
        return ["entrezs", "entrez", "entrez_id", "index"]
    else:
        return ["index"]


def get_disease_data(disease, debug):
    """Get data for a disease.

    Parameters
    ----------
    disease : pathlib.Path
        Path to the disease configuration file.

    Returns
    -------
    pandas.DataFrame
        Gene expression data.
    pandas.DataFrame
        Circuit activation data (hipathia).
    pandas.DataFrame
        Circuit definition binary matrix.
    pandas.DataFrame
        KDT definition binary matrix.
    """

    # Load data
    experiment_env_path = pathlib.Path(disease)
    env = dotenv_values(experiment_env_path)
    genes_column = env["genes_column"]
    circuits_column = env["circuits_column"]

    gene_exp = fetch_file(disease, key="gene_exp", version="latest", debug=debug)
    gene_exp.columns = gene_exp.columns.str.replace("X", "")

    pathvals = fetch_file(disease, key="pathvals", version="latest", debug=debug)
    pathvals.columns = pathvals.columns.str.replace("-", ".").str.replace(" ", ".")
    circuits = fetch_file(disease, key="circuits", version="latest", debug=debug)
    circuits.index = circuits.index.str.replace("-", ".").str.replace(" ", ".")
    genes = fetch_file(disease, key="genes", version="latest", debug=debug)

    # gene_exp = gene_exp[genes.index[genes[genes_column]]]

    getx_entrez = gene_exp.columns
    this_genes = genes.index[genes[genes_column]]
    if this_genes.difference(getx_entrez).size > 0:
        print(f"# genes not present in GTEx: {this_genes.difference(getx_entrez).size}")

    usable_genes = this_genes.intersection(getx_entrez)

    gene_exp = gene_exp[usable_genes]
    pathvals = pathvals[circuits.index[circuits[circuits_column]]]

    return gene_exp, pathvals, circuits, genes
