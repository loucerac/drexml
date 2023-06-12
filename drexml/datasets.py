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
    "circuits": "circuits_to_genes.tsv.gz"
}

NAMES = {True: DEBUG_NAMES, False: PRODUCTION_NAMES}

RECORD_ID = "7737166"


def fetch_file(disease, key, env, version="latest", debug=False):
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
                    ".data", "zenodo", RECORD_ID, "20230612"
                )
    else:
        data_path = pathlib.Path(env["data_path"]).absolute()
        print(data_path)
        if data_path.name.lower() == DEFAULT_STR:
            print(disease, env[key], data_path)
            data_path = experiment_env_path.parent
        path = data_path.joinpath(env[key])

    frame = load_df(path, key)
    frame = preprocess_frame(frame, env, key)

    return frame


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

    return res


def get_index_name_options(key):
    """
    Returns a list of possible index names based on the input key.

    Parameters
    ----------
    key : str
        The key for the data frame.

    Returns
    -------
    list of str
        A list of possible index names based on the input key.

    Examples
    --------
    >>> get_index_name_options("circuits")
    ["hipathia_id", "hipathia", "circuits_id", "index"]

    Notes
    -----
    This function returns a list of possible index names based on the input key. If the
    key is "circuits", it returns a list of four possible index names. If the key is
    "genes", it returns a list of three possible index names. Otherwise, it returns a
    list with only one element, "index".
    """

    if key == "circuits":
        return ["hipathia_id", "hipathia", "circuits_id", "index"]
    elif key == "genes":
        return ["entrezs", "entrez", "entrez_id", "index"]
    else:
        return ["index"]


def preprocess_frame(res, env, key):
    """
    Preprocesses the input data frame.

    Parameters
    ----------
    res : pandas.DataFrame
        The input data frame.
    env : dict
        The environment variables.
    key : str
        The key for the data frame.

    Returns
    -------
    pandas.DataFrame
        The preprocessed data frame.
    """

    if key is not None:
        index_name_options = get_index_name_options(key)

    for name in index_name_options:
        if name in res.columns:
            res = res.set_index(name, drop=True)
    res.index = res.index.astype(str)

    if key == "gene_exp":
        return preprocess_gexp(res)
    elif key == "pathvals":
        return preprocess_activities(res)
    elif key == "circuits":
        return preprocess_map(res, env["disease_seed_genes"], env["circuits_column"])
    elif key == "genes":
        return preprocess_genes(res, env["genes_column"])


def preprocess_gexp(frame):
    """
    Preprocesses a gene expression data frame.

    Parameters
    ----------
    frame : pandas.DataFrame
        The gene expression data frame to preprocess.

    Returns
    -------
    pandas.DataFrame
        The preprocessed gene expression data frame.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"X1": [1, 2], "X2": [3, 4]})
    >>> preprocess_gexp(df)
       1  2
    0  1  3
    1  2  4

    Notes
    -----
    This function removes the "X" prefix from the column names of the input data frame
    and returns the resulting data frame.
    """

    frame.columns = frame.columns.str.replace("X", "")
    return frame


def preprocess_activities(frame):
    """
    Preprocesses an activities data frame.

    Parameters
    ----------
    frame : pandas.DataFrame
        The activities data frame to preprocess.

    Returns
    -------
    pandas.DataFrame
        The preprocessed activities data frame.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"-": [1, 2], "Activity 1": [3, 4]})
    >>> preprocess_activities(df)
       .  Activity.1
    0  1          3
    1  2          4

    Notes
    -----
    This function replaces hyphens and spaces in the column names of the input data frame with periods and returns the resulting data frame.

    """
    frame.columns = frame.columns.str.replace("-", ".").str.replace(" ", ".")
    return frame


def preprocess_map(frame, disease_seed_genes, circuits_column):
    """
    Preprocesses a map data frame.

    Parameters
    ----------
    frame : pandas.DataFrame
        The map data frame to preprocess.
    circuits_column : str
        The name of the column containing circuit information.

    Returns
    -------
    pandas.DataFrame
        The preprocessed map data frame.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"-": [1, 2], "Activity 1": [3, 4]}, index=["A-B", "C-D"])
    >>> preprocess_map(df, "Activity 1")
       .  Activity.1
    A.B          3
    C.D          4

    Notes
    -----
    This function replaces hyphens and spaces in the index labels of the input data
    frame with periods and converts the values in the specified circuits column to
    boolean values. It then returns the resulting data frame.

    """
    frame.index = frame.index.str.replace("-", ".").str.replace(" ", ".")
    if disease_seed_genes != DEFAULT_STR:
        gene_seeds = disease_seed_genes.split(",")
        gene_seeds = frame.columns.intersection(gene_seeds)
        circuits = frame.index[frame[gene_seeds].any(axis=1)].tolist()
    else:   
        if circuits_column == DEFAULT_STR:
            circuits_column = "in_disease"
        frame[circuits_column] = frame[circuits_column].astype(bool)
        circuits = frame.index[frame[circuits_column]].tolist()

    return circuits


def preprocess_genes(frame, genes_column):
    """
    Preprocesses a gene expression data frame.

    Parameters
    ----------
    frame : pandas.DataFrame
        The gene expression data frame to preprocess.
    genes_column : str
        The name of the column containing gene information.

    Returns
    -------
    pandas.DataFrame
        The preprocessed gene expression data frame.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"Gene": ["A", "B", "C"], "Value": [1, 2, 3]})
    >>> preprocess_genes(df, "Gene")
      Gene  Value
    0    A      1
    1    B      2
    2    C      3

    Notes
    -----
    This function selects rows from the input data frame based on the values in the specified genes column and returns the resulting data frame.

    """
    if genes_column == DEFAULT_STR:
        genes_column = "drugbank_approved_targets"
    frame = frame.loc[frame[genes_column]]
    return frame


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

    gene_exp = fetch_file(
        disease, key="gene_exp", env=env, version="latest", debug=debug
    )
    pathvals = fetch_file(
        disease, key="pathvals", env=env, version="latest", debug=debug
    )
    circuits = fetch_file(
        disease, key="circuits", env=env, version="latest", debug=debug
    )
    genes = fetch_file(disease, key="genes", env=env, version="latest", debug=debug)

    # gene_exp = gene_exp[genes.index[genes[genes_column]]]

    gtex_entrez = gene_exp.columns
    gene_diff = genes.index.difference(gtex_entrez).size
    if gene_diff > 0:
        print(f"# genes not present in GTEx: {gene_diff}")

    usable_genes = genes.index.intersection(gtex_entrez)

    gene_exp = gene_exp[usable_genes]
    pathvals = pathvals[circuits]

    print(pathvals.shape)

    return gene_exp, pathvals, circuits, genes
