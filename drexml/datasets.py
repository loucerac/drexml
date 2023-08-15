# -*- coding: utf-8 -*-
"""
IO module for DREXML.
"""
import pathlib

import pandas as pd
import pystow
from pandas.errors import ParserError
from requests.exceptions import ConnectTimeout
from zenodo_client import Zenodo

from drexml.utils import get_resource_path, read_disease_config

RECORD_ID = "6020480"


def load_disgenet():
    """Download if necessary and load the Disgenet curated list of gene-disease
    associations.

    Returns
    -------
    pd.DataFrame
        Disgenet curated dataset of gene-disease associations.
    """

    url = "/".join(
        [
            "https:/",
            "www.disgenet.org",
            "static",
            "disgenet_ap1",
            "files",
            "downloads",
            "curated_gene_disease_associations.tsv.gz",
        ]
    )

    disgenet: pd.DataFrame = pystow.ensure_csv(
        "drexml", "datasets", url=url, read_csv_kwargs={"sep": "\t"}
    )

    disgenet = disgenet.rename(
        columns={
            "geneId": "entrez_id",
            "diseaseId": "disease_id",
            "diseaseName": "disease_name",
            "score": "dga_score",
        }
    ).loc[:, ["disease_name", "disease_id", "entrez_id", "dga_score"]]

    return disgenet


def get_gda(disease_id, k_top=40):
    """Retrieve the list of genes associated to a disese according to the Disgenet
    curated list of gene-disease associations.

    Parameters
    ----------
    disease_id : str
        Disease ID.

    k_top: int
        Retrieve at most k_top genes based on the GDA score.

    Returns
    -------
    list
        List of gene IDs.
    """
    disgenet = load_disgenet()
    disgenet = disgenet.loc[disgenet["disease_id"] == disease_id]
    disgenet = disgenet.nlargest(k_top, "dga_score")

    return disgenet.entrez_id.astype(str).unique().tolist()


def load_physiological_circuits():
    """Load the list of physiological circuits.

    Returns
    -------
    list
        List of physiological circuit IDs.

    """
    fpath = get_resource_path("circuit_names.tsv.gz")
    circuit_names = pd.read_csv(fpath, sep="\t").set_index("circuit_id")
    circuit_names.index = circuit_names.index.str.replace("-", ".").str.replace(
        " ", "."
    )
    return circuit_names.index[circuit_names["is_physiological"]].tolist()


def fetch_file(key, env, version="latest"):
    """Retrieve file from the environment.

    Parameters
    ----------
    key : str
        Key of the file to retrieve.
    env : dict
        Environment.
    version : str
        Version of the file to retrieve.

    Returns
    -------
    pathlib.Path
        Path to the file.

    Raises
    ------
    NotImplementedError
        Not implemented yet.
    """

    print(f"Retrieving {key}")
    if env[key + "_zenodo"]:  # pragma: no cover
        if version == "latest":
            try:
                zenodo = Zenodo()
                path = zenodo.download_latest(RECORD_ID, env[key], force=False)
            except ConnectTimeout as err:
                print(err)
                path = pathlib.Path.home().joinpath(
                    ".data", "zenodo", RECORD_ID, "20230612"
                )
    else:
        path = env[key]

    print(key, path)

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
            raise new_err

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
        return ["hipathia_id", "hipathia", "circuits_id", "index", "circuit_id"]
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
        gene_list = []
        if env["seed_genes"]:
            gene_list += env["seed_genes"]
        if env["disease_id"]:
            gene_list += [str(gene) for gene in get_gda(env["disease_id"])]
        return preprocess_map(res, gene_list, env["circuits_column"], env["use_physio"])
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
    This function replaces hyphens and spaces in the column names of the input data
    frame with periods and returns the resulting data frame.

    """
    frame.columns = frame.columns.str.replace("-", ".").str.replace(" ", ".")
    return frame


def preprocess_map(frame, disease_seed_genes, circuits_column, use_physio):
    """
    Preprocesses a map data frame.

    Parameters
    ----------
    frame : pandas.DataFrame
        The map data frame to preprocess.
    disease_seed_genes : str
        The comma separated list of disease seed genes.
    circuits_column : str
        The name of the column containing circuit information.

    Returns
    -------
    list of str
        The list of circuits.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"in_disease": [True, False], "hipathia": ["A", "B"]})
    >>> preprocess_map(df, "A,B", "in_disease")
    ['A', 'B']

    Notes
    -----
    This function replaces hyphens and spaces in the index names of the input data frame
      with periods and returns the resulting list of circuits.
    """
    frame.index = frame.index.str.replace("-", ".").str.replace(" ", ".")
    if disease_seed_genes:
        print(disease_seed_genes)
        disease_seed_genes = frame.columns.intersection(disease_seed_genes)
        circuits = frame.index[frame[disease_seed_genes].any(axis=1)].tolist()
    else:
        frame[circuits_column] = frame[circuits_column].astype(bool)
        circuits = frame.index[frame[circuits_column]].tolist()

    if use_physio:
        physio_lst = load_physiological_circuits()
        circuits = [c for c in circuits if c in physio_lst]

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
    This function selects rows from the input data frame based on the values in the
    specified genes column and returns the resulting data frame.
    """

    frame = frame.loc[frame[genes_column]]
    return frame


def get_disease_data(disease):
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
    env = read_disease_config(experiment_env_path)

    gene_exp = fetch_file(key="gene_exp", env=env, version="latest")
    pathvals = fetch_file(key="pathvals", env=env, version="latest")
    circuits = fetch_file(key="circuits", env=env, version="latest")
    genes = fetch_file(key="genes", env=env, version="latest")

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


def get_data(disease, debug):
    """Load disease data and metadata.

    Parameters
    ----------
    disease : path-like
        Path to disease config file.
    debug : bool
        _description_, by default False.
    scale : bool, optional
        _description_, by default False.

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
    gene_xpr, pathvals, circuits, genes = get_disease_data(disease)

    print(gene_xpr.shape, pathvals.shape)

    if debug:
        size = 9
        gene_xpr = gene_xpr.sample(n=size, random_state=0)
        pathvals = pathvals.loc[gene_xpr.index, :]

    return gene_xpr, pathvals, circuits, genes
