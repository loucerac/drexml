# -*- coding: utf-8 -*-
"""
Utilities module.
"""

import ctypes
import importlib.resources as pkg_resources
import warnings
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd
import pystow
import requests
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", module="shap", message="IPython could not be loaded!"
    )
    warnings.filterwarnings("ignore", module="shap", category=NumbaDeprecationWarning)
    warnings.filterwarnings(
        "ignore", module="shap", category=NumbaPendingDeprecationWarning
    )
    import shap

from dotenv.main import dotenv_values
from sklearn.model_selection import ShuffleSplit, train_test_split

from drexml.config import DEFAULT_DICT, VERSION_DICT
from drexml.models import get_model

DEFAULT_STR = "$default$"


def check_cli_arg_is_bool(arg):
    """Check if argument is a boolean.

    Parameters
    ----------
    arg : str
        Argument.

    Returns
    -------
    bool
        Argument.
    """
    if arg in ["true", "True", "TRUE", "1"]:
        arg = True
    elif arg in ["false", "False", "FALSE", "0"]:
        arg = False
    else:
        raise ValueError(f"argument {arg} must be a boolean")

    return arg


def parse_stab(argv):
    """Parse stab arguments.
    Parameters
    ----------
    argv : list
        List of arguments.

    Returns
    -------
    path-like
        Path to data folder.
    int
        Number of hyperparameter optimizations.
    int
        Number of GPUs.
    int
        Number of CPUs.
    int
        Number of splits.
    bool
        Debug flag.
    """
    _, data_folder, n_iters, n_gpus, n_cpus, debug, add, mode = argv
    n_iters = int(n_iters)
    data_folder = Path(data_folder)
    n_gpus = int(n_gpus)
    n_cpus = int(n_cpus)
    debug = check_cli_arg_is_bool(debug)
    add = check_cli_arg_is_bool(add)

    if mode == "final":
        n_splits = 1
    else:
        n_splits = 3 if debug else 100

    return data_folder, n_iters, n_gpus, n_cpus, n_splits, debug, add


def get_stab(data_folder, n_splits, n_cpus, debug, n_iters):
    """Get stab data.

    Parameters
    ----------
    data_folder : path-like
        Path to data folder.
    n_splits : int
        Number of splits.
    n_cpus : int
        Number of CPUs.
    debug : bool
        Debug flag, by default False.
    n_iters : int
        Number of hyperparameter optimization iterations.

    Returns
    -------
    drexml.models.Model
        Model.
    list
        List of splits.
    panda.DataFrame
        Gene expression data.
    panda.DataFrame
        Circuit activation data (hipathia).
    """
    features_orig_fpath = data_folder.joinpath("features.jbl")
    features_orig = joblib.load(features_orig_fpath)

    use_imputer = features_orig.isna().any(axis=None)

    targets_orig_fpath = data_folder.joinpath("target.jbl")
    targets_orig = joblib.load(targets_orig_fpath)

    stab_cv = ShuffleSplit(n_splits=n_splits, train_size=0.5, random_state=0)
    stab_cv = list(stab_cv.split(features_orig, targets_orig))
    stab_cv = [
        (*train_test_split(stab_cv[i][0], test_size=0.3, random_state=i), stab_cv[i][1])
        for i in range(n_splits)
    ]

    name = "cv.jbl"
    path = data_folder.joinpath(name)
    joblib.dump(stab_cv, path)
    n_features = features_orig.shape[1]
    n_targets = targets_orig.shape[1]

    model = get_model(
        n_features, n_targets, n_cpus, debug, n_iters, use_imputer=use_imputer
    )

    return model, stab_cv, features_orig, targets_orig


def get_version():
    """Get drexml version."""
    return version("drexml")


def get_out_path(disease):
    """Construct the path where the model must be saved.

    Returns
    -------
    pathlib.Path
        The desired path.
    """

    env_possible = Path(disease).absolute()

    if env_possible.exists() and (env_possible.suffix == ".env"):
        print(f"Working with experiment {env_possible.parent.name}")
        out_path = env_possible.parent.joinpath("results")
    else:
        raise NotImplementedError("Error loading a .env describing the experiment")

    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Storage folder: {out_path}")

    return out_path


def get_cuda_lib():  # pragma: no cover
    """Get CUDA library name."""
    lib_names = ("libcuda.so", "libcuda.dylib", "cuda.dll")
    for lib_name in lib_names:
        try:
            cuda = ctypes.CDLL(lib_name)
        except OSError:
            continue
        else:
            break
    else:
        raise OSError("could not load any of: " + " ".join(lib_names))

    return cuda


def get_number_cuda_devices():  # pragma: no cover
    """Get number of CUDA devices."""

    try:
        n_gpus = get_number_cuda_devices_()
    except OSError as exc:
        print(exc)
        print("No CUDA devices found.")
        n_gpus = 0

    return n_gpus


def get_number_cuda_devices_():  # pragma: no cover
    """Get number of CUDA devices."""

    cuda = get_cuda_lib()
    cuda.cuInit(0)
    n_gpus = ctypes.c_int()
    cuda.cuDeviceGetCount(ctypes.byref(n_gpus))

    return int(n_gpus.value)


def get_cuda_version():  # pragma: no cover
    """Get CUDA version."""
    cuda = get_cuda_lib()
    cuda.cuInit(0)
    cc_major = ctypes.c_int()
    cc_minor = ctypes.c_int()
    cuda.cuDeviceComputeCapability(ctypes.byref(cc_major), ctypes.byref(cc_minor), 0)

    return int(cc_major.value), int(cc_minor.value)


def check_gputree_availability():  # pragma: no cover
    """Check if GPUTree has been correctly compiled."""
    try:
        shap.utils.assert_import("cext_gpu")
        return True
    except ImportError as ierr:
        print(ierr)
        return False


def get_resource_path(fname):
    """Get path to example disease env path.
    Returns
    -------
    pathlib.PosixPath
        Path to file.
    """
    with pkg_resources.path("drexml.resources", fname) as f:
        data_file_path = f
    return Path(data_file_path)


def convert_names(dataset, keys, axis):
    """
    Convert names in the dataset.

    Parameters
    ----------
    dataset : pandas.DataFrame
        Dataset.
    keys : list
        List of keys.
    axis : list
        List of axis.

    Returns
    -------
    panda.DataFrame
        Dataset.

    Raises
    ------
    NotImplementedError
        If key is not supported.

    Examples
    --------
    >>> dataset = pd.DataFrame({"circuits": ["C1", "C2"], "genes": [1, 2]})
    >>> keys = ["circuits", "genes"]
    >>> axis = [0, 1]
    >>> convert_names(dataset, keys, axis)
       circuits  genes
    0        C1      1
    1        C2      2

    >>> dataset = pd.DataFrame({"circuits": ["C1", "C2"], "genes": [1, 2]})
    >>> keys = ["circuits", "genes"]
    >>> axis = [0, 1]
    >>> convert_names(dataset, keys, axis)
       circuits  genes
    0        C1      1
    1        C2      2

    """
    for i, key in enumerate(keys):
        if key == "circuits":
            fname = "circuit_names.tsv.gz"
            index_name = "circuit_id"
            col_name = "circuit_name"
        elif key == "genes":
            fname = "genes_drugbank-v050110_gtex-v8_mygene-v20230220.tsv.gz"
            index_name = "entrez_id"
            col_name = "symbol_id"
        else:
            raise NotImplementedError()

        name_dict = pd.read_csv(get_resource_path(fname), sep="\t").set_index(
            index_name
        )
        name_dict.index = name_dict.index.astype(str)
        if key == "circuits":
            name_dict.index = name_dict.index.str.replace("-", ".").str.replace(
                " ", "."
            )
        name_dict = name_dict[col_name].to_dict()

        dataset = dataset.rename(name_dict, axis=axis[i])

    return dataset


def read_disease_id(config):
    """Read disease id from config file. It expects a disease id using the UMLS.

    Parameters
    ----------
    config : dict
        Parsed config dict.

    Returns
    -------
    dict
        Updated config dict.

    Raises
    ------
    ValueError
        Raise error if format is unsupported.
    """
    try:
        if config["disease_id"] is not None:
            config["disease_id"] = str(config["disease_id"])
    except ValueError as err:
        print(err)
        print("seed_genes should be a comma-separated list of genes.")
        raise

    return config


def read_seed_genes(config):
    """Read seed genes from config file. It expect a comma-separated list of entrez ids.

    Parameters
    ----------
    config : dict
        Parsed config dict.

    Returns
    -------
    dict
        Updated config dict.

    Raises
    ------
    ValueError
        Raise error if format is unsupported.
    """
    try:
        if config["seed_genes"] is not None:
            config["seed_genes"] = str(config["seed_genes"]).split(",")
            for int_str in config["seed_genes"]:
                int_str = int_str.strip()
                if not int_str.isdigit():
                    raise ValueError(f"{int_str} is not a valid integer.")
    except ValueError as err:
        print(err)
        print("seed_genes should be a comma-separated list of genes.")
        raise

    return config


def read_use_physio(config):
    """Read use_physio from config file. It expects a boolean.

    Parameters
    ----------
    config : dict
        Parsed config dict.

    Returns
    -------
    dict
        Updated config dict.

    Raises
    ------
    ValueError
        Raise error if format is unsupported.
    """

    try:
        config["use_physio"] = check_cli_arg_is_bool(config["use_physio"])
    except ValueError as err:
        print(err)
        print("use_physio should be a boolean.")
        raise

    return config


def read_activity_normalizer(config):
    """Read activity_normalizer from config file. It expects a boolean.

    Parameters
    ----------
    config : dict
        Parsed config dict.

    Returns
    -------
    dict
        Updated config dict.

    Raises
    ------
    ValueError
        Raise error if format is unsupported.
    """

    try:
        config["activity_normalizer"] = check_cli_arg_is_bool(
            config["activity_normalizer"]
        )
    except ValueError as err:
        print(err)
        print("activity_normalizer should be a boolean.")
        raise

    return config


def read_path_based(config, key, data_path):
    """Read path based.

    Parameters
    ----------
    config : dict
        Config dict.
    key : str
        Key in config dict.
    data_path : path-like
        Storage path.

    Returns
    -------
    dict
        Updated config dict.

    Raises
    ------
    ValueError
        Raise error if key is not present in config dict.
    FileNotFoundError
        Raise error if path does not exist.
    """
    try:
        if config[key] is not None:
            path = data_path.joinpath(config[key])
            print("here")
            if not path.exists():
                print("inside")
                print(config[key])
                path = Path(config[key]).expanduser()
            print(path)
            config[key] = path
            with open(path, "r", encoding="utf-8") as _:
                pass

    except (ValueError, FileNotFoundError) as err:
        print(err)
        print(f"{key} should be a path.")
        raise err

    return config


def read_circuits_column(config):
    """Read circuits column.

    Parameters
    ----------
    config : dict
        Config dict.

    Returns
    -------
    dict
        Updated config dict.

    Raises
    ------
    ValueError
        Raise error if format is unsupported.
    """
    try:
        config["circuits_column"] = str(config["circuits_column"])
        if not config["circuits_column"]:
            raise ValueError(f"{config['circuits_column']} is not alphanumeric.")
    except ValueError as err:
        print(err)
        print("circuits_column should be string-like.")
        raise

    return config


def read_version_based(config, key, version_dict):
    """Read version based.

    Parameters
    ----------
    config : dict
        Config dict.
    key : str
        Key in config dict.
    version_dict : dict
        Version dict.

    Returns
    -------
    dict
        Updated config dict.

    Raises
    ------
    ValueError
        Raise error if format is unsupported.
    """

    try:
        config[key] = str(config[key])
        if config[key] not in version_dict[key]:
            raise ValueError(f"{key} should be one of {version_dict[key]}.")
    except ValueError as err:
        print(err)
        print("{key} should be one of {version_dict[key]}.")
        raise

    return config


def read_disease_config(disease):
    """Read disease config file.

    Parameters
    ----------
    disease : str
        Path to disease config file.

    Returns
    -------
    dict
        Config dictionary.

    """

    # TODO: when moving to Python >= 3.9 use '|' to update dicts
    config = dotenv_values(disease)
    env_parent_path = Path(disease).parent

    for key, _ in DEFAULT_DICT.items():
        if key not in config:
            config[key] = DEFAULT_DICT[key]

    config = read_seed_genes(config)
    config = read_disease_id(config)
    config = read_use_physio(config)
    config = read_activity_normalizer(config)
    config = read_path_based(config, key="gene_exp", data_path=env_parent_path)
    config = read_path_based(config, key="pathvals", data_path=env_parent_path)
    config = read_path_based(config, key="genes", data_path=env_parent_path)
    config = read_path_based(config, key="circuits", data_path=env_parent_path)
    config = read_circuits_column(config)

    config = read_version_based(config, key="GTEX_VERSION", version_dict=VERSION_DICT)
    config = read_version_based(config, key="MYGENE_VERSION", version_dict=VERSION_DICT)
    config = read_version_based(
        config, key="DRUGBANK_VERSION", version_dict=VERSION_DICT
    )
    config = read_version_based(
        config, key="HIPATHIA_VERSION", version_dict=VERSION_DICT
    )
    config = read_version_based(config, key="EDGER_VERSION", version_dict=VERSION_DICT)

    config = update_gene_exp(config)
    config = update_pathvals(config)
    config = update_genes(config)
    config = update_circuits(config)

    return config


def build_gene_exp_fname(config):
    """Build gene_exp filename.

    Parameters
    ----------
    config : dict
        Config dict.

    Returns
    -------
    str
        Filename.
    """

    return (
        "_".join(
            [
                "gexp",
                f"gtex-{config['GTEX_VERSION']}",
                f"edger-{config['EDGER_VERSION']}",
            ]
        )
        + ".feather"
    )


def build_pathvals_fname(config):
    """Build pathvals filename.

    Parameters
    ----------
    config : dict
        Config dict.

    Returns
    -------
    str
        Filename.

    """
    if config["activity_normalizer"]:
        hipathia_str = f"hipathia-norm-{config['HIPATHIA_VERSION']}"
    else:
        hipathia_str = f"hipathia-{config['HIPATHIA_VERSION']}"

    print(hipathia_str)

    return (
        "_".join(
            [
                "pathvals",
                f"gtex-{config['GTEX_VERSION']}",
                f"edger-{config['EDGER_VERSION']}",
                hipathia_str,
            ]
        )
        + ".feather"
    )


def build_genes_fname(config):
    """Build genes filename.

    Parameters
    ----------
    config : dict
        Config dict.

    Returns
    -------
    str
        Filename.

    """

    return (
        "_".join(
            [
                "genes",
                f"drugbank-{config['DRUGBANK_VERSION']}",
                f"gtex-{config['GTEX_VERSION']}",
                f"mygene-{config['MYGENE_VERSION']}",
            ]
        )
        + ".tsv.gz"
    )


def build_circuits_fname(config):
    """Build circuits filename.

    Parameters
    ----------
    config : dict
        Config dict.

    Returns
    -------
    str
        Filename.

    """

    return (
        "_".join(
            [
                "circuits2genes",
                f"gtex-{config['GTEX_VERSION']}",
                f"hipathia-{config['HIPATHIA_VERSION']}",
            ]
        )
        + ".tsv.gz"
    )


def update_gene_exp(config):
    """Update gene_exp key from config.

    Parameters
    ----------
    config : dict
        Config dict.

    Returns
    -------
    dict
        Updated config dict.

    Raises
    ------
    ValueError
        Raise error if format is unsupported.

    Notes
    -----
    If gene_exp is not provided, it will be built from the other keys.

    If gene_exp is provided, it will be checked if it is a path.

    If gene_exp is a path, it will be checked if it is a zenodo resource.

    """
    if config["gene_exp"] is None:
        config["gene_exp"] = build_gene_exp_fname(config)
        config["gene_exp_zenodo"] = True

    return config


def update_pathvals(config):
    """Update pathvals key from config.

    Parameters
    ----------
    config : dict
        Config dict.

    Returns
    -------
    dict
        Updated config dict.

    Raises
    ------
    ValueError
        Raise error if format is unsupported.

    Notes
    -----
    If pathvals is not provided, it will be built from the other keys.

    If pathvals is provided, it will be checked if it is a path.

    If pathvals is a path, it will be checked if it is a zenodo resource.

    """
    if config["pathvals"] is None:
        config["pathvals"] = build_pathvals_fname(config)
        config["pathvals_zenodo"] = True

    return config


def update_genes(config):
    """Update genes key from config.

    Parameters
    ----------
    config : dict
        Config dict.

    Returns
    -------
    dict
        Updated config dict.

    Raises
    ------
    ValueError
        Raise error if format is unsupported.

    Notes
    -----
    If genes is not provided, it will be built from the other keys.

    If genes is provided, it will be checked if it is a path.

    If genes is a path, it will be checked if it is a zenodo resource.

    """
    if config["genes"] is None:
        if (
            (config["GTEX_VERSION"] != DEFAULT_DICT["GTEX_VERSION"])
            or (config["DRUGBANK_VERSION"] != DEFAULT_DICT["DRUGBANK_VERSION"])
            or (config["MYGENE_VERSION"] != DEFAULT_DICT["MYGENE_VERSION"])
        ):
            config["genes_zenodo"] = True
            config["genes"] = build_genes_fname(config)
    else:
        if config["genes"].exists():
            config["genes_zenodo"] = False
            return config

    if not config["genes_zenodo"]:
        config["genes"] = get_resource_path(build_genes_fname(config))

    return config


def update_circuits(config):
    """Update circuits key from config.

    Parameters
    ----------
    config : dict
        Config dict.

    Returns
    -------
    dict
        Updated config dict.

    Raises
    ------
    ValueError
        Raise error if format is unsupported.

    Notes
    -----
    If circuits is not provided, it will be built from the other keys.

    If circuits is provided, it will be checked if it is a path.

    If circuits is a path, it will be checked if it is a zenodo resource.

    """

    # build circuits_dict
    def build_circuits_dict_path(config, key):
        if (config["GTEX_VERSION"] != DEFAULT_DICT["GTEX_VERSION"]) or (
            config["HIPATHIA_VERSION"] != DEFAULT_DICT["HIPATHIA_VERSION"]
        ):
            config[key] = build_circuits_fname(config)
            config["circuits_zenodo"] = True
        else:
            config[key] = get_resource_path(
                "circuits2genes_gtex-v8_hipathia-v2-14-0.tsv.gz"
            )

        return config

    if config["circuits"] is None:
        if config["seed_genes"] is None and config["disease_id"] is None:
            raise ValueError("Provide on of circuits, disease_id or gene_seeds.")
        config = build_circuits_dict_path(config, key="circuits")

    config = build_circuits_dict_path(config, key="circuits_dict")

    return config


def get_latest_record(record_id):
    """Get latest zenodo record ID from a given deposition identifier

    Parameters
    ----------
    record_id : str
        deposition identifier

    Returns
    -------
    str
        latest record ID

    """

    url = requests.get(f"https://zenodo.org/records/{record_id}", timeout=10).url
    return url.split("/")[-1]


def ensure_zenodo(name, record_id="6020480"):
    """Ensure file availability and download it from zenodo

    Parameters
    ----------
    name : str
        file name
    record_id : str
        deposition identifier

    Returns
    -------
    path : path-like
        PosixPath to downloaded file

    """

    record_id = get_latest_record(record_id)

    url = f"https://zenodo.org/records/{record_id}/files/{name}?download=1"

    path = pystow.ensure("drexml", "datasets", record_id, url=url)

    return path
