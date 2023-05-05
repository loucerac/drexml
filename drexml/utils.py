# -*- coding: utf-8 -*-
"""
Utilities module.
"""

import ctypes
import importlib.resources as pkg_resources
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd
from shap.utils import assert_import
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler

from drexml.datasets import get_disease_data
from drexml.models import get_model


def rename_results(folder):
    """Translate entrez to symbols, and KEGG circuit IDs to names."""
    folder = Path(folder)

    for path in folder.rglob("shap_selection*.tsv"):
        if "symbol" in path.stem:
            continue
        dataset = pd.read_csv(path, sep="\t", index_col=0)
        path_out = path.absolute().parent.joinpath(f"{path.stem}_symbol.tsv")
        dataset_out = convert_names(dataset, ["circuits", "genes"], axis=[0, 1])
        dataset_out.to_csv(path_out, sep="\t", index_label="circuit_name")

    for path in folder.rglob("shap_summary*.tsv"):
        if "symbol" in path.stem:
            continue
        dataset = pd.read_csv(path, sep="\t", index_col=0)
        path_out = path.absolute().parent.joinpath(f"{path.stem}_symbol.tsv")
        dataset_out = convert_names(dataset, ["circuits", "genes"], axis=[0, 1])
        dataset_out.to_csv(path_out, sep="\t", index_label="circuit_name")

    for path in folder.rglob("stability_results.tsv"):
        dataset = pd.read_csv(path, sep="\t", index_col=0)
        path_out = path.absolute().parent.joinpath(f"{path.stem}_symbol.tsv")
        dataset_out = convert_names(dataset, ["circuits"], axis=[0])
        dataset_out.to_csv(path_out, sep="\t", index_label="circuit_name")


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
    debug = bool(int(debug))

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
        Number of hyperparemeter optimization iterations.

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

    model = get_model(n_features, n_targets, n_cpus, debug, n_iters)

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
        raise NotImplementedError("Use experiment")

    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Storage folder: {out_path}")

    return out_path


def get_data(disease, debug, scale=False):
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
    gene_xpr, pathvals, circuits, genes = get_disease_data(disease, debug)

    if scale:

        pathvals = pd.DataFrame(
            MinMaxScaler().fit_transform(pathvals),
            columns=pathvals.columns,
            index=pathvals.index,
        )

    print(gene_xpr.shape, pathvals.shape)

    if debug:
        size = 9
        gene_xpr = gene_xpr.sample(n=size)
        pathvals = pathvals.loc[gene_xpr.index, :]

    return gene_xpr, pathvals, circuits, genes


def get_cuda_lib():
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


def get_number_cuda_devices():
    """Get number of CUDA devices."""

    try:
        n_gpus = get_number_cuda_devices_()
    except OSError as exc:
        print(exc)
        print("No CUDA devices found.")
        n_gpus = 0

    return n_gpus


def get_number_cuda_devices_():
    """Get number of CUDA devices."""

    cuda = get_cuda_lib()
    cuda.cuInit(0)
    n_gpus = ctypes.c_int()
    cuda.cuDeviceGetCount(ctypes.byref(n_gpus))

    return int(n_gpus.value)


def check_gputree_availability():
    """Check if GPUTree has been corectly compiled."""
    try:
        assert_import("cext_gpu")
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
    for i, key in enumerate(keys):
        if key == "circuits":
            fname = "circuit_names.tsv"
            index_name = "hipathia_id"
            col_name = "name"
        elif key == "genes":
            fname = "entrez_sym-table.tsv"
            index_name = "entrez"
            col_name = "symbol"
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
