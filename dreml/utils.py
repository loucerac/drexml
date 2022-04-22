# -*- coding: utf-8 -*-
"""
Utilities module.
"""

import ctypes
from pathlib import Path

import joblib
import pandas as pd
import pkg_resources
from shap.utils import assert_import
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler

from dreml.datasets import get_disease_data
from dreml.models import get_model


def parse_stab(argv):
    _, data_folder, n_iters, n_gpus, n_cpus, debug = argv
    n_iters = int(n_iters)
    data_folder = Path(data_folder)
    n_gpus = int(n_gpus)
    n_cpus = int(n_cpus)
    debug = bool(int(debug))

    n_splits = 5 if debug else 100

    return data_folder, n_iters, n_gpus, n_cpus, n_splits, debug


def get_stab(data_folder, n_splits, n_cpus, debug, n_iters):

    features_orig_fpath = data_folder.joinpath("features.jbl")
    features_orig = joblib.load(features_orig_fpath)

    targets_orig_fpath = data_folder.joinpath("target.jbl")
    targets_orig = joblib.load(targets_orig_fpath)

    stab_cv = ShuffleSplit(n_splits=n_splits, train_size=0.5, random_state=0)
    stab_cv = list(stab_cv.split(features_orig, targets_orig))
    stab_cv = [
        (*train_test_split(stab_cv[i][0], test_size=0.3), stab_cv[i][1])
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
    """Get DREML version."""
    return pkg_resources.get_distribution("dreml").version


def get_out_path(disease):
    """Construct the path where the model must be saved.

    Returns
    -------
    pathlib.Path
        The desired path.
    """

    env_possible = Path(disease)

    if env_possible.exists() and (env_possible.suffix == ".env"):
        print(f"Working with experiment {env_possible.parent.name}")
        out_path = env_possible.parent.joinpath("ml")
    else:
        raise NotImplementedError("Use experiment")

    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Storage folder: {out_path}")

    return out_path


def get_data(disease, debug, scale=True):
    """Load disease data and metadata."""
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
