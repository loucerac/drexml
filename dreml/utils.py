# -*- coding: utf-8 -*-
"""
Author: Carlos Loucera <carlos.loucera@juntadeandalucia.es>
Author: Maria Pena Chilet <maria.pena.chilet.ext@juntadeandalucia.es>
Author: Marina Esteban <marina.estebanm@gmail.com>

Utilities module.
"""

import ctypes
from pathlib import Path

import pandas as pd
import pkg_resources
from sklearn.preprocessing import MinMaxScaler

from dreml.datasets import get_disease_data


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
    except Exception as exc:
        print(exc)
        print("No CUDA devices found.")
        n_gpus = 0

    return n_gpus


def get_number_cuda_devices_():
    """Get number of CUDA devices."""

    cuda = get_cuda_lib()
    cuda_query = cuda.cuInit(0)
    n_gpus = ctypes.c_int()
    cuda_query = cuda.cuDeviceGetCount(ctypes.byref(n_gpus))

    return int(n_gpus.value)
