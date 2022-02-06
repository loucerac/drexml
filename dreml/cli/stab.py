#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Carlos Loucera <carlos.loucera@juntadeandalucia.es>
Author: Maria Pena Chilet <maria.pena.chilet.ext@juntadeandalucia.es>
Author: Marina Esteban <marina.estebanm@gmail.com>

Entry CLI point for stab.
"""

import multiprocessing
import os
import pathlib
import subprocess
from functools import partial

import click
import joblib
from sklearn.base import clone
from sklearn.model_selection import ShuffleSplit, train_test_split

from dreml.explain import compute_shap_fs, compute_shap_relevance, compute_shap_values
from dreml.models import get_model
from dreml.utils import get_number_cuda_devices, get_out_path, get_version

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources


def get_cli_file(fname):
    """Get cli file path."""
    with pkg_resources.path("dreml.cli", fname) as f:
        data_file_path = f
    return pathlib.Path(data_file_path)


@click.command()
@click.option(
    "--debug/--no-debug", is_flag=True, default=False, help="Flag to run in debug mode."
)
@click.option(
    "--n-iters",
    default=0,
    type=int,
    help="Number of Optimization iterations. 0 means use sensible hyperparameters.",
)
@click.option(
    "--n-gpus",
    default=-1,
    type=int,
    help="Number of CUDA devices, -1 use all decices.",
)
@click.option(
    "--n-cpus",
    default=-1,
    type=int,
    help="Number of CPUs, -1 use all decices.",
)
@click.argument("disease-path", type=click.Path(exists=True))
@click.version_option(get_version())
def stability(disease_path, debug, n_iters, n_gpus, n_cpus):
    """[summary]"""

    click.echo(f"Running DREML stability v {get_version()}")
    output_folder = get_out_path(disease_path)
    data_folder = output_folder.joinpath("tmp")

    if n_gpus < 0:
        n_gpus = get_number_cuda_devices()
    click.echo(f"Using {n_gpus} GPU devices.")

    if n_cpus < 0:
        n_cpus = multiprocessing.cpu_count()
    click.echo(f"Using {n_cpus} CPU devices.")

    # run_stab(data_folder, n_iters, n_gpus, n_cpus, debug)
    explain_fpath = get_cli_file("stab_runner.py").as_posix()

    cmd = [
        "python",
        explain_fpath,
        data_folder.as_posix(),
        str(n_iters),
        str(int(n_gpus)),
        str(n_cpus),
        str(int(debug)),
    ]
    subprocess.Popen(cmd).wait()
