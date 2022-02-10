#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Carlos Loucera <carlos.loucera@juntadeandalucia.es>
Author: Maria Pena Chilet <maria.pena.chilet.ext@juntadeandalucia.es>
Author: Marina Esteban <marina.estebanm@gmail.com>

Entry CLI point for stab.
"""

import multiprocessing
import pathlib
import subprocess

import click
import joblib

from dreml.utils import get_data, get_number_cuda_devices, get_out_path, get_version

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

FNAME_DICT = {
    "train": "stab_trainer.py",
    "explain": "stab_explainer.py",
    "score": "stab_scorer.py",
}


_n_iters_option = [
    click.option(
        "--n-iters",
        default=0,
        type=int,
        help="Number of Optimization iterations. 0 means use sensible hyperparameters.",
    )
]
_n_gpus_option = [
    click.option(
        "--n-gpus",
        default=-1,
        type=int,
        help="Number of CUDA devices, -1 use all decices.",
    )
]
_n_cpus_option = [
    click.option(
        "--n-cpus",
        default=-1,
        type=int,
        help="Number of CPUs, -1 use all decices.",
    )
]

_overwrite_option = [
    click.option(
        "--overwrite",
        default=False,
        is_flag=True,
        help="Overwrite previous options.",
    )
]


_debug_option = [
    click.option(
        "--debug/--no-debug",
        is_flag=True,
        default=False,
        help="Flag to run in debug mode.",
    )
]


def add_options(options):
    """Add options to click command."""

    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func

    return _add_options


def get_cli_file(fname):
    """Get cli file path."""
    with pkg_resources.path("dreml.cli", fname) as f:
        data_file_path = f
    return pathlib.Path(data_file_path)


def build_cmd(ctx):
    """Generate command to launch"""
    script_path = get_cli_file(FNAME_DICT[ctx.obj["mode"]]).as_posix()

    cmd = [
        "python",
        script_path,
        ctx.obj["data_folder"],
        str(ctx.obj["n_iters"]),
        str(int(ctx.obj["n_gpus"])),
        str(ctx.obj["n_cpus"]),
        str(int(ctx.obj["debug"])),
    ]

    return cmd


def run_cmd(ctx):
    """Train/explain/score each stability partition"""
    cmd = build_cmd(ctx)
    # Unpythonic, update with daks's LocalCudaCluster (currently unreliable).
    subprocess.Popen(cmd).wait()


@click.command()
@add_options(_debug_option)
@click.argument("disease-path", type=click.Path(exists=True))
@click.version_option(get_version())
def orchestrate(disease_path, **kwargs):
    """[summary]"""

    print(f"running DREML orchestrate v {get_version()}")
    output_folder = get_out_path(disease_path)
    data_folder = output_folder.joinpath("tmp")
    data_folder.mkdir(parents=True, exist_ok=True)
    print(f"Tmp storage: {data_folder}")

    # Load data
    gene_xpr, pathvals, _, _ = get_data(disease_path, debug)
    joblib.dump(gene_xpr, data_folder.joinpath("features.jbl"))
    joblib.dump(pathvals, data_folder.joinpath("target.jbl"))


@click.command()
@click.option(
    "--mode",
    type=click.Choice(["train", "explain", "score"], case_sensitive=False),
)
@add_options(_debug_option)
@add_options(_n_iters_option)
@add_options(_n_gpus_option)
@add_options(_n_cpus_option)
@click.argument("disease-path", type=click.Path(exists=True))
@click.version_option(get_version())
def stability(**kwargs):
    """[summary]"""

    click.echo(f"Running DREML stability v {get_version()}")
    output_folder = get_out_path(kwargs["disease_path"])
    data_folder = output_folder.joinpath("tmp")
    kwargs["output_folder"] = output_folder
    kwargs["data_folder"] = data_folder

    if kwargs["n_gpus"] < 0:
        kwargs["n_gpus"] = get_number_cuda_devices()

    if kwargs["n_cpus"] < 0:
        kwargs["n_cpus"] = multiprocessing.cpu_count()

    run_cmd(kwargs)
