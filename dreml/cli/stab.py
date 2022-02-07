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

from dreml.utils import get_number_cuda_devices, get_out_path, get_version
from dreml.cli.stab_explainer import score_stab
import joblib

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources


FNAME_DICT = {
    "train": "stab_trainer.py",
    "explain": "stab_explainer.py",
    "score": "stab_scorer.py"
}

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
@click.option(
    "--mode",
    type=click.Choice(
        ["train", "explain", "score"],
        case_sensitive=False
    ),
)
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
@click.pass_context
def stability(ctx, disease_path, debug, n_iters, n_gpus, n_cpus, mode):
    """[summary]"""

    ctx.ensure_object(dict)
    ctx.obj["disease_path"] = disease_path
    ctx.obj["debug"] = debug
    ctx.obj["n_iters"] = n_iters
    ctx.obj["mode"] = mode

    click.echo(f"Running DREML stability v {get_version()}")
    output_folder = get_out_path(disease_path)
    data_folder = output_folder.joinpath("tmp")
    ctx.obj["data_folder"] = data_folder

    if n_gpus < 0:
        n_gpus = get_number_cuda_devices()
    click.echo(f"Using {n_gpus} GPU devices.")

    if n_cpus < 0:
        n_cpus = multiprocessing.cpu_count()
    click.echo(f"Using {n_cpus} CPU devices.")

    ctx.obj["n_gpus"] = n_gpus
    ctx.obj["n_cpus"] = n_cpus

    run_cmd(ctx)
