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

from dreml.datasets import fetch_data
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


@click.group()
@click.option(
    "--download/--no-download",
    default=False,
    help="Download data from zenodo.",
)
@click.option(
    "--debug/--no-debug", is_flag=True, default=False, help="Flag to run in debug mode."
)
@click.version_option(get_version())
@click.argument("disease-path", type=click.Path(exists=True))
@click.pass_context
def main(ctx, disease_path, debug, download):
    """CLI entry point."""

    click.echo(f"Running DREML stability v {get_version()}")
    output_folder = get_out_path(disease_path)
    data_folder = output_folder.joinpath("tmp")

    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug
    ctx.obj["data_folder"] = data_folder
    ctx.obj["output_folder"] = output_folder

    if download:

        fetch_data()


@main.command("orchestrate")
@click.option("-f", "--format-data", default="tsv.gz", type=str, help="Data format.")
@click.version_option(get_version())
@click.pass_context
def orchestrate(ctx, format_data):
    """Orchestrate disease."""

    print(f"running DREML orchestrate v {get_version()}")
    # Load data
    gene_xpr, pathvals, _, _ = get_data(
        ctx.obj["disease_path"], ctx.obj["debug"], fmt=ctx.obj["format_data"]
    )
    joblib.dump(gene_xpr, ctx.obj["data_folder"].joinpath("features.jbl"))
    joblib.dump(pathvals, ctx.obj["data_folder"].joinpath("target.jbl"))


@main.command("run")
@click.option(
    "--mode",
    type=click.Choice(["train", "explain", "score"], case_sensitive=False),
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
@click.option("-f", "--format-data", default="tsv.gz", type=str, help="Data format.")
@click.version_option(get_version())
@click.pass_context
def run(ctx):
    """Run the full procedure."""


@main.command()
@click.option(
    "--mode",
    type=click.Choice(["train", "explain", "score"], case_sensitive=False),
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
def stability(ctx, disease_path, debug, n_iters, n_gpus, n_cpus, mode, overwrite):
    """[summary]"""

    ctx.obj["mode"] = mode

    click.echo(f"Running DREML stability v {get_version()}")
    output_folder = get_out_path(disease_path)
    data_folder = output_folder.joinpath("tmp")

    if n_gpus < 0:
        n_gpus = get_number_cuda_devices()
    click.echo(f"Using {n_gpus} GPU devices.")

    if n_cpus < 0:
        n_cpus = multiprocessing.cpu_count()
    click.echo(f"Using {n_cpus} CPU devices.")

    if overwrite:
        ctx.obj["disease_path"] = disease_path
        ctx.obj["debug"] = debug
        ctx.obj["n_iters"] = n_iters
        ctx.obj["data_folder"] = data_folder
        ctx.obj["n_gpus"] = n_gpus
        ctx.obj["n_cpus"] = n_cpus

    run_cmd(ctx)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
