#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Carlos Loucera <carlos.loucera@juntadeandalucia.es>
Author: Maria Pena Chilet <maria.pena.chilet.ext@juntadeandalucia.es>
Author: Marina Esteban <marina.estebanm@gmail.com>

Entry CLI point run everything.
"""

import multiprocessing

import click

from dreml.utils import get_number_cuda_devices, get_out_path, get_version


@click.command()
@click.option("-f", "--format-data", default="tsv.gz", type=str, help="Data format.")
@click.option(
    "--download/--no-download",
    is_flag=True,
    default=False,
    help="Download data from zenodo.",
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
def run(ctx, disease_path, debug, n_iters, n_gpus, n_cpus, mode):
    """Run the full procedure."""

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
