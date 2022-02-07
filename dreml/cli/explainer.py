#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Carlos Loucera <carlos.loucera@juntadeandalucia.es>
Author: Maria Pena Chilet <maria.pena.chilet.ext@juntadeandalucia.es>
Author: Marina Esteban <marina.estebanm@gmail.com>

Entry CLI point for explainer.
"""

import multiprocessing

import click
import joblib

from dreml.explain import compute_shap
from dreml.models import get_model
from dreml.utils import get_number_cuda_devices, get_out_path, get_version


def run_explainer(ctx):
    """Run explainer."""

    use_gpu = ctx.obj["n_gpus"] > 0

    features_orig_fpath = ctx.obj["data_folder"].joinpath("features.jbl")
    features_orig = joblib.load(features_orig_fpath)

    targets_orig_fpath = ctx.obj["data_folder"].joinpath("target.jbl")
    targets_orig = joblib.load(targets_orig_fpath)

    n_features = features_orig.shape[1]
    n_targets = targets_orig.shape[1]

    estimator = get_model(
        n_features, n_targets, ctx.obj["n_cpus"], ctx.obj["debug"], ctx.obj["n_iters"]
    )

    # Compute shap relevances
    shap_summary, fs_df = compute_shap(estimator, features_orig, targets_orig, use_gpu)

    # Save results
    shap_summary_fname = "shap_summary.tsv"
    shap_summary_fpath = ctx.obj["output_folder"].joinpath(shap_summary_fname)
    shap_summary.to_csv(shap_summary_fpath, sep="\t")
    print(f"Shap summary results saved to: {shap_summary_fpath}")

    # Save results
    fs_fname = "shap_selection.tsv"
    fs_fpath = ctx.obj["output_folder"].joinpath(fs_fname)
    fs_df.to_csv(fs_fpath, sep="\t")
    print(f"Shap selection results saved to: {fs_fpath}")



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
@click.pass_context
def explainer(ctx, disease_path, debug, n_iters, n_gpus, n_cpus):
    """[summary]"""

    ctx.ensure_object(dict)
    ctx.obj["disease_path"] = disease_path
    ctx.obj["debug"] = debug
    ctx.obj["n_iters"] = n_iters

    click.echo(f"Running DREML stability v {get_version()}")
    output_folder = get_out_path(disease_path)
    data_folder = output_folder.joinpath("tmp")
    ctx.obj["output_folder"] = output_folder
    ctx.obj["data_folder"] = data_folder

    if n_gpus < 0:
        n_gpus = get_number_cuda_devices()
    click.echo(f"Using {n_gpus} GPU devices.")

    if n_cpus < 0:
        n_cpus = multiprocessing.cpu_count()
    click.echo(f"Using {n_cpus} CPU devices.")

    ctx.obj["n_gpus"] = n_gpus
    ctx.obj["n_cpus"] = n_cpus

    run_explainer(ctx)

