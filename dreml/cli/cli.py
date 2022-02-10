#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Carlos Loucera <carlos.loucera@juntadeandalucia.es>
Author: Maria Pena Chilet <maria.pena.chilet.ext@juntadeandalucia.es>
Author: Marina Esteban <marina.estebanm@gmail.com>

Entry CLI point for stab.
"""

import importlib.resources as pkg_resources
import multiprocessing
import pathlib
import subprocess
import sys

import click
import joblib

from dreml.explain import compute_shap
from dreml.models import get_model
from dreml.utils import get_data, get_number_cuda_devices, get_out_path, get_version


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


FNAME_DICT = {
    "train": "stab_trainer.py",
    "explain": "stab_explainer.py",
    "score": "stab_scorer.py",
}

STEPS = {
    "stab-train": {"previous": "orchestrate", "names": ["features.jbl", "targets.jbl"]},
    "stab-explain": {"previous": "stab-train", "names": ["cv.jbl", "model_0.jbl"]},
    "stab-score": {"previous": "stab-explain", "name": ["stability_results_df.tsv"]},
    "explain": {"previous": "stab-score", "names": ["features.jbl", "targets.jbl"]},
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


def build_ctx(ctx, required_step=None):
    """Generate context for command."""
    ctx_new = dict(ctx)
    output_folder = get_out_path(ctx_new["disease_path"])
    data_folder = output_folder.joinpath("tmp")
    ctx_new["output_folder"] = output_folder
    ctx_new["data_folder"] = data_folder

    if "n_gpus" in ctx_new.keys():
        if ctx_new["n_gpus"] < 0:
            ctx_new["n_gpus"] = get_number_cuda_devices()
    if "n_cpus" in ctx_new.keys():
        if ctx_new["n_cpus"] < 0:
            ctx_new["n_cpus"] = multiprocessing.cpu_count()

    if required_step is not None:
        previous_step = STEPS[required_step]["previous"]
        for fname in STEPS[required_step]["names"]:
            if not data_folder.joinpath(fname).exists():
                sys.exit(f"{previous_step} step is missing")

    return ctx_new


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
@click.version_option(get_version())
def main():
    """DREML CLI entry point."""
    print(f"running DREML orchestrate v {get_version()}")


@main.command()
@add_options(_debug_option)
@click.argument("disease-path", type=click.Path(exists=True))
@click.version_option(get_version())
def orchestrate(**kwargs):
    """[summary]"""

    print(f"running DREML orchestrate v {get_version()}")
    ctx = build_ctx(kwargs)

    # Load data
    gene_xpr, pathvals, _, _ = get_data(kwargs["disease_path"], kwargs["debug"])
    joblib.dump(gene_xpr, ctx["data_folder"].joinpath("features.jbl"))
    joblib.dump(pathvals, ctx["data_folder"].joinpath("target.jbl"))


@main.command()
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
    if kwargs["mode"].lower() == "train":
        previous_step = "orchestrate"
    elif kwargs["mode"].lower() == "explain":
        previous_step = "stab-train"
    elif kwargs["mode"].lower() == "score":
        previous_step = "stab-explain"
    else:
        sys.exit("Unknown stability analysis step.")

    ctx = build_ctx(kwargs, required_step=previous_step)

    run_cmd(ctx)


@click.command()
@add_options(_debug_option)
@add_options(_n_iters_option)
@add_options(_n_gpus_option)
@add_options(_n_cpus_option)
@click.argument("disease-path", type=click.Path(exists=True))
@click.version_option(get_version())
@click.pass_context
def explainer(**kwargs):
    """[summary]"""
    ctx = build_ctx(kwargs, required_step="orchestrate")

    run_explainer(ctx)


if __name__ == "__main__":
    main()
