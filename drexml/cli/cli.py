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
import shutil
import subprocess
import sys
import warnings

import click
import joblib
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", module="shap", message="IPython could not be loaded!"
    )
    warnings.filterwarnings("ignore", module="shap", category=NumbaDeprecationWarning)
    warnings.filterwarnings(
        "ignore", module="shap", category=NumbaPendingDeprecationWarning
    )

from drexml.datasets import get_data
from drexml.plotting import RepurposingResult
from drexml.utils import (
    check_gputree_availability,
    get_number_cuda_devices,
    get_out_path,
    get_version,
)

FNAME_DICT = {
    "train": "stab_trainer.py",
    "explain": "stab_explainer.py",
    "score": "stab_scorer.py",
    "final": "stab_explainer.py",
}

STEPS = {
    "stab-train": {"previous": "orchestrate", "names": ["features.jbl", "target.jbl"]},
    "stab-explain": {"previous": "stab-train", "names": ["cv.jbl", "model_0.jbl"]},
    "stab-score": {"previous": "stab-explain", "names": ["fs.jbl"]},
    "explain": {"previous": "orchestrate", "names": ["features.jbl", "target.jbl"]},
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


_check_add_option = [
    click.option(
        "--add/--no-add",
        is_flag=True,
        default=True,
        help="Check the additivity when computing the SHAP values.",
    )
]

_verb_option = [
    click.option(
        "--verbosity/--no-verbosity",
        is_flag=True,
        default=False,
        help="Verbosity level.",
    )
]


def copy_files(ctx, fnames):
    """Copy files from tmp to ml folder."""
    for fname in fnames:
        shutil.copy(
            ctx["data_folder"].joinpath(fname).as_posix(),
            ctx["output_folder"].joinpath(fname).as_posix(),
        )


def add_options(options):
    """Add options to click command."""

    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func

    return _add_options


def get_cli_file(fname):
    """Get cli file path."""
    with pkg_resources.path("drexml.cli", fname) as f:
        data_file_path = f
    return pathlib.Path(data_file_path)


def build_ctx(ctx, step=None):
    """Generate context for command."""
    ctx_new = dict(ctx)
    output_folder = get_out_path(ctx_new["disease_path"])
    data_folder = output_folder.joinpath("tmp")
    ctx_new["output_folder"] = output_folder
    ctx_new["data_folder"] = data_folder
    data_folder.mkdir(parents=True, exist_ok=True)

    if step is not None:
        if step == "explain":
            ctx_new["mode"] = "final"

    if "n_gpus" in ctx_new.keys():
        if check_gputree_availability():
            if ctx_new["n_gpus"] < 0:  # pragma: no cover
                ctx_new["n_gpus"] = get_number_cuda_devices()
        else:
            ctx_new["n_gpus"] = 0  # pragma: no cover
    if "n_cpus" in ctx_new.keys():
        if ctx_new["n_cpus"] < 0:
            ctx_new["n_cpus"] = multiprocessing.cpu_count()

    if step is not None:
        previous_step = STEPS[step]["previous"]
        for fname in STEPS[step]["names"]:
            if not data_folder.joinpath(fname).exists():
                sys.exit(f"{previous_step} step is missing")

    return ctx_new


def build_cmd(ctx):
    """Generate command to launch"""
    script_path = get_cli_file(FNAME_DICT[ctx["mode"]]).as_posix()

    cmd = [
        "python",
        script_path,
        ctx["data_folder"].as_posix(),
        str(ctx["n_iters"]),
        str(int(ctx["n_gpus"])),
        str(ctx["n_cpus"]),
        str(int(ctx["debug"])),
        str(int(ctx["add"])),
        ctx["mode"],
    ]

    return cmd


def run_cmd(ctx):
    """Train/explain/score each stability partition"""
    cmd = build_cmd(ctx)
    # Unpythonic, update with dasks's LocalCudaCluster (currently unreliable).
    print(" ".join(cmd))
    try:
        output = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as err:  # pragma: no cover
        click.echo("Ping stdout output:\n", err.output)

    if ctx["verbosity"]:
        click.echo(output.stderr)
        click.echo(output.stdout)


@click.group()
@click.version_option(get_version())
def main():
    """drexml CLI entry point."""
    print(f"running drexml v{get_version()}")


@main.command()
@add_options(_debug_option)
@add_options(_verb_option)
@click.argument("disease-path", type=click.Path(exists=True))
@click.version_option(get_version())
def orchestrate(**kwargs):
    """Orchestrate the drexml procedure. Entry point for multi-disease workflows."""

    click.echo(f"running drexml explainer v {get_version()}")
    ctx = build_ctx(kwargs)

    # Load data
    gene_xpr, pathvals, _, _ = get_data(ctx["disease_path"], ctx["debug"])
    click.echo(ctx["data_folder"].joinpath("features.jbl"))
    joblib.dump(gene_xpr, ctx["data_folder"].joinpath("features.jbl"))
    joblib.dump(pathvals, ctx["data_folder"].joinpath("target.jbl"))


@main.command()
@click.option(
    "--mode",
    type=click.Choice(["train", "explain", "score"], case_sensitive=False),
)
@add_options(_debug_option)
@add_options(_verb_option)
@add_options(_n_iters_option)
@add_options(_n_gpus_option)
@add_options(_n_cpus_option)
@click.argument("disease-path", type=click.Path(exists=True))
@click.version_option(get_version())
def stability(**kwargs):
    """Run the stability analyses."""

    if kwargs["mode"].lower() == "train":
        current_step = "stab-train"
    elif kwargs["mode"].lower() == "explain":
        current_step = "stab-explain"
    elif kwargs["mode"].lower() == "score":
        current_step = "stab-score"

    click.echo(f"Running drexml {current_step} v {get_version()}")

    ctx = build_ctx(kwargs, step=current_step)

    run_cmd(ctx)

    if ctx["mode"].lower() == "score":
        fnames = ["stability_results.tsv", "stability_results_symbol.tsv"]
        copy_files(ctx, fnames)


@main.command()
@add_options(_debug_option)
@add_options(_verb_option)
@add_options(_check_add_option)
@add_options(_n_iters_option)
@add_options(_n_gpus_option)
@add_options(_n_cpus_option)
@click.argument("disease-path", type=click.Path(exists=True))
@click.version_option(get_version())
def explain(**kwargs):
    """Explain how KDTs modulate a given disease map."""
    ctx = build_ctx(kwargs, step="explain")

    run_cmd(ctx)

    fnames = [
        "shap_selection.tsv",
        "shap_summary.tsv",
        "shap_selection_symbol.tsv",
        "shap_summary_symbol.tsv",
    ]
    copy_files(ctx, fnames)


@main.command()
@add_options(_debug_option)
@add_options(_verb_option)
@add_options(_check_add_option)
@add_options(_n_iters_option)
@add_options(_n_gpus_option)
@add_options(_n_cpus_option)
@click.argument("disease-path", type=click.Path(exists=True))
@click.version_option(get_version())
@click.pass_context
def run(ctx, **kwargs):
    """Run the full procedure."""
    # ctx = build_ctx(kwargs, step=None)
    # orchestrate(kwargs["disease_path"], **orchestrate_ctx)
    ctx.forward(orchestrate)
    ctx.forward(stability, mode="train")
    ctx.forward(stability, mode="explain")
    ctx.forward(stability, mode="score")
    ctx.forward(explain)


@main.command()
@click.argument("sel-path", type=click.Path(exists=True))
@click.argument("score-path", type=click.Path(exists=True))
@click.argument("stability-path", type=click.Path(exists=True))
@click.argument("output-folder", type=click.Path(exists=True))
@click.option(
    "--gene",
    type=str,
    help="Gene (KDT) Symbol to plot its repurposing profile.",
)
@click.version_option(get_version())
@click.pass_context
def plot(ctx, sel_path, score_path, stability_path, output_folder, gene):
    """Plot the stability results"""

    results = RepurposingResult(
        sel_mat=sel_path, score_mat=score_path, stab_mat=stability_path
    )

    # Tests already covered in plotting.
    if gene:  # pragma: no cover
        try:
            results.plot_gene_profile(gene=gene, output_folder=output_folder)
        except KeyError as kerr:
            print(kerr)
            click.echo(f"Gene {gene} not in relevance matrix.")
        except Exception as e:
            print(e)

    else:
        try:
            results.plot_metrics(output_folder=output_folder)
        except Exception as e:  # pragma: no cover
            print(e)
            click.echo("skipping metrics plot.")

        for use_filter in [True, False]:
            try:
                results.plot_relevance_heatmap(
                    remove_unstable=use_filter, output_folder=output_folder
                )
            except Exception as e:  # pragma: no cover
                print(e)
                click.echo(
                    f"skipping relevance heatmap for filter set to: {use_filter}"
                )


if __name__ == "__main__":
    main()  # pragma: no cover
