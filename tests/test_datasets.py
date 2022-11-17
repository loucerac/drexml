# -*- coding: utf-8 -*-
"""
Unit testing for datasets module.
"""
import importlib.resources as pkg_resources
import shutil
import tempfile
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from drexml.cli.cli import main
from drexml.datasets import get_disease_data, load_df
from drexml.utils import check_gputree_availability

N_GPU_LST = [-1, 0] if check_gputree_availability() else [0]

DATA_NAMES = [
    "circuits.tsv.gz",
    "gene_exp.tsv.gz",
    "pathvals.tsv.gz",
    "genes.tsv.gz",
]


def get_resource_path(fname):
    """Get path to example disease env path.

    Returns
    -------
    pathlib.PosixPath
        Path to file.
    """
    with pkg_resources.path("drexml.resources", fname) as f:
        data_file_path = f
    return Path(data_file_path)


def make_disease_path(use_default, one=False):
    """Prepare fake disease data folder."""
    tmp_dir = Path(tempfile.mkdtemp())
    if use_default:
        disease_path_in = get_resource_path("experiment_default.env")
    else:
        if one:
            disease_path_in = get_resource_path("experiment1.env")
        else:
            disease_path_in = get_resource_path("experiment.env")
    disease_path_out = tmp_dir.joinpath(disease_path_in.name)
    # Use as_posix to make it compatible with python<=3.7
    shutil.copy(disease_path_in.as_posix(), disease_path_out.as_posix())

    with open(disease_path_out, "r", encoding="utf8") as this_file:
        disease_path_out_data = this_file.read()
    if not use_default:
        disease_path_out_data = disease_path_out_data.replace(
            "%THIS_PATH", disease_path_in.parent.as_posix()
        )
    with open(disease_path_out, "w", encoding="utf8") as this_file:
        this_file.write(disease_path_out_data)

    if use_default:
        shutil.copy(
            get_resource_path("circuits.tsv.gz").as_posix(),
            tmp_dir.joinpath("circuits.tsv.gz").as_posix(),
        )

    return disease_path_out


@pytest.mark.parametrize("fname", DATA_NAMES)
def test_load_df(fname):
    """Test load_df"""

    fpath = get_resource_path(fname)
    data = load_df(fpath)
    assert data.shape[0] > 0


@pytest.mark.parametrize("default", [True, False])
def get_disease_data_(default):
    """Test get_disease_data."""

    disease_path = make_disease_path(use_default=default)
    gene_exp, pathvals, circuits, genes = get_disease_data(disease_path, debug=True)

    assert gene_exp.to_numpy().ndim == 3
    assert pathvals.to_numpy().ndim == 2
    assert circuits.to_numpy().ndim == 2
    assert genes.to_numpy().ndim == 2


@pytest.mark.parametrize("n_gpus", N_GPU_LST)
def test_cli_run(n_gpus):
    """Unit tests for CLI app."""

    click.echo("Running CLI tests fro DREXML.")

    disease_path = make_disease_path(use_default=False, one=True)
    ml_folder_expected = disease_path.parent.joinpath("ml")

    opts = ["run", "--debug", f"--n-gpus {n_gpus}", f"{disease_path.as_posix()}"]
    click.echo(" ".join(opts))
    runner = CliRunner()
    runner.invoke(main, " ".join(opts))

    exist_files = [
        ml_folder_expected.joinpath(fname)
        for fname in ["stability_results.tsv", "shap_selection.tsv", "shap_summary.tsv"]
    ]

    assert all([x.exists() for x in exist_files])

    renamed_files = [
        ml_folder_expected.joinpath(f"{x.stem}_symbol.tsv") for x in exist_files
    ]

    opts = ["rename", f"{ml_folder_expected.as_posix()}"]
    click.echo(" ".join(opts))
    runner = CliRunner()
    runner.invoke(main, " ".join(opts))

    assert all([x.exists() for x in renamed_files])

    plot_files = [
        ml_folder_expected.joinpath(f"{x.stem}_symbol.png")
        for x in exist_files
        if "stability" in x.stem
    ]

    opts = [
        "plot",
        ml_folder_expected.joinpath("stability_results_symbol.tsv").as_posix(),
    ]
    click.echo(" ".join(opts))
    runner = CliRunner()
    runner.invoke(main, " ".join(opts))

    assert all([x.exists() for x in plot_files])
