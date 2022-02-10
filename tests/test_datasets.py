# -*- coding: utf-8 -*-
"""
Author: Carlos Loucera <carlos.loucera@juntadeandalucia.es>
Author: Maria Pena Chilet <maria.pena.chilet.ext@juntadeandalucia.es>
Author: Marina Esteban <marina.estebanm@gmail.com>

Unit testing for datasets module.
"""
import shutil
import tempfile
from pathlib import Path

import joblib
import pytest

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

import click
from click.testing import CliRunner

from dreml.cli.explainer import explainer
from dreml.cli.orchestrate import orchestrate
from dreml.cli.stab import stability
from dreml.datasets import get_disease_data, load_df

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
    with pkg_resources.path("dreml.resources", fname) as f:
        data_file_path = f
    return Path(data_file_path)


@pytest.mark.parametrize("fname", DATA_NAMES)
def test_load_df(fname):
    """Test load_df"""

    fpath = get_resource_path(fname)
    data = load_df(fpath)
    assert data.shape[0] > 0


def make_disease_path(default=False):
    """Prepare fake disease data folder."""
    tmp_dir = Path(tempfile.mkdtemp())
    if default:
        disease_path_in = get_resource_path("experiment_default.env")
    else:
        disease_path_in = get_resource_path("experiment.env")
    disease_path_out = tmp_dir.joinpath(disease_path_in.name)
    # Use as_posix to make it compatible with python<=3.7
    shutil.copy(disease_path_in.as_posix(), disease_path_out.as_posix())

    with open(disease_path_out, "r", encoding="utf8") as this_file:
        disease_path_out_data = this_file.read()
    if not default:
        disease_path_out_data = disease_path_out_data.replace(
            "%THIS_PATH", disease_path_in.parent.as_posix()
        )
    with open(disease_path_out, "w", encoding="utf8") as this_file:
        this_file.write(disease_path_out_data)

    if default:
        shutil.copy(
            get_resource_path("circuits.tsv.gz").as_posix(),
            disease_path_out.parent.as_posix(),
        )

    return disease_path_out


def test_get_disease_data():
    """Test get_disease_data."""

    disease_path = make_disease_path(default=False)
    gene_exp, pathvals, circuits, genes = get_disease_data(disease_path, debug=True)

    assert gene_exp.to_numpy().ndim == 2
    assert pathvals.to_numpy().ndim == 2
    assert circuits.to_numpy().ndim == 2
    assert genes.to_numpy().ndim == 2


@pytest.mark.parametrize("debug", [True, False])
@pytest.mark.parametrize("default", [True, False])
def test_orchestrate(debug, default):
    """Unit tests for CLI app."""
    click.echo("Running CLI tests fro DREML.")

    disease_path = make_disease_path(default)

    opts = ["--debug" if debug else "--no-debug", f"{disease_path}"]
    click.echo(" ".join(opts))
    runner = CliRunner()
    runner.invoke(orchestrate, " ".join(opts))

    ml_folder_expected = disease_path.parent.joinpath("ml")
    tmp_folder_expected = ml_folder_expected.joinpath("tmp")

    assert ml_folder_expected.exists()
    assert tmp_folder_expected.exists()

    features = joblib.load(tmp_folder_expected.joinpath("features.jbl"))
    if debug:
        assert features.shape[0] == 9
    else:
        assert features.shape[0] > 9


def get_cli_file(fname):
    """Get cli file path."""
    with pkg_resources.path("dreml.cli", fname) as f:
        data_file_path = f
    return Path(data_file_path)


@pytest.mark.parametrize("n_gpus", [0, -1])
def test_stab(n_gpus):
    """Unit tests for CLI app."""
    click.echo("Running CLI tests fro DREML.")

    disease_path = make_disease_path(default=False)

    ml_folder_expected = disease_path.parent.joinpath("ml")
    tmp_folder_expected = ml_folder_expected.joinpath("tmp")

    opts = ["--debug", f"{disease_path}"]
    click.echo(" ".join(opts))
    runner = CliRunner()
    runner.invoke(orchestrate, " ".join(opts))

    opts = ["--mode train", "--debug", f"--n-gpus {n_gpus}", f"{disease_path}"]
    click.echo(" ".join(opts))
    runner = CliRunner()
    runner.invoke(stability, " ".join(opts))

    model_fpath = tmp_folder_expected.joinpath("model_0.jbl")
    assert model_fpath.exists()

    opts = ["--mode explain", "--debug", f"--n-gpus {n_gpus}", f"{disease_path}"]
    click.echo(" ".join(opts))
    runner = CliRunner()
    runner.invoke(stability, " ".join(opts))

    opts = ["--mode score", "--debug", f"--n-gpus {n_gpus}", f"{disease_path}"]
    click.echo(" ".join(opts))
    runner = CliRunner()
    runner.invoke(stability, " ".join(opts))

    res_fpath = tmp_folder_expected.joinpath("stability_results_df.jbl")
    assert res_fpath.exists()

    opts = ["--debug", f"--n-gpus {n_gpus}", f"{disease_path}"]
    click.echo(" ".join(opts))
    runner = CliRunner()
    runner.invoke(explainer, " ".join(opts))

    shap_fpath = ml_folder_expected.joinpath("shap_summary.tsv")
    assert shap_fpath.exists()

    fs_fpath = ml_folder_expected.joinpath("shap_selection.tsv")
    assert fs_fpath.exists()
