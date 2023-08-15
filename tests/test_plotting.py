# -*- coding: utf-8 -*-
"""
Unit testing for utils module.
"""
import pathlib
import tempfile

import pytest

from drexml.plotting import RepurposingResult

from .this_utils import THIS_DIR

PLOTTING_EXTENSIONS = ["pdf", "png"]


def setup_results():
    """
    Setup results for testing.
    """
    stab_path = pathlib.Path(THIS_DIR, "stability_results.tsv")
    score_path = pathlib.Path(THIS_DIR, "shap_summary.tsv")
    sel_path = pathlib.Path(THIS_DIR, "shap_selection.tsv")

    return RepurposingResult(sel_mat=sel_path, score_mat=score_path, stab_mat=stab_path)


def test_plot_metrics():
    """
    Test plotting metrics.
    """
    tmp_folder = pathlib.Path(tempfile.mkdtemp())
    results = setup_results()

    results.plot_metrics(output_folder=tmp_folder)

    plot_files = [tmp_folder.joinpath(f"metrics.{ext}") for ext in PLOTTING_EXTENSIONS]

    assert all([x.exists() for x in plot_files])


def test_plot_gene():
    """
    Test plotting metrics.
    """
    tmp_folder = pathlib.Path(tempfile.mkdtemp())
    results = setup_results()

    this_gene = "3066"
    results.plot_gene_profile(output_folder=tmp_folder, gene=this_gene)

    plot_files = [
        tmp_folder.joinpath(f"profile_{this_gene}.{ext}") for ext in PLOTTING_EXTENSIONS
    ]

    assert all([x.exists() for x in plot_files])


@pytest.mark.xfail(raises=(KeyError,))
def test_plot_gene_fails():
    """
    Test that plot raises an error when a gene is not part of the relevance matrix.
    """

    tmp_folder = pathlib.Path(tempfile.mkdtemp())
    results = setup_results()

    results.plot_gene_profile(output_folder=tmp_folder, gene="VADER")
