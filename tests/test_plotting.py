# -*- coding: utf-8 -*-
"""
Unit testing for utils module.
"""
import pathlib
import tempfile

from drexml.plotting import plot_metrics

from .this_utils import THIS_DIR

PLOTTING_EXTENSIONS = ["pdf", "png"]


def test_plot_metrics():
    """
    Test plotting metrics.
    """
    fpath = pathlib.Path(THIS_DIR, "stability_results.tsv")
    tmp_folder = pathlib.Path(tempfile.mkdtemp())

    plot_files = [
        tmp_folder.joinpath(f"{x.stem}.{ext}")
        for ext in PLOTTING_EXTENSIONS
        for x in [fpath]
    ]

    plot_metrics(input_path=fpath, output_folder=tmp_folder)
    assert all([x.exists() for x in plot_files])
