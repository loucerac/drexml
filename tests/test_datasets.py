# -*- coding: utf-8 -*-
"""
Unit testing for datasets module.
"""

import pytest

from drexml.datasets import get_disease_data

from .test_utils import make_disease_config


@pytest.mark.parametrize("use_seeds", [True, False])
@pytest.mark.parametrize("update", [False])
def test_get_disease_data(use_seeds, update):
    """Test get_disease_data."""

    expected_circuits = ["P.hsa03320.28", "P.hsa04920.43"]

    disease_path = make_disease_config(use_seeds=use_seeds, update=update)
    gene_exp, pathvals, circuits, genes = get_disease_data(disease_path, debug=True)

    assert gene_exp.to_numpy().ndim == 2
    assert pathvals.to_numpy().ndim == 2
    assert len(circuits) == 2
    assert set(pathvals.columns) == set(circuits)
    assert set(circuits) == set(expected_circuits)
    assert genes.to_numpy().ndim == 2
    assert gene_exp.columns.isin(genes.index).all()
    assert gene_exp.columns.isin(genes.index[genes.drugbank_approved_targets]).all()


# @pytest.mark.xfail(raises=(ValueError, FileNotFoundError, UnicodeDecodeError))
# def test_load_df_fails_tsv():
#     """Test load_df fails when loading a non TSV or FEATHER file."""
