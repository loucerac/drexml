# -*- coding: utf-8 -*-
"""
Unit testing for datasets module.
"""

from tempfile import mkstemp

import pandas as pd
import pytest
from pandas.errors import ParserError

from drexml.datasets import get_disease_data, load_df

from .test_utils import make_disease_config


@pytest.mark.parametrize("use_seeds", [True, False])
@pytest.mark.parametrize("update", [False])
def test_get_disease_data(use_seeds, update):
    """Test get_disease_data."""

    expected_circuits = ["P.hsa03320.28", "P.hsa04920.43"]

    disease_path = make_disease_config(use_seeds=use_seeds, update=update)
    gene_exp, pathvals, circuits, genes = get_disease_data(disease_path)

    assert gene_exp.to_numpy().ndim == 2
    assert pathvals.to_numpy().ndim == 2
    assert len(circuits) == 2
    assert set(pathvals.columns) == set(circuits)
    assert set(circuits) == set(expected_circuits)
    assert genes.to_numpy().ndim == 2
    assert gene_exp.columns.isin(genes.index).all()
    assert gene_exp.columns.isin(genes.index[genes.drugbank_approved_targets]).all()


@pytest.mark.xfail(raises=(NotImplementedError,))
def test_load_df_fails_empty():
    """Unit test that load_df fails with an empty df."""
    _, tmp_file = mkstemp()
    pd.DataFrame().to_csv(tmp_file, sep="\t")
    load_df(tmp_file)


@pytest.mark.xfail(raises=(NotImplementedError,))
def test_load_df_fails_tsv():
    """Unit test that load_df fails with an ill-formed TSV."""
    _, tmp_file = mkstemp()
    with open(tmp_file, "w", encoding="utf8") as f:
        f.write("\t \t")
        f.write("\t")
    load_df(tmp_file)


@pytest.mark.xfail(
    raises=(
        ParserError,
        KeyError,
    )
)
def test_load_df_fails_feather():
    """Unit test that load_df fails when loading a file in feather format."""
    _, tmp_file = mkstemp()
    pd.DataFrame().reset_index(names="vader").to_feather(tmp_file)
    load_df(tmp_file)
