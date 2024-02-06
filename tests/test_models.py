# -*- coding: utf-8 -*-
"""
Unit testing for datasets module.
"""
import warnings

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", module="shap", message="IPython could not be loaded!"
    )
    import shap

from drexml.models import get_model


@pytest.mark.parametrize("debug", [True, False])
def test_model_hp(debug):
    """Check the different options to construct the MORF model."""

    n_features = 700
    n_targets = 100
    n_jobs = -1

    max_features_expected = 1.0 if debug else int(np.sqrt(n_features) + 20)
    n_estimators_expected = 2 if debug else 200
    max_depth_expected = 2 if debug else 8

    model = get_model(n_features, n_targets, n_jobs, debug)

    assert shap.utils.safe_isinstance(model, "sklearn.ensemble.RandomForestRegressor")
    assert model.n_estimators == n_estimators_expected
    assert model.max_depth == max_depth_expected
    assert model.max_features == max_features_expected


@pytest.mark.parametrize(
    "n_iters, expected",
    [(0, RandomForestRegressor), (100, (HalvingRandomSearchCV, RandomizedSearchCV))],
)
@pytest.mark.parametrize("use_imputer", [True, False])
def test_get_model(n_iters, expected, use_imputer):
    """Test that get_model fails when triyng to perform HP opt."""

    model = get_model(
        n_features=1,
        n_targets=1,
        n_jobs=1,
        debug=False,
        n_iters=n_iters,
        use_imputer=use_imputer,
    )

    if use_imputer:
        assert isinstance(model, Pipeline)
    else:
        assert isinstance(model, expected)
