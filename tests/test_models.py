# -*- coding: utf-8 -*-
"""
Unit testing for datasets module.
"""
import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from dreml.models import get_model
from shap.utils import safe_isinstance

@pytest.mark.parametrize("debug", [True, False])
def test_model_hp(debug):
    """Check the different options to construct the MORF model."""

    n_features = 700
    n_targets = 100
    n_jobs = -1

    max_features_expected = int(np.sqrt(n_features) + 20)
    n_estimators_expected = 20 if debug else int(1.5 * (n_features + n_targets))

    model = get_model(n_features, n_targets, n_jobs, debug)

    assert safe_isinstance(model, "sklearn.ensemble.RandomForestRegressor")
