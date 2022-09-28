# -*- coding: utf-8 -*-
"""
Model definition.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor


def get_model(n_features, n_targets, n_jobs, debug, n_iters=None):
    """Create a model.

    Parameters
    ----------
    n_features : int
        Number of features (KDTs).
    n_targets : int
        Number of targets (circuits).
    n_jobs : int
        The number of jobs to run in parallel.
    debug : bool
        Debug flag.
    n_iters : int, optional
        Number of ietrations for hyperparatemer optimization, by default None

    Returns
    -------
    sklearn.ensemble.RandomForestRegressor
        The model to be fitted.
    """
    mtry = int(np.sqrt(n_features) + 20)
    if debug:
        n_estimators = 20
    else:
        n_estimators = int(1.5 * (n_features + n_targets))

    model = RandomForestRegressor(
        n_jobs=n_jobs,
        n_estimators=n_estimators,
        max_depth=8,
        max_features=mtry,
    )

    return model
