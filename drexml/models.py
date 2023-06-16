# -*- coding: utf-8 -*-
"""
Model definition.
"""

import copy

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

    this_seed = 275
    mtry = int(np.sqrt(n_features) + 20)
    if debug:
        n_estimators = 2
    else:
        n_estimators = 200

    model = RandomForestRegressor(
        n_jobs=n_jobs,
        n_estimators=n_estimators,
        max_depth=8,
        max_features=mtry,
        random_state=this_seed,
    )

    print(f"Predicting {n_targets} circuits with {n_features} KDTs")

    return model
