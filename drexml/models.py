# -*- coding: utf-8 -*-
"""
Model definition.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor


def get_model(n_features, n_targets, n_jobs, debug, n_iters=0):
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
    if debug:
        max_features = 1.0
    else:
        max_features = int(np.sqrt(n_features) + 20)
    if debug:
        n_estimators = 2
        max_depth = 2
    else:
        n_estimators = 200
        max_depth = 8

    model = RandomForestRegressor(
        n_jobs=n_jobs,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=max_features,
        bootstrap=True,
        criterion="squared_error",
        random_state=this_seed,
    )

    if n_iters != 0:
        raise NotImplementedError("Hyperparameter optimization disabled since v0.2 .")

    print(f"Predicting {n_targets} circuits with {n_features} KDTs")

    return model
