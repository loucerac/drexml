import numpy as np
from sklearn.ensemble import RandomForestRegressor


def get_model(n_features, n_targets, n_jobs, debug, n_iters):
    """[summary]

    Parameters
    ----------
    n_iters : [type]
        [description]
    gpu : [type]
        [description]
    n_jobs : [type]
        [description]
    debug : [type]
        [description]

    Returns
    -------
    [type]
        [description]
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
