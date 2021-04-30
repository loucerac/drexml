from sklearn.ensemble import RandomForestRegressor
import numpy as np


def get_model(n_features, n_jobs, debug):
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
        n_estimators = 200
    else:
        n_estimators = 200
    
    model = RandomForestRegressor(
        n_jobs=n_jobs, n_estimators=n_estimators, max_depth=8, max_features=mtry
    )

    return model
