#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Entry CLI point for stab.
"""

import joblib
from sklearn.base import clone

from drexml.utils import get_stab, parse_stab

if __name__ == "__main__":
    import sys

    # client = Client('127.0.0.1:8786')
    # pylint: disable=unbalanced-tuple-unpacking
    data_path, n_iters, n_gpus, n_cpus, n_splits, debug, add = parse_stab(sys.argv)
    model, stab_cv, X, Y = get_stab(data_path, n_splits, n_cpus, debug, n_iters)

    for i, split in enumerate(stab_cv):
        X_learn = X.iloc[split[0], :]
        X_val = X.iloc[split[1], :]
        y_learn = Y.iloc[split[0], :]
        y_val = Y.iloc[split[1], :]
        with joblib.parallel_backend("loky", n_jobs=n_cpus):
            model_ = clone(model)
            model_.random_state = i
            if hasattr(model_, "n_estimators_min"):
                model_.fit(X_learn, y_learn, X_val=X_val, y_val=y_val)
            else:
                model_.fit(X_learn, y_learn)
            fname = f"model_{i}.jbl"
            fpath = data_path.joinpath(fname)
            joblib.dump(model_, fpath)
            print(f"50-50 split {i}")
