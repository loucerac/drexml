#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Entry CLI point for stab.
"""

import joblib
from sklearn.base import clone

from dreml.utils import get_stab, parse_stab

if __name__ == "__main__":
    import sys

    # client = Client('127.0.0.1:8786')
    # pylint: disable=unbalanced-tuple-unpacking
    data_path, n_iters, n_gpus, n_cpus, n_splits, debug = parse_stab(sys.argv)
    model, stab_cv, X, Y = get_stab(data_path, n_splits, n_cpus, debug, n_iters)

    for i, split in enumerate(stab_cv):
        with joblib.parallel_backend("loky", n_jobs=n_cpus):
            model_ = clone(model)
            model_.set_params(random_state=i)
            model_.fit(X.iloc[split[0], :], Y.iloc[split[0], :])
            fname = f"model_{i}.jbl"
            fpath = data_path.joinpath(fname)
            joblib.dump(model_, fpath)
            print(f"50-50 split {i}")
