#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Entry CLI point for stab.
"""

import pathlib

import joblib
from sklearn.base import clone
from sklearn.model_selection import ShuffleSplit, train_test_split

from dreml.models import get_model

if __name__ == "__main__":
    import sys

    # client = Client('127.0.0.1:8786')

    _, data_folder, n_iters, n_gpus, n_cpus, debug = sys.argv
    n_iters = int(n_iters)
    data_folder = pathlib.Path(data_folder)
    n_gpus = int(n_gpus)
    use_gpu = n_gpus > 0
    n_cpus = int(n_cpus)
    debug = bool(debug)

    features_orig_fpath = data_folder.joinpath("features.jbl")
    features_orig = joblib.load(features_orig_fpath)

    targets_orig_fpath = data_folder.joinpath("target.jbl")
    targets_orig = joblib.load(targets_orig_fpath)

    n_splits = 5 if debug else 100
    stab_cv = ShuffleSplit(n_splits=n_splits, train_size=0.5, random_state=0)
    stab_cv = list(stab_cv.split(features_orig, targets_orig))
    stab_cv = [
        (*train_test_split(stab_cv[i][0], test_size=0.3), stab_cv[i][1])
        for i in range(n_splits)
    ]

    fname = "cv.jbl"
    fpath = data_folder.joinpath(fname)
    joblib.dump(stab_cv, fpath)
    n_features = features_orig.shape[1]
    n_targets = targets_orig.shape[1]

    model = get_model(n_features, n_targets, n_cpus, debug, n_iters)

    for i, split in enumerate(stab_cv):
        with joblib.parallel_backend("loky", n_jobs=n_cpus):
            model_ = clone(model)
            model_.set_params(random_state=i)
            model_.fit(features_orig.iloc[split[0], :], targets_orig.iloc[split[0], :])
            fname = f"model_{i}.jbl"
            fpath = data_folder.joinpath(fname)
            joblib.dump(model_, fpath)
            print(f"50-50 split {i}")
