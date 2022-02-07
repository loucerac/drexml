#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Carlos Loucera <carlos.loucera@juntadeandalucia.es>
Author: Maria Pena Chilet <maria.pena.chilet.ext@juntadeandalucia.es>
Author: Marina Esteban <marina.estebanm@gmail.com>

Entry CLI point for stab.
"""

import multiprocessing
import os
import pathlib
from functools import partial

import joblib
from sklearn.base import clone
from sklearn.model_selection import ShuffleSplit, train_test_split

from dreml.explain import compute_shap_fs, compute_shap_relevance, compute_shap_values
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

    X_fpath = data_folder.joinpath("features.jbl")
    X = joblib.load(X_fpath)

    Y_fpath = data_folder.joinpath("target.jbl")
    Y = joblib.load(Y_fpath)

    n_splits = 5 if debug else 100
    cv = ShuffleSplit(n_splits=n_splits, train_size=0.5, random_state=0)
    cv = list(cv.split(X, Y))
    cv = [
        (*train_test_split(cv[i][0], test_size=0.3), cv[i][1]) for i in range(n_splits)
    ]

    fname = f"cv.jbl"
    fpath = data_folder.joinpath(fname)
    joblib.dump(cv, fpath)
    n_features = X.shape[1]
    n_targets = Y.shape[1]

    model = get_model(n_features, n_targets, n_cpus, debug, n_iters)

    for i, split in enumerate(cv):
        with joblib.parallel_backend("loky", n_jobs=n_cpus):
            model_ = clone(model)
            model_.set_params(random_state=i)
            model_.fit(X.iloc[split[0], :], Y.iloc[split[0], :])
            fname = f"model_{i}.jbl"
            fpath = data_folder.joinpath(fname)
            joblib.dump(model_, fpath)
            print(f"50-50 split {i}")

    n_devices = n_gpus if n_gpus > 0 else n_cpus
    print(n_devices)

    # filts = Parallel(n_jobs=N_GPU, backend="multiprocessing")(
    #    delayed(runner)(n_split, data_folder) for n_split in range(n))

    device_list = list(range(n_devices))

    queue = multiprocessing.Queue()
    # initialize the queue with the GPU ids

    def runner(data_path, gpu_flag, split_id):
        """Run instance."""
        gpu_id = queue.get()
        filt_i = None
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            print("device", os.environ["CUDA_VISIBLE_DEVICES"])

            cv_fname = "cv.jbl"
            cv_fpath = data_path.joinpath(cv_fname)
            cv_gen = joblib.load(cv_fpath)
            sample_ids = cv_gen[split_id]

            features_fpath = data_path.joinpath("features.jbl")
            features = joblib.load(features_fpath)

            targets_fpath = data_path.joinpath("target.jbl")
            targets = joblib.load(targets_fpath)

            features_learn = features.iloc[sample_ids[0], :]

            features_val = features.iloc[sample_ids[1], :]
            targets_val = targets.iloc[sample_ids[1], :]

            cv_fname = f"model_{split_id}.jbl"
            cv_fpath = data_path.joinpath(cv_fname)
            this_model = joblib.load(cv_fpath)

            shap_values = compute_shap_values(
                this_model, features_learn, features_val, gpu_flag
            )
            shap_relevances = compute_shap_relevance(
                shap_values, features_val, targets_val
            )
            filt_i = compute_shap_fs(
                shap_relevances,
                model=this_model,
                X=features_val,
                Y=targets_val,
                q="r2",
                by_circuit=True,
            )

            cv_fname = f"filt_{split_id}.jbl"
            cv_fpath = data_path.joinpath(cv_fname)
            joblib.dump(filt_i, cv_fpath)
            print(gpu_id, filt_i.sum(axis=1).value_counts())

        finally:
            queue.put(gpu_id)

        return filt_i

    # Put indices in queue
    for gpu_ids in range(n_devices):
        queue.put(gpu_ids)

    r = partial(runner, data_path=data_folder, gpu_flag=use_gpu)

    with joblib.parallel_backend("multiprocessing", n_jobs=n_devices):
        fs = joblib.Parallel()(
            joblib.delayed(runner)(
                data_path=data_folder,
                gpu_flag=use_gpu,
                split_id=i_split,
            )
            for i_split in range(n_splits)
        )
    joblib.dump(fs, data_folder.joinpath("fs.jbl"))
