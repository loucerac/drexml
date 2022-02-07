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

from dreml.explain import (
    compute_shap_fs,
    compute_shap_relevance,
    compute_shap_values,
    build_stability_dict,
)
import dreml.stability as stab
from dreml.models import get_model
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score


def score_stab(model, X, Y, cv, fs, n_jobs, alpha=0.05):
    """[summary]

    Parameters
    ----------
    model : [type]
        [description]
    X : [type]
        [description]
    Y : [type]
        [description]
    cv : [type]
        [description]
    fs : [type]
        [description]
    n_jobs : [type]
        [description]
    alpha : float, optional
        [description], by default 0.05

    Returns
    -------
    [type]
        [description]
    """
    n_bootstraps = len(fs)
    n_samples, n_features = X.shape

    Z = np.zeros((n_bootstraps, n_features), dtype=np.int8)
    errors = np.zeros(n_bootstraps)

    def stab_i(model, X, Y, n_split, split):
        print(n_split)
        learn, val, test = split
        X_learn = X.iloc[learn, :]
        Y_learn = Y.iloc[learn, :]
        X_val = X.iloc[val, :]
        Y_val = Y.iloc[val, :]
        X_test = X.iloc[test, :]
        Y_test = Y.iloc[test, :]
        X_train = pd.concat((X_learn, X_val), axis=0)
        Y_train = pd.concat((Y_learn, Y_val), axis=0)

        filt_i = fs[n_split].any().values
        if not filt_i.any():
            model.fit(X_learn, Y_learn)
            filt_i[model.feature_importances_.argmax()] = True

        X_train_filt = X_train.loc[:, filt_i]
        X_test_filt = X_test.loc[:, filt_i]

        with joblib.parallel_backend("multiprocessing", n_jobs=n_jobs):
            sub_model = clone(model)
            # sub_model.set_params(**{"max_depth": 32, "max_features": filt_i.sum()})
            sub_model.set_params(max_features=1.0)
            sub_model.fit(X_train_filt, Y_train)
            Y_test_filt_preds = sub_model.predict(X_test_filt)

        r2 = r2_score(Y_test, Y_test_filt_preds)
        mo_r2 = r2_score(Y_test, Y_test_filt_preds, multioutput="raw_values")

        return (filt_i, r2, mo_r2)

    stab_values = []
    for n_split, split in enumerate(cv):
        stab_values.append(stab_i(model, X, Y, n_split, split))

    for n_split, values in enumerate(stab_values):
        Z[n_split, :] = values[0] * 1
        errors[n_split] = values[1]

    score_by_circuit = [
        pd.Series(values[2], index=Y.columns)
        for n_split, values in enumerate(stab_values)
    ]
    score_by_circuit = (
        pd.concat(score_by_circuit, axis=1)
        .T.describe()
        .T[["mean", "std", "25%", "75%"]]
    )
    score_by_circuit.columns = "r2_" + score_by_circuit.columns

    stab_by_circuit = {
        y: stab.confidenceIntervals(
            pd.concat([x.loc[y] for x in fs], axis=1).T.values * 1
        )
        for y in Y.columns
    }

    stab_by_circuit = pd.DataFrame(stab_by_circuit).T

    res_by_circuit = pd.concat((stab_by_circuit, score_by_circuit), axis=1)

    res = build_stability_dict(Z, errors, alpha)

    res_df = pd.DataFrame(
        {
            "stability": [res["stability_score"]],
            "lower": [res["stability_score"] - res["stability_error"]],
            "upper": [res["stability_score"] + res["stability_error"]],
            "r2_mean": [np.mean(res["scores"])],
            "r2_std": [np.std(res["scores"])],
            "r2_25%": [np.quantile(res["scores"], 0.25)],
            "r2_75%": [np.quantile(res["scores"], 0.75)],
        },
        index=["map"],
    )

    res_df = pd.concat((res_df, res_by_circuit), axis=0)
    res_df = res_df.rename(
        {"lower": "stability_lower_95ci", "upper": "stability_upper_95ci"}, axis=1
    )

    print("Stability score for the disease map: ", res["stability_score"])

    return res, res_df


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
