#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Carlos Loucera <carlos.loucera@juntadeandalucia.es>
Author: Maria Pena Chilet <maria.pena.chilet.ext@juntadeandalucia.es>
Author: Marina Esteban <marina.estebanm@gmail.com>

Entry CLI point for stab.
"""

import pathlib

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import r2_score

import dreml.stability as stab
from dreml.explain import build_stability_dict
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

    n_samples, n_features = features_orig.shape
    n_targets = targets_orig.shape[1]

    model = get_model(n_features, n_targets, n_cpus, debug, n_iters)
    fs_mat = joblib.load(data_folder.joinpath("fs.jbl"))
    n_bootstraps = len(fs_mat)

    zmat = np.zeros((n_bootstraps, n_features), dtype=np.int8)
    errors = np.zeros(n_bootstraps)

    def stab_i(estimator, features, targets, split_id, this_split):
        """Score each stab partition."""
        learn, val, test = this_split
        features_learn = features.iloc[learn, :]
        targets_learn = targets.iloc[learn, :]
        features_val = features.iloc[val, :]
        targets_val = targets.iloc[val, :]
        features_test = features.iloc[test, :]
        targets_test = targets.iloc[test, :]
        features_train = pd.concat((features_learn, features_val), axis=0)
        targets_train = pd.concat((targets_learn, targets_val), axis=0)

        filt_i = fs_mat[split_id].any().values
        if not filt_i.any():
            estimator_ = clone(estimator)
            estimator_.fit(features_learn, targets_learn)
            filt_i[estimator_.feature_importances_.argmax()] = True

        features_train_filt = features_train.loc[:, filt_i]
        features_test_filt = features_test.loc[:, filt_i]

        with joblib.parallel_backend("loky", n_jobs=n_cpus):
            estimator_filt = clone(estimator)
            # sub_model.set_params(**{"max_depth": 32, "max_features": filt_i.sum()})
            estimator_filt.set_params(max_features=1.0)
            estimator_filt.fit(features_train_filt, targets_train)
            targets_test_filt_preds = estimator_filt.predict(features_test_filt)

        r2_split = r2_score(targets_test, targets_test_filt_preds)
        mo_r2_split = r2_score(
            targets_test, targets_test_filt_preds, multioutput="raw_values"
        )

        return (filt_i, r2_split, mo_r2_split)

    stab_values = []
    cv = joblib.load(data_folder.joinpath("cv.jbl"))
    for n_split, split in enumerate(cv):
        stab_values.append(stab_i(model, features_orig, targets_orig, n_split, split))

    for n_split, values in enumerate(stab_values):
        zmat[n_split, :] = values[0] * 1
        errors[n_split] = values[1]

    score_by_circuit = [
        pd.Series(values[2], index=targets_orig.columns)
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
            pd.concat([x.loc[y] for x in fs_mat], axis=1).T.values * 1
        )
        for y in targets_orig.columns
    }

    stab_by_circuit = pd.DataFrame(stab_by_circuit).T

    res_by_circuit = pd.concat((stab_by_circuit, score_by_circuit), axis=1)

    stability_results = build_stability_dict(zmat, errors, alpha=0.05)

    stability_results_df = pd.DataFrame(
        {
            "stability": [stability_results["stability_score"]],
            "lower": [
                stability_results["stability_score"]
                - stability_results["stability_error"]
            ],
            "upper": [
                stability_results["stability_score"]
                + stability_results["stability_error"]
            ],
            "r2_mean": [np.mean(stability_results["scores"])],
            "r2_std": [np.std(stability_results["scores"])],
            "r2_25%": [np.quantile(stability_results["scores"], 0.25)],
            "r2_75%": [np.quantile(stability_results["scores"], 0.75)],
        },
        index=["map"],
    )

    stability_results_df = pd.concat((stability_results_df, res_by_circuit), axis=0)
    stability_results_df = stability_results_df.rename(
        {"lower": "stability_lower_95ci", "upper": "stability_upper_95ci"}, axis=1
    )

    print("Stability score for the disease map: ", stability_results["stability_score"])

    joblib.dump(stability_results_df, data_folder.joinpath("stability_results_df.jbl"))
