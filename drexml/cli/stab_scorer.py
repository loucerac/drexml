#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Entry CLI point for stab.
"""


import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.metrics import r2_score
from sklearn.model_selection import HalvingRandomSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from drexml.explain import build_stability_dict
from drexml.pystab import nogueria_test
from drexml.utils import convert_names, get_stab, parse_stab

if __name__ == "__main__":
    import sys

    # client = Client('127.0.0.1:8786')
    # pylint: disable=unbalanced-tuple-unpacking
    data_folder, n_iters, n_gpus, n_cpus, n_splits, debug, add = parse_stab(sys.argv)
    model, stab_cv, features, targets = get_stab(
        data_folder, n_splits, n_cpus, debug, n_iters
    )

    fs_mat = joblib.load(data_folder.joinpath("fs.jbl"))
    n_bootstraps = len(fs_mat)

    zmat = np.zeros((n_bootstraps, features.shape[1]), dtype=np.int8)
    errors = np.zeros(n_bootstraps)

    def stab_i(estimator, X, Y, split_id, this_split):
        """Score each stab partition."""
        learn, val, test = this_split
        features_learn = X.iloc[learn, :]
        targets_learn = Y.iloc[learn, :]
        features_val = X.iloc[val, :]
        targets_val = Y.iloc[val, :]
        features_test = X.iloc[test, :]
        targets_test = Y.iloc[test, :]
        features_train = pd.concat((features_learn, features_val), axis=0)
        targets_train = pd.concat((targets_learn, targets_val), axis=0)

        filt_i = fs_mat[split_id].any().values
        if not filt_i.any():
            estimator_ = clone(estimator)
            estimator_.fit(features_learn, targets_learn)
            if hasattr(estimator_, "feature_importances_"):
                fimp = estimator_.feature_importances_
            elif hasattr(estimator_, "best_estimator_"):
                fimp = estimator_.best_estimator_.feature_importances_
            elif hasattr(estimator_, "named_steps"):
                if hasattr(estimator_[-1], "best_estimator_"):
                    fimp = estimator_[-1].best_estimator_.feature_importances_
                else:
                    fimp = estimator_[-1].feature_importances_
            filt_i[fimp.argmax()] = True

        features_train_filt = features_train.loc[:, filt_i]
        features_test_filt = features_test.loc[:, filt_i]

        with joblib.parallel_backend("loky", n_jobs=n_cpus):
            if isinstance(estimator, RandomForestRegressor):
                estimator_filt = clone(estimator)
                estimator_filt.max_features = 1.0
            elif isinstance(estimator, (RandomizedSearchCV, HalvingRandomSearchCV)):
                estimator_filt = clone(estimator)
            elif isinstance(estimator, Pipeline):
                estimator_filt = clone(estimator)
                if isinstance(estimator[-1], RandomForestRegressor):
                    estimator_filt[-1].max_features = 1.0

                # estimator_filt.set_params(**{"max_features": 1.0, "random_state": 42})
            # sub_model.set_params(**{"max_depth": 32, "max_features": filt_i.sum()})
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
        stab_values.append(stab_i(model, features, targets, n_split, split))

    for n_split, values in enumerate(stab_values):
        zmat[n_split, :] = values[0] * 1
        errors[n_split] = values[1]

    score_by_circuit = [
        pd.Series(values[2], index=targets.columns)
        for n_split, values in enumerate(stab_values)
    ]
    score_by_circuit = (
        pd.concat(score_by_circuit, axis=1)
        .T.describe()
        .T[["mean", "std", "25%", "75%"]]
    )
    score_by_circuit.columns = "r2_" + score_by_circuit.columns

    stab_by_circuit = {
        y: nogueria_test(
            pd.concat([x.loc[y] for x in fs_mat], axis=1).T.values * 1, as_dict=True
        )
        for y in targets.columns
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

    joblib.dump(stability_results_df, data_folder.joinpath("stability_results.jbl"))

    stability_results_df.to_csv(
        data_folder.joinpath("stability_results.tsv"), sep="\t", index_label="name"
    )

    stability_results_renamed_df = convert_names(
        stability_results_df, ["circuits"], axis=[0]
    )

    stability_results_renamed_df.to_csv(
        data_folder.joinpath("stability_results_symbol.tsv"),
        sep="\t",
        index_label="name",
    )
