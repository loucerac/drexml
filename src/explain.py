# -*- coding: utf-8 -*-
"""
Author: Carlos Loucera <carlos.loucera@juntadeandalucia.es>
Author: Maria Pena Chilet <maria.pena.chilet.ext@juntadeandalucia.es>
Author: Marina Esteban <marina.estebanm@gmail.com>

Explainability module for multi-task framework.
"""

import os
from time import time

import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import (
    ShuffleSplit,
    train_test_split,
)

import src.stability as stab
from joblib import Parallel, delayed
from scipy.stats import pearsonr
import joblib


def compute_shap_fs(relevances, q=0.95, by_circuit=False):

    by_circuit_frame = relevances.abs().apply(lambda x: x > np.quantile(x, q), axis=1)

    if by_circuit:
        res = by_circuit_frame
    else:
        res = by_circuit_frame.any().values

    return res


def compute_shap_values(estimator, X, y=None, approximate=True, check_additivity=False):
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(
        X, approximate=approximate, check_additivity=check_additivity
    )

    return shap_values


def compute_shap_relevance(shap_values, X, Y):

    feature_names = X.columns
    task_names = Y.columns

    n_features = len(feature_names)
    n_tasks = len(task_names)

    corr_sign = lambda x, y: np.sign(pearsonr(x, y)[0])
    signs = Parallel(n_jobs=-1)(
        delayed(corr_sign)(X.iloc[:, x_col], shap_values[y_col][:, x_col])
        for x_col in range(n_features)
        for y_col in range(n_tasks)
    )

    signs = np.array(signs).reshape((n_tasks, n_features), order="F")
    signs = pd.DataFrame(signs, index=Y.columns, columns=X.columns)

    shap_values = np.array(shap_values)

    shap_relevance = pd.DataFrame(
        np.abs(shap_values).mean(axis=(1)), index=task_names, columns=feature_names
    )

    shap_relevance = shap_relevance * signs
    shap_relevance = shap_relevance.fillna(0.0)

    return shap_relevance


def run_stability(model, X, Y, alpha=0.05, approximate=False, check_additivity=False):
    n_bootstraps = 100
    n_samples, n_variables = X.shape
    sample_fraction = 0.5
    n_subsamples = np.floor(sample_fraction * n_samples).astype(int)

    q = 0.95

    # lambda: quantile selected

    Z = np.zeros((n_bootstraps, n_variables), dtype=np.int8)
    errors = np.zeros(n_bootstraps)

    stability_cv = ShuffleSplit(
        n_splits=n_bootstraps, train_size=n_subsamples, random_state=0
    )

    def stab_i(model, X, Y, n_split, split, q=0.95):
        train, test = split
        X_train = X.iloc[train, :]
        Y_train = Y.iloc[train, :]
        X_test = X.iloc[test, :]
        Y_test = Y.iloc[test, :]

        X_learn, X_val, Y_learn, Y_val = train_test_split(
            X_train, Y_train, test_size=0.3, random_state=n_split
        )

        model_ = clone(model)
        model_.fit(X_learn, Y_learn)

        # FS using shap relevances
        shap_values = compute_shap_values(
            model_,
            X_val,
            approximate=approximate,
            check_additivity=check_additivity,
        )
        shap_relevances = compute_shap_relevance(shap_values, X_val, Y_val)
        filt_i = compute_shap_fs(shap_relevances, q=q, by_circuit=False)

        X_train_filt = X_train.loc[:, filt_i]
        X_test_filt = X_test.loc[:, filt_i]

        sub_model = clone(model_)
        # sub_model.set_params(max_features=1.0)
        sub_model.fit(X_train_filt, Y_train)
        Y_test_filt_preds = sub_model.predict(X_test_filt)

        r2_loss = 1.0 - r2_score(Y_test, Y_test_filt_preds)
        mo_r2_loss = 1.0 - r2_score(
            Y_test, Y_test_filt_preds, multioutput="raw_values"
        )

        return (filt_i, r2_loss, mo_r2_loss)

    stab_values = Parallel(n_jobs=-1)(
        delayed(stab_i)(model, X, Y, n_split, split)
        for n_split, split in enumerate(stability_cv.split(X, Y))
    )

    for n_split, values in enumerate(stab_values):
        Z[n_split, :] = values[0]
        errors[n_split] = values[1]

    res = build_stability_dict(Z, errors, alpha)

    return res


def build_stability_dict(z_mat, errors, alpha=0.05):

    support_matrix = np.squeeze(z_mat)

    scores = np.squeeze(1 - errors)
    stab_res = stab.confidenceIntervals(support_matrix, alpha=alpha)
    stability = stab_res["stability"]
    stability_error = stab_res["stability"] - stab_res["lower"]

    res = {
        "scores": scores,
        "stability_score": stability,
        "stability_error": stability_error,
        "alpha": alpha,
    }

    return res


def compute_shap(
    model, X, Y, test_size=0.3, q=0.95, approximate=True, check_additivity=False
):
    X_learn, X_val, Y_learn, Y_val = train_test_split(
        X, Y, test_size=test_size, random_state=42
    )

    model_ = clone(model)
    model_.fit(X_learn, Y_learn)

    shap_values = compute_shap_values(
        model_,
        X_val,
        approximate=approximate,
        check_additivity=check_additivity,
    )
    shap_relevances = compute_shap_relevance(shap_values, X_val, Y_val)
    fs = compute_shap_fs(shap_relevances, q=q, by_circuit=True)
    fs = fs * 1

    return shap_relevances, fs
