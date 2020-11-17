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


def compute_shap_fs(estimator, X, y, q=0.95):
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X, approximate=True, check_additivity=False)
    shap_values = np.array(shap_values)
    shap_values_summary = pd.DataFrame(
        np.abs(shap_values).mean(axis=(1)), index=y.columns, columns=X.columns
    )

    return (
        shap_values_summary.T.apply(lambda x: x > np.quantile(x, q)).any(axis=1).values
    )


def get_shap_values(explainer, X, y):
    """Using a saved copy of a model, calculate SHAP values.
    Args:
        explainer (shap TreeExplainer):
        X (array-like): Training data.
        y (array-like): Training target.
        samples (int): Number of samples to compute SHAP values for.
        est (Tree-Estimator): If given, a new TreeExplainer will be constructed and
            used, meaning that `explainer` is not used.
    """

    start = time()
    values = explainer.shap_values(X, y, approximate=False, check_additivity=False)
    shap_time = time() - start
    return (
        values,
        " ".join(map(str, ("PID:", os.getpid(), "time taken for SHAP", shap_time))),
    )


def run_stability(model, X, Y, alpha=0.05):
    n_bootstraps = 100
    n_samples, n_variables = X.shape
    n_circuits = Y.shape[1]
    sample_fraction = 0.5
    n_subsamples = np.floor(sample_fraction * n_samples).astype(int)

    lambdas = np.arange(0.95, 1, step=0.05)[:-1]
    n_lambdas = lambdas.size

    # lambda: quantile selected

    Z = np.zeros((n_lambdas, n_bootstraps, n_variables), dtype=np.int8)
    errors = np.zeros((n_bootstraps, n_lambdas))
    errors_mo = np.zeros((n_bootstraps, n_lambdas, n_circuits))

    stability_cv = ShuffleSplit(
        n_splits=n_bootstraps, train_size=n_subsamples, random_state=0
    )

    for n_split, split in enumerate(stability_cv.split(X, Y)):
        print(n_split)
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
        model_.predict(X_test)

        for i, lambda_i in enumerate(lambdas):
            filt_i = compute_shap_fs(model, X_val, Y_val, q=lambda_i)
            X_train_filt = X_train.loc[:, filt_i]
            X_test_filt = X_test.loc[:, filt_i]
            Z[i, n_split, :] = filt_i * 1

            sub_model = clone(model_)
            # sub_model.set_params(max_features=1.0)
            sub_model.fit(X_train_filt, Y_train)
            Y_test_filt_preds = sub_model.predict(X_test_filt)

            r2_loss = 1.0 - r2_score(Y_test, Y_test_filt_preds)
            mo_r2_loss = 1.0 - r2_score(
                Y_test, Y_test_filt_preds, multioutput="raw_values"
            )

            errors[n_split, i] = r2_loss
            errors_mo[n_split, i, :] = mo_r2_loss

            Z[i, n_split, :] = filt_i

            print(f"\t {r2_loss}")

    res = build_stability_dict(Z, errors, alpha)

    return res


def build_stability_dict(z_mat, errors, alpha=0.05):

    support_matrix = np.squeeze(z_mat)

    scores = np.squeeze(1 - errors)
    stab_res = stab.confidenceIntervals(support_matrix, alpha=alpha)
    stability = stab_res["stability"]
    stability_error = stab_res["stability"] - stab_res["lower"]

    res = {
        "support": support_matrix,
        "scores": scores,
        "stability_score": stability,
        "stability_error": stability_error,
        "alpha": 0.05,
    }

    return res


def compute_shap(model, X, Y, test_size=0.3):
    X_learn, X_val, Y_learn, _ = train_test_split(
        X, Y, test_size=0.3, random_state=42
    )

    n_predictors = X.shape[1]
    n_targets = Y.shape[1]

    model_ = clone(model)
    model_.fit(X_learn, Y_learn)

    explainer = shap.TreeExplainer(model_)
    shap_values = explainer.shap_values(
        X_val, approximate=False, check_additivity=False
    )

    corr_sign = lambda x, y: np.sign(pearsonr(x, y)[0])
    signs = Parallel()(
        delayed(corr_sign)(X_val.iloc[:, x_col], shap_values[y_col][:, x_col])
        for x_col in range(n_predictors)
        for y_col in range(n_targets)
    )
    signs = np.array(signs).reshape(n_targets, n_predictors, "F")
    signs = pd.DataFrame(signs, index=Y.columns, columns=X.columns)

    shap_values = np.array(shap_values)
    shap_values_summary = pd.DataFrame(
        np.abs(shap_values).mean(axis=(1)), index=Y.columns, columns=X.columns
    )
    shap_values_summary = shap_values_summary * signs

    shap_values = {
        Y.columns[y_col]: pd.DataFrame(
            shap_values[y_col], columns=X_val.columns, index=X_val.index
        )
        for y_col in range(n_targets)
    }

    return shap_values, shap_values_summary
