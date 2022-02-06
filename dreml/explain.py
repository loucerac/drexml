# -*- coding: utf-8 -*-
"""
Author: Carlos Loucera <carlos.loucera@juntadeandalucia.es>
Author: Maria Pena Chilet <maria.pena.chilet.ext@juntadeandalucia.es>
Author: Marina Esteban <marina.estebanm@gmail.com>

Explainability module for multi-task framework.
"""

import multiprocessing
import os
import pathlib
from functools import partial
from time import time

import joblib
import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import ShuffleSplit, train_test_split

from dreml import stability as stab
from dreml.models import get_model


def matcorr(O, P):
    """[summary]

    Parameters
    ----------
    O : [type]
        [description]
    P : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    (n, t) = O.shape  # n traces of t samples
    (n_bis, m) = P.shape  # n predictions for each of m candidates

    DO = O - (
        np.einsum("nt->t", O, optimize="optimal") / np.double(n)
    )  # compute O - mean(O)
    DP = P - (
        np.einsum("nm->m", P, optimize="optimal") / np.double(n)
    )  # compute P - mean(P)

    cov = np.einsum("nm,nt->mt", DP, DO, optimize="optimal")

    varP = np.einsum("nm,nm->m", DP, DP, optimize="optimal")
    varO = np.einsum("nt,nt->t", DO, DO, optimize="optimal")
    tmp = np.einsum("m,t->mt", varP, varO, optimize="optimal")

    return cov / np.sqrt(tmp)


def compute_shap_values(estimator, background, new, gpu, split=True):
    """[summary]

    Parameters
    ----------
    estimator : [type]
        [description]
    background : [type]
        [description]
    new : [type]
        [description]
    gpu : [type]
        [description]
    split : bool, optional
        [description], by default True

    Returns
    -------
    [type]
        [description]
    """
    if gpu:
        check_add = True
        explainer = shap.GPUTreeExplainer(estimator, background)
    else:
        check_add = False
        explainer = shap.TreeExplainer(estimator, background)

    shap_values = np.array(explainer.shap_values(new, check_additivity=check_add))
    return shap_values


def compute_shap_relevance(shap_values, X, Y):
    """[summary]

    Parameters
    ----------
    shap_values : [type]
        [description]
    X : [type]
        [description]
    Y : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    feature_names = X.columns
    task_names = Y.columns

    n_features = len(feature_names)
    n_tasks = len(task_names)

    c = lambda x, y: np.sign(np.diag(matcorr(x, y)))

    signs = Parallel(n_jobs=-1)(
        delayed(c)(X.values, shap_values[y_col, :, :]) for y_col in range(n_tasks)
    )

    signs = np.array(signs).reshape((n_tasks, n_features), order="F")
    signs = pd.DataFrame(signs, index=Y.columns, columns=X.columns)

    shap_relevance = pd.DataFrame(
        np.abs(shap_values).mean(axis=(1)), index=task_names, columns=feature_names
    )

    shap_relevance = shap_relevance * signs
    shap_relevance = shap_relevance.fillna(0.0)

    return shap_relevance


def build_stability_dict(z_mat, scores, alpha=0.05):
    """[summary]

    Parameters
    ----------
    z_mat : [type]
        [description]
    scores : [type]
        [description]
    alpha : float, optional
        [description], by default 0.05

    Returns
    -------
    [type]
        [description]
    """
    support_matrix = np.squeeze(z_mat)
    scores = np.squeeze(scores)

    stab_res = stab.confidenceIntervals(support_matrix, alpha=alpha)
    stability = stab_res["stability"]
    stability_error = stab_res["stability"] - stab_res["lower"]

    res = {
        "scores": scores.tolist(),
        "stability_score": stability,
        "stability_error": stability_error,
        "alpha": alpha,
    }

    return res


def run_stability(model, X, Y, cv, fs, n_jobs, alpha=0.05):
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


def compute_shap(model, X, Y, gpu, test_size=0.3, q="r2"):
    """[summary]

    Parameters
    ----------
    model : [type]
        [description]
    X : [type]
        [description]
    Y : [type]
        [description]
    gpu : [type]
        [description]
    test_size : float, optional
        [description], by default 0.3
    q : str, optional
        [description], by default "r2"

    Returns
    -------
    [type]
        [description]
    """
    X_learn, X_val, Y_learn, Y_val = train_test_split(
        X, Y, test_size=test_size, random_state=42
    )

    model_ = clone(model)
    model_.fit(X_learn, Y_learn)

    shap_values = compute_shap_values(model_, X_learn, X_val, gpu)
    shap_relevances = compute_shap_relevance(shap_values, X_val, Y_val)
    fs = compute_shap_fs(
        shap_relevances, model=model_, X=X_val, Y=Y_val, q=q, by_circuit=True
    )
    fs = fs * 1

    return shap_relevances, fs


def get_quantile_by_circuit(model, X, Y, threshold=0.5):
    """[summary]

    Parameters
    ----------
    model : [type]
        [description]
    X : [type]
        [description]
    Y : [type]
        [description]
    threshold : float, optional
        [description], by default 0.5

    Returns
    -------
    [type]
        [description]
    """
    r = r2_score(Y, model.predict(X), multioutput="raw_values")
    r[r < threshold] = threshold
    q = 0.95 + ((1 - r) / (1 - threshold)) * (1 - 0.95)
    q = {y: q[i] for i, y in enumerate(Y)}

    return q


def compute_shap_fs(relevances, model=None, X=None, Y=None, q="r2", by_circuit=False):
    """[summary]

    Parameters
    ----------
    relevances : [type]
        [description]
    model : [type], optional
        [description], by default None
    X : [type], optional
        [description], by default None
    Y : [type], optional
        [description], by default None
    q : str, optional
        [description], by default "r2"
    by_circuit : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """
    if q == "r2":
        q = get_quantile_by_circuit(model, X, Y)
        by_circuit_frame = pd.concat(
            [
                relevances.loc[p].abs() > np.quantile(relevances.loc[p].abs(), q[p])
                for p in relevances.index
            ],
            axis=1,
        )
        by_circuit_frame = by_circuit_frame.T
    else:
        by_circuit_frame = relevances.abs().apply(
            lambda x: x > np.quantile(x, q), axis=1
        )

    if by_circuit:
        res = by_circuit_frame
    else:
        res = by_circuit_frame.any().values

    return res
