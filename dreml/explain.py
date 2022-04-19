# -*- coding: utf-8 -*-
"""
Explainability module for multi-task framework.
"""

import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from dreml.pystab import nogueria_test


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

    stab_test = nogueria_test(support_matrix, alpha=alpha)

    res = {
        "scores": scores.tolist(),
        "stability_score": stab_test.estimator,
        "stability_error": stab_test.error,
        "alpha": alpha,
    }

    return res


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
