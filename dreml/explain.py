# -*- coding: utf-8 -*-
"""
Explainability module for multi-task framework.
"""

import os

import joblib
import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from dreml.pystab import nogueria_test


def matcorr(O, P):
    """Fast correlation matrix computation.

    Parameters
    ----------
    O : ndarray
        A matrix of observations.
    P : ndarray
        A matrix of predictions.

    Returns
    -------
    ndarray
        The cross-correlation matrix.
    """
    n = O.shape[0]

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


def compute_shap_values_(x, explainer, check_add, gpu_id, gpu, n_devices):
    """Partial function to compute the shap values."""

    if gpu and n_devices > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(gpu_id)

    return np.array(explainer.shap_values(x, check_additivity=check_add))


def compute_shap_values(estimator, background, new, gpu, n_devices=1):
    """SHAP values for a given dataset and model.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        The model to explain the data.
    background : pandas.DataFrame [n_samples_bg, n_features]
        The background dataset.
    new : pandas.DataFrame [n_samples_new, n_features]
        The dataset to explain.
    gpu : bool
        Whether to use GPU or not.
    split : bool, optional
        Whether to split the dataset or not.

    Returns
    -------
    ndarray [n_samples_new, n_features, n_tasks]
        The SHAP values.
    """
    if background.shape[0] > 1000:
        background = shap.sample(background, nsamples=999, random_state=0)
    if gpu:
        check_add = True
        explainer = shap.GPUTreeExplainer(estimator, background)
    else:
        check_add = False
        explainer = shap.TreeExplainer(estimator, background)

    chunk_size = len(new) // n_devices + 1
    new_gb = new.groupby(np.arange(len(new)) // chunk_size)
    with joblib.parallel_backend("loky", n_jobs=n_devices):
        shap_values = joblib.Parallel()(
            joblib.delayed(compute_shap_values_)(
                x=gb[1],
                explainer=explainer,
                check_add=check_add,
                gpu_id=i,
                gpu=gpu,
                n_devices=n_devices,
            )
            for i, gb in enumerate(new_gb)
        )
    shap_values = np.concatenate(shap_values, axis=0)

    return shap_values


def compute_corr_sign(x, y):
    """Coompute the correlation sign.

    Parameters
    ----------
    x : ndarray [n_samples, n_features]
        The feature dataset.
    y : ndarray [n_samples, n_tasks]
        The task dataset.

    Returns
    -------
    ndarray [n_features, n_tasks]
        SHAP feature-task (linear) interaction sign.
    """
    return np.sign(np.diag(matcorr(x, y)))


def compute_shap_relevance(shap_values, X, Y):
    """Convert the SHAP values to relevance scores.

    Parameters
    ----------
    shap_values : ndarray [n_samples_new, n_features, n_tasks]
        The SHAP values.
    X : pandas.DataFrame [n_samples, n_features]
        The feature dataset to explain.
    Y : pandas.DataFrame [n_samples, n_tasks]
        The task dataset to explain.

    Returns
    -------
    pandas.DataFrame [n_features, n_tasks]
        The task-wise feature relevance scores.
    """
    feature_names = X.columns
    task_names = Y.columns

    n_features = len(feature_names)
    n_tasks = len(task_names)

    if shap_values.ndim < 3:
        shap_values = np.expand_dims(shap_values, axis=0)
    print(X.shape, Y.shape, shap_values.shape)

    signs = Parallel(n_jobs=-1)(
        delayed(compute_corr_sign)(X.values, shap_values[y_col])
        for y_col in range(n_tasks)
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
    """Adapt NogueiraTest to old version of dreml (use dicts).

    Parameters
    ----------
    z_mat : ndarray [n_model_samples, n_features]
        The stability matrix.
    scores : ndarray [n_model_samples]
        The metric scores over the test sets.
    alpha : float, optional
        Signficance level for Nogueira's test, by default 0.05

    Returns
    -------
    dict
        Dictionary with the Nogueira test results and test metric scores.
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


def compute_shap(model, X, Y, gpu, test_size=0.3, q="r2", n_devices=1):
    """Compute relevance KDT-signalization scores for a given model.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        The model to explain the data.
    X : pandas.DataFrame [n_samples, n_features]
        The dataset to explain.
    Y : pandas.DataFrame [n_samples, n_tasks]
        The dataset to explain.
    gpu : bool
        Whether to use GPU or not.
    test_size : float, optional
        The proportion of the dataset to use for the test set.
    q : str, optional
        The quality measure to use. Either "r2" or "mse".

    Returns
    -------
    pandas.DataFrame [n_features, n_tasks]
        The relevance scores.
    pandas.DataFrame [n_features, n_tasks]
        The features selected for each task.
    """
    X_learn, X_val, Y_learn, Y_val = train_test_split(
        X, Y, test_size=test_size, random_state=42
    )

    model_ = clone(model)
    model_.fit(X_learn, Y_learn)

    shap_values = compute_shap_values(model_, X_learn, X_val, gpu, n_devices=n_devices)
    shap_relevances = compute_shap_relevance(shap_values, X_val, Y_val)
    fs = compute_shap_fs(
        shap_relevances, model=model_, X=X_val, Y=Y_val, q=q, by_circuit=True
    )
    fs = fs * 1

    return shap_relevances, fs


def get_quantile_by_circuit(model, X, Y, threshold=0.5):
    """Get the selection quantile of the model by circuit (or globally). Select features
    whose relevance score is above said quantile.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        Fited model.
    X : pandas.DataFrame [n_samples, n_features]
        The feature dataset to explain.
    Y : pandas.DataFrame [n_samples, n_tasks]
        The task dataset to explain.
    threshold : float, optional
        Theshold to use to dicriminate ill-conditioned circuits when performing feature
        selection, by default 0.5

    Returns
    -------
    float
        Qauntile to use.
    """
    r = r2_score(Y, model.predict(X), multioutput="raw_values")
    r[r < threshold] = threshold
    q = 0.95 + ((1 - r) / (1 - threshold)) * (1 - 0.95)
    q = {y: q[i] for i, y in enumerate(Y)}

    return q


def compute_shap_fs(relevances, model=None, X=None, Y=None, q="r2", by_circuit=False):
    """Compute the feature selection scores.

    Parameters
    ----------
    relevances : pandas.DataFrame [n_features, n_tasks]
        The relevance scores.
    model : sklearn.base.BaseEstimator, optional
        The model to explain the data.
    X : pandas.DataFrame [n_samples, n_features], optional
        The feature dataset to explain, by default None.
    Y : pandas.DataFrame [n_samples, n_tasks], optional
        The task dataset to explain, by default None
    q : float or str, optional
        Either a metric string to discriminate fs tasks or predefined quantile, by
        default "r2"
    by_circuit : bool, optional
        Feature selection by circuit or globally, by default False

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
