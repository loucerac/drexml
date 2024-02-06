# -*- coding: utf-8 -*-
"""
Explainability module for multi-task framework.
"""
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import r2_score

from drexml.pystab import nogueria_test


def matcorr(features, targets):
    """Fast correlation matrix computation.

    Parameters
    ----------
    features : ndarray [n_samples, n_features]
        A matrix of observations.
    targets : ndarray [n_samples, n_tasks]
        A matrix of predictions.

    Returns
    -------
    ndarray
        The correlation matrix.
    """
    n = features.shape[0]

    features_center = features - (
        np.einsum("nt->t", features, optimize="optimal") / np.double(n)
    )
    targets_center = targets - (
        np.einsum("nm->m", targets, optimize="optimal") / np.double(n)
    )

    cov = np.einsum("nm,nt->mt", targets_center, features_center, optimize="optimal")

    targets_var = np.einsum(
        "nm,nm->m", targets_center, targets_center, optimize="optimal"
    )
    features_var = np.einsum(
        "nt,nt->t", features_center, features_center, optimize="optimal"
    )
    tmp = np.einsum("m,t->mt", targets_var, features_var, optimize="optimal")

    return cov / np.sqrt(tmp)


def compute_shap_values_(x, explainer, check_add, gpu_id=None):
    """
    Partial function to compute the SHAP values.

    Parameters
    ----------
    x : ndarray [n_samples, n_features]
        The feature dataset.
    explainer : shap.TreeExplainer or shap.GPUTreeExplainer
        The SHAP explainer.
    check_add : bool
        Check if the SHAP values add up to the model output.
    gpu_id : int
        The GPU ID.

    Returns
    -------
    shap_values : ndarray [n_samples, n_features, n_tasks]
        The SHAP values.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    shap_values = np.array(explainer.shap_values(x, check_additivity=check_add))

    if shap_values.ndim < 3:
        shap_values = np.expand_dims(shap_values, axis=0)

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
    """Adapt NogueiraTest to old version of drexml (use dicts).

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


def get_quantile_by_circuit(model, X, Y, threshold=0.5):
    """Get the selection quantile of the model by circuit (or globally). Select features
    whose relevance score is above said quantile.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        Fitted model.
    X : pandas.DataFrame [n_samples, n_features]
        The feature dataset to explain.
    Y : pandas.DataFrame [n_samples, n_tasks]
        The task dataset to explain.
    threshold : float, optional
        Threshold to use to discriminate ill-conditioned circuits when performing feature
        selection, by default 0.5

    Returns
    -------
    float
        Qauntile to use.
    """
    cut = 0.95
    r = r2_score(Y, model.predict(X), multioutput="raw_values")
    r[r < threshold] = threshold
    q = cut + ((1 - r) / (1 - threshold)) * (1 - cut)
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
    pandas.Series [n_features]
        The feature selection scores.

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
