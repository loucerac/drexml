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

from src import stability as stab
from src.models import get_model
from joblib import Parallel, delayed
from scipy.stats import pearsonr
import joblib

import multiprocessing
from functools import partial
import pathlib


def matcorr(O, P):
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
    print(gpu)
    if gpu:
        explainer = shap.GPUTreeExplainer(estimator, background)
    else:
        explainer = shap.TreeExplainer(estimator, background)

    # shap_values = np.zeros((estimator.n_outputs_, new.shape[0], new.shape[1]))
    # for i in np.arange(new.shape[0])[:100:]:
    #    shap_values[:, i, :] = np.array(explainer.shap_values(new.values[i, :]))
    shap_values = np.array(explainer.shap_values(new))
    return shap_values


def compute_shap_relevance(shap_values, X, Y):

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


def build_stability_dict(z_mat, errors, alpha=0.05):

    support_matrix = np.squeeze(z_mat)
    scores = np.squeeze(1 - errors)

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

        r2_loss = 1.0 - r2_score(Y_test, Y_test_filt_preds)
        mo_r2_loss = 1.0 - r2_score(Y_test, Y_test_filt_preds, multioutput="raw_values")

        return (filt_i, r2_loss, mo_r2_loss)

    stab_values = []
    for n_split, split in enumerate(cv):
        stab_values.append(stab_i(model, X, Y, n_split, split))

    for n_split, values in enumerate(stab_values):
        Z[n_split, :] = values[0] * 1
        errors[n_split] = values[1]

    err_by_circuit = [
        pd.Series(values[2], index=Y.columns)
        for n_split, values in enumerate(stab_values)
    ]
    err_by_circuit = pd.concat(err_by_circuit, axis=1).T.describe().T[["mean", "std"]]
    err_by_circuit.columns = "r2_" + err_by_circuit.columns

    stab_by_circuit = {
        y: stab.confidenceIntervals(
            pd.concat([x.loc[y] for x in fs], axis=1).T.values * 1
        )
        for y in Y.columns
    }

    stab_by_circuit = pd.DataFrame(stab_by_circuit).T

    res_by_circuit = pd.concat((stab_by_circuit, err_by_circuit), axis=1)

    res = build_stability_dict(Z, errors, alpha)

    res_df = pd.DataFrame(
        {
            "stability": [res["stability_score"]],
            "lower": [res["stability_score"] - res["stability_error"]],
            "upper": [res["stability_score"] + res["stability_error"]],
            "r2_mean": [np.mean(res["scores"])],
            "r2_std": [np.std(res["scores"])],
        },
        index=["map"],
    )

    res_df = pd.concat((res_df, res_by_circuit), axis=0)

    print(res["stability_score"])

    return res, res_df


def compute_shap(model, X, Y, gpu, test_size=0.3, q="r2"):
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
    r = r2_score(Y, model.predict(X), multioutput="raw_values")
    r[r < threshold] = threshold
    q = 0.95 + ((1 - r) / (1 - threshold)) * (1 - 0.95)
    q = {y: q[i] for i, y in enumerate(Y)}

    return q


def compute_shap_fs(relevances, model=None, X=None, Y=None, q="r2", by_circuit=False):
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


if __name__ == "__main__":
    import sys
    from joblib import parallel_backend

    # client = Client('127.0.0.1:8786')

    _, data_folder, n_iters, gpu, n_jobs, debug = sys.argv

    data_folder = pathlib.Path(data_folder)
    gpu = bool(int(gpu))
    debug = bool(int(debug))
    n_iters = int(n_iters)
    n_jobs = int(n_jobs)

    X_fpath = data_folder.joinpath("features.jbl")
    X = joblib.load(X_fpath)

    Y_fpath = data_folder.joinpath("target.jbl")
    Y = joblib.load(Y_fpath)

    n = 100
    cv = ShuffleSplit(n_splits=n, train_size=0.5, random_state=0)
    cv = list(cv.split(X, Y))
    cv = [(*train_test_split(cv[i][0], test_size=0.3), cv[i][1]) for i in range(n)]

    fname = f"cv.jbl"
    fpath = data_folder.joinpath(fname)
    joblib.dump(cv, fpath)
    n_features = X.shape[1]

    model = get_model(n_features, n_jobs, debug)

    for i, split in enumerate(cv):
        with joblib.parallel_backend("multiprocessing", n_jobs=n_jobs):
            model_ = clone(model)
            model_.set_params(random_state=i)
            model_.fit(X.iloc[split[0], :], Y.iloc[split[0], :])
            fname = f"model_{i}.jbl"
            fpath = data_folder.joinpath(fname)
            joblib.dump(model_, fpath)
            print(f"50-50 split {i}")

    # Define number of GPUs available
    if gpu:
        N_GPU = 3
    else:
        N_GPU = n_jobs

    # filts = Parallel(n_jobs=N_GPU, backend="multiprocessing")(
    #    delayed(runner)(n_split, data_folder) for n_split in range(n))

    gpu_id_list = list(range(N_GPU))

    queue = multiprocessing.Queue()
    # initialize the queue with the GPU ids

    def runner(data_folder, gpu, gpu_id_list, i):
        cpu_name = multiprocessing.current_process().name
        cpu_id = int(cpu_name[cpu_name.find("-") + 1 :]) - 1
        # print(gpu0, gpu)
        gpu_id = queue.get()
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        # print(os.environ["CUDA_VISIBLE_DEVICES"])
        filt_i = None
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            print("device", os.environ["CUDA_VISIBLE_DEVICES"])

            fname = f"cv.jbl"
            fpath = data_folder.joinpath(fname)
            cv = joblib.load(fpath)
            split = cv[i]

            X_fpath = data_folder.joinpath("features.jbl")
            X = joblib.load(X_fpath)

            Y_fpath = data_folder.joinpath("target.jbl")
            Y = joblib.load(Y_fpath)

            X_learn = X.iloc[split[0], :]
            Y_learn = Y.iloc[split[0], :]

            X_val = X.iloc[split[1], :]
            Y_val = Y.iloc[split[1], :]

            fname = f"model_{i}.jbl"
            fpath = data_folder.joinpath(fname)
            model = joblib.load(fpath)

            shap_values = compute_shap_values(model, X_learn, X_val, gpu)
            shap_relevances = compute_shap_relevance(shap_values, X_val, Y_val)
            filt_i = compute_shap_fs(
                shap_relevances, model=model, X=X_val, Y=Y_val, q="r2", by_circuit=True
            )

            fname = f"filt_{i}.jbl"
            fpath = data_folder.joinpath(fname)
            joblib.dump(filt_i, fpath)
            print(gpu_id, filt_i.sum(axis=1).value_counts())

        finally:
            queue.put(gpu_id)

        return filt_i

    # Put indices in queue
    for gpu_ids in range(N_GPU):
        queue.put(gpu_ids)

    r = partial(runner, data_folder, gpu, gpu_id_list)

    with joblib.parallel_backend("multiprocessing", n_jobs=N_GPU):
        fs = joblib.Parallel()(joblib.delayed(r)(n_split) for n_split in range(n))
    joblib.dump(fs, data_folder.joinpath("fs.jbl"))
