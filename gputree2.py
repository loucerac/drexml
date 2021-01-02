#!/usr/bin/env python
# coding: utf-8

import json
import multiprocessing
import os
import pathlib

import dotenv
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed, parallel_backend
from scipy.stats import pearsonr
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import ShuffleSplit, train_test_split

import stability as stab

plt.style.use("ggplot")


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


def compute_shap_fs(relevances, q=0.95, by_circuit=False):

    by_circuit_frame = relevances.abs().apply(lambda x: x > np.quantile(x, q), axis=1)

    if by_circuit:
        res = by_circuit_frame
    else:
        res = by_circuit_frame.any().values

    return res


def compute_shap_values(
    estimator, background, new, approximate=True, check_additivity=False
):
    explainer = shap.GPUTreeExplainer(estimator, background)
    shap_values = np.array(explainer.shap_values(new))

    return shap_values


def compute_shap_relevance(shap_values, X, Y):

    feature_names = X.columns
    task_names = Y.columns

    n_features = len(feature_names)
    n_tasks = len(task_names)

    c = lambda x, y: np.sign(np.diag(matcorr(x, y)))

    corr_sign = lambda x, y: np.sign(pearsonr(x, y)[0])
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


def run_stability(model, X, Y, cv, fs, alpha=0.05):
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

        filt_i = fs[n_split]

        X_train_filt = X_train.loc[:, filt_i]
        X_test_filt = X_test.loc[:, filt_i]

        with parallel_backend("multiprocessing", n_jobs=24):
            sub_model = clone(model)
            sub_model.set_params(**{"max_depth": 32, "max_features": filt_i.sum()})
            # sub_model.set_params(max_features=1.0)
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

    res = build_stability_dict(Z, errors, alpha)
    print(res["stability_score"])

    return res


if __name__ == "__main__":

    project_folder = pathlib.Path(dotenv.find_dotenv()).parent.absolute()
    data_folder = project_folder.joinpath("notebooks", "data")

    X_fpath = data_folder.joinpath("features.pkl")
    X = joblib.load(X_fpath)

    Y_fpath = data_folder.joinpath("target.pkl")
    Y = joblib.load(Y_fpath)

    n = 100
    cv = ShuffleSplit(n_splits=n, train_size=0.5, random_state=0)
    cv = list(cv.split(X, Y))
    cv = [(*train_test_split(cv[i][0], test_size=0.3), cv[i][1]) for i in range(n)]

    fname = f"cv.jbl"
    fpath = pathlib.Path(".", "tmp", fname)
    joblib.dump(cv, fpath)

    n_features = X.shape[1]
    mtry = int(np.sqrt(n_features) + 20)

    model = RandomForestRegressor(
        n_estimators=200, n_jobs=24, max_depth=8, max_features=mtry
    )

    for i, split in enumerate(cv):
        with parallel_backend("multiprocessing", n_jobs=24):
            model_ = clone(model)
            model_.set_params(random_state=i)
            model_.fit(X.iloc[split[0], :], Y.iloc[split[0], :])
            fname = f"model_{i}.jbl"
            fpath = pathlib.Path(".", "tmp", fname)
            joblib.dump(model_, fpath)

    # Define number of GPUs available
    N_GPU = 3

    

    # filts = Parallel(n_jobs=N_GPU, backend="multiprocessing")(
    #    delayed(runner)(n_split, data_folder) for n_split in range(n))

    gpu_id_list = list(range(N_GPU))

    queue = multiprocessing.Queue()
    # initialize the queue with the GPU ids

    def runner(data_folder, gpu_id_list, i):
        cpu_name = multiprocessing.current_process().name
        cpu_id = int(cpu_name[cpu_name.find("-") + 1 :]) - 1
        # print(gpu0, gpu)
        gpu_id = queue.get()
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        # print(os.environ["CUDA_VISIBLE_DEVICES"])
        filt_i = None
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            print(os.environ["CUDA_VISIBLE_DEVICES"])

            fname = f"cv.jbl"
            fpath = pathlib.Path(".", "tmp", fname)
            cv = joblib.load(fpath)
            split = cv[i]

            X_fpath = data_folder.joinpath("features.pkl")
            X = joblib.load(X_fpath)

            Y_fpath = data_folder.joinpath("target.pkl")
            Y = joblib.load(Y_fpath)

            X_learn = X.iloc[split[0], :]
            Y_learn = Y.iloc[split[0], :]

            X_val = X.iloc[split[1], :]
            Y_val = Y.iloc[split[1], :]

            fname = f"model_{i}.jbl"
            fpath = pathlib.Path(".", "tmp", fname)
            model = joblib.load(fpath)

            shap_values = compute_shap_values(model, X_learn, X_val)

            shap_relevances = compute_shap_relevance(shap_values, X_val, Y_val)
            filt_i = compute_shap_fs(shap_relevances, q=0.95, by_circuit=False)

            fname = f"filt_{i}.jbl"
            fpath = pathlib.Path(".", "tmp", fname)
            joblib.dump(filt_i, fpath)

        finally:
            queue.put(gpu_id)

        return filt_i

    # Put indices in queue
    for gpu_ids in range(N_GPU):
        queue.put(gpu_ids)

    r = partial(runner, data_folder, gpu_id_list)

    with parallel_backend("multiprocessing", n_jobs=N_GPU):
        fs = Parallel()(delayed(r)(n_split) for n_split in range(n))

    res = run_stability(model, X, Y, cv, fs)

    with open("results/stability.json", "w") as fjson:
        json.dump(res, fjson, indent=4)
