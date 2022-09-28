#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Carlos Loucera <carlos.loucera@juntadeandalucia.es>
Author: Maria Pena Chilet <maria.pena.chilet.ext@juntadeandalucia.es>
Author: Marina Esteban <marina.estebanm@gmail.com>

Entry CLI point for HORD multi-task framework.
"""

import json
import os
import pathlib
import subprocess
import warnings
from pathlib import Path

import click
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dotenv import find_dotenv, load_dotenv
from sklearn import metrics
from sklearn.base import clone
from sklearn.model_selection import RepeatedKFold

from scripts import ml_plots
from drexml.datasets import get_disease_data
from drexml.explain import compute_shap, run_stability
from drexml.models import get_model


def warn(*args, **kwargs):
    pass


warnings.warn = warn

dotenv_filepath = find_dotenv()
load_dotenv(dotenv_filepath)
project_path = os.path.dirname(dotenv_filepath)

warnings.filterwarnings("ignore", category=DeprecationWarning, module="sklearn")


@click.command()
@click.option("--disease", type=click.Path(exists=True), help="Experiment path.")
@click.option("--n-iters", default=100, help="Number of Optimization iterations.")
@click.option(
    "--gpu/--no-gpu", is_flag=True, default=True, help="Flag to use CUDA-enabled GPUs."
)
@click.option("--n-jobs", default=-1, help="Number of jobs, -1 uses all devices.")
@click.option("--debug", is_flag=True, default=False, help="Flag to run in debug mode.")
@click.version_option("1.0.0")
def hord(disease, n_iters, gpu, n_jobs, debug):
    print(
        "Working on disease {} {} {} {} {}".format(disease, n_iters, gpu, n_jobs, debug)
    )

    run_(disease, n_iters, gpu, n_jobs, debug)

    print(get_out_path(disease, n_iters, gpu, n_jobs, debug))


def get_out_path(disease, n_iters, gpu, n_jobs, debug):
    """Construct the path where the model must be saved.

    Returns
    -------
    pathlib.Path
        The desired path.
    """

    env_possible = Path(disease)

    if env_possible.exists() and (env_possible.suffix == ".env"):
        print("Working with experiment {}".format(env_possible.parent.name))
        out_path = env_possible.parent.joinpath("ml")
    else:
        raise NotImplementedError("Use experiment")
    if debug:
        out_path.joinpath("debug")
    out_path.mkdir(parents=True, exist_ok=True)
    print("Storage folder: {}".format(out_path))

    return out_path


def get_data(disease, debug, scale=True):
    """Load disease data and metadata."""
    gene_xpr, pathvals, circuits, genes = get_disease_data(disease)

    if scale:
        from sklearn.preprocessing import MinMaxScaler

        pathvals = pd.DataFrame(
            MinMaxScaler().fit_transform(pathvals),
            columns=pathvals.columns,
            index=pathvals.index,
        )

    print(gene_xpr.shape, pathvals.shape)

    if debug:
        size = 100
        gene_xpr = gene_xpr.iloc[:size, :]
        pathvals = pathvals.iloc[:size, :]

    return gene_xpr, pathvals, circuits, genes


def run_(disease, n_iters, gpu, n_jobs, debug):
    """Full model training, with hyperparametr optimization, unbiased CV
    performance estimation and relevance computation.
    """

    output_folder = get_out_path(disease, n_iters, gpu, n_jobs, debug)
    data_folder = output_folder.joinpath("tmp")
    data_folder.mkdir(parents=True, exist_ok=True)

    # Load data
    gene_xpr, pathvals, circuits, genes = get_data(disease, debug)
    joblib.dump(gene_xpr, data_folder.joinpath("features.jbl"))
    joblib.dump(pathvals, data_folder.joinpath("target.jbl"))

    n_features = gene_xpr.shape[1]

    # Get ML model
    estimator = get_model(n_features, n_jobs, debug)

    # fs, cv = build_shap_fs(estimator, gene_xpr, pathvals, output_folder, gpu)
    cmd = [
        "python",
        "drexml/explain.py",
        data_folder.as_posix(),
        str(n_iters),
        str(int(gpu)),
        str(n_jobs),
        str(int(debug)),
    ]
    subprocess.Popen(cmd).wait()
    # from dreml.explain2 import run_gpu
    # fs, cv = run_gpu(data_folder, n_iters, gpu, n_jobs, debug)
    fs = joblib.load(data_folder.joinpath("fs.jbl"))
    cv = joblib.load(data_folder.joinpath("cv.jbl"))
    stability_results, stab_res_df = run_stability(
        estimator, gene_xpr, pathvals, cv, fs, n_jobs, alpha=0.05
    )
    # Save results
    stability_results_fname = "stability_results.json"
    stability_results_fpath = output_folder.joinpath(stability_results_fname)
    with open(stability_results_fpath, "w") as fjson:
        json.dump(stability_results, fjson, indent=4)
    print("Stability results saved to: {}".format(stability_results_fpath))
    print(stability_results["stability_score"])
    print(stability_results["stability_error"])

    # Compute shap relevances
    shap_summary, fs = compute_shap(estimator, gene_xpr, pathvals, gpu)

    # Save results
    shap_summary_fname = "shap_summary.tsv"
    shap_summary_fpath = output_folder.joinpath(shap_summary_fname)
    shap_summary.to_csv(shap_summary_fpath, sep="\t")
    print("Shap summary results saved to: {}".format(shap_summary_fpath))

    # Save results
    fs_fname = "shap_selection.tsv"
    fs_fpath = output_folder.joinpath(fs_fname)
    fs.to_csv(fs_fpath, sep="\t")
    print("Shap selection results saved to: {}".format(fs_fpath))

    # CV with optimal hyperparameters (unbiased performance)
    genes_selected = fs.any(axis=0).index
    features = gene_xpr.loc[:, genes_selected]
    rf_params = {"max_features": 1.0}
    estimator.set_params(**rf_params)
    cv_stats = perform_cv(features, pathvals, estimator, debug)

    # Save results
    stats_fname = "cv_stats.pkl"
    stats_fpath = output_folder.joinpath(stats_fname)
    with open(stats_fpath, "wb") as f:
        joblib.dump(cv_stats, f)
    print("Unbiased CV stats saved to: {}".format(stats_fpath))

    tsv_stability_results_fname = "performance_stability_results.tsv"
    tsv_res_results_fpath = output_folder.joinpath(tsv_stability_results_fname)

    cvmo = pd.DataFrame(cv_stats["r2_mo"]["test"], columns=pathvals.columns)
    cvmo = cvmo.describe().T[["mean", "std", "25%", "75%"]]
    cvmo.columns = "r2_cv_" + cvmo.columns
    cvmap = pd.DataFrame(cv_stats["r2_ua"]["test"])
    cvmap = cvmap.describe().T[["mean", "std", "25%", "75%"]]
    cvmap.columns = "r2_cv_" + cvmap.columns
    cvmap.index = ["map"]
    cv_df = pd.concat((cvmap, cvmo), axis=0)
    stab_res_df = pd.concat((stab_res_df, cv_df), axis=1)

    stab_res_df.to_csv(tsv_res_results_fpath, sep="\t")
    circuit_dict = ml_plots.get_circuit_dict()
    ml_plots.convert_frame_ids(
        tsv_stability_results_fname, output_folder, circuit_dict, frame=stab_res_df
    )

    plot(output_folder, features.columns, pathvals.columns, cv_stats)


def perform_cv(X, y, model, debug, n_jobs=-1):
    """Unbiased performance estimation.

    Parameters
    ----------
    X : array-like, shape = (n_samples, n_features)
        Test samples. For some estimators this may be a
        precomputed kernel matrix instead, shape = (n_samples,
        n_samples_fitted], where n_samples_fitted is the number of
        samples used in the fitting for the estimator.
    y : (n_samples, n_outputs)
        True values for X.
    estimator : scikit-learn estimator.
        The model to test its performance.
    seed : int
        Random seed.
    tissue : array-like, [n_samples, ]
        A categorical variable to use for CV stratification.

    Returns
    -------
    dict
        A dictionary with per fold regression stats.
    """

    stats = {
        "evs_mo": {"train": [], "test": []},
        "evs_ua": {"train": [], "test": []},
        "mae_mo": {"train": [], "test": []},
        "mae_ua": {"train": [], "test": []},
        "mse_mo": {"train": [], "test": []},
        "mse_ua": {"train": [], "test": []},
        "msle_mo": {"train": [], "test": []},
        "msle_ua": {"train": [], "test": []},
        "r2_mo": {"train": [], "test": []},
        "r2_ua": {"train": [], "test": []},
        "relevance": [],
    }

    n_repeats = 5 if debug else 10
    n_splits = 2 if debug else 10
    skf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)

    iter_ = 0
    for train_index, test_index in skf.split(X):
        iter_ = iter_ + 1
        print("RepeatedKFold iteration {}. ".format(iter_))

        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index, :], y.iloc[test_index, :]

        estimator = clone(model)
        with joblib.parallel_backend("multiprocessing", n_jobs=n_jobs):
            estimator.fit(X_train, y_train)

        y_train_hat = estimator.predict(X_train)
        y_test_hat = estimator.predict(X_test)

        # metrics computation

        # Explained variance

        evs_mo_train = metrics.explained_variance_score(
            y_train, y_train_hat, multioutput="raw_values"
        )
        stats["evs_mo"]["train"].append(evs_mo_train)

        evs_mo_test = metrics.explained_variance_score(
            y_test, y_test_hat, multioutput="raw_values"
        )
        stats["evs_mo"]["test"].append(evs_mo_test)

        evs_ua_train = metrics.explained_variance_score(
            y_train, y_train_hat, multioutput="uniform_average"
        )
        stats["evs_ua"]["train"].append(evs_ua_train)

        evs_ua_test = metrics.explained_variance_score(
            y_test, y_test_hat, multioutput="uniform_average"
        )
        stats["evs_ua"]["test"].append(evs_ua_test)

        # MAE

        mae_mo_train = metrics.mean_absolute_error(
            y_train, y_train_hat, multioutput="raw_values"
        )
        stats["mae_mo"]["train"].append(mae_mo_train)

        mae_mo_test = metrics.mean_absolute_error(
            y_test, y_test_hat, multioutput="raw_values"
        )
        stats["mae_mo"]["test"].append(mae_mo_test)

        mae_ua_train = metrics.mean_absolute_error(
            y_train, y_train_hat, multioutput="uniform_average"
        )
        stats["mae_ua"]["train"].append(mae_ua_train)

        mae_ua_test = metrics.mean_absolute_error(
            y_test, y_test_hat, multioutput="uniform_average"
        )
        stats["mae_ua"]["test"].append(mae_ua_test)

        # MSE

        mse_mo_train = metrics.mean_squared_error(
            y_train, y_train_hat, multioutput="raw_values"
        )
        stats["mse_mo"]["train"].append(mse_mo_train)

        mse_mo_test = metrics.mean_squared_error(
            y_test, y_test_hat, multioutput="raw_values"
        )
        stats["mse_mo"]["test"].append(mse_mo_test)

        mse_ua_train = metrics.mean_squared_error(
            y_train, y_train_hat, multioutput="uniform_average"
        )
        stats["mse_ua"]["train"].append(mse_ua_train)

        mse_ua_test = metrics.mean_squared_error(
            y_test, y_test_hat, multioutput="uniform_average"
        )
        stats["mse_ua"]["test"].append(mse_ua_test)

        # MSLE

        msle_mo_train = metrics.mean_squared_log_error(
            y_train, y_train_hat, multioutput="raw_values"
        )
        stats["msle_mo"]["train"].append(msle_mo_train)

        msle_mo_test = metrics.mean_squared_log_error(
            y_test, y_test_hat, multioutput="raw_values"
        )
        stats["msle_mo"]["test"].append(msle_mo_test)

        msle_ua_train = metrics.mean_squared_log_error(
            y_train, y_train_hat, multioutput="uniform_average"
        )
        stats["msle_ua"]["train"].append(msle_ua_train)

        msle_ua_test = metrics.mean_squared_log_error(
            y_test, y_test_hat, multioutput="uniform_average"
        )
        stats["msle_ua"]["test"].append(msle_ua_test)

        # r2

        r2_mo_train = metrics.r2_score(y_train, y_train_hat, multioutput="raw_values")
        stats["r2_mo"]["train"].append(r2_mo_train)

        r2_mo_test = metrics.r2_score(y_test, y_test_hat, multioutput="raw_values")
        stats["r2_mo"]["test"].append(r2_mo_test)

        r2_ua_train = metrics.r2_score(
            y_train, y_train_hat, multioutput="uniform_average"
        )
        stats["r2_ua"]["train"].append(r2_ua_train)

        r2_ua_test = metrics.r2_score(y_test, y_test_hat, multioutput="uniform_average")
        stats["r2_ua"]["test"].append(r2_ua_test)

        if hasattr(estimator, "feature_importances_"):
            stats["relevance"].append(estimator.feature_importances_)
        if hasattr(estimator, "coef_"):
            stats["relevance"].append(estimator.coef_)

    return stats


def plot(results_path, gene_ids, circuit_ids, cv_stats):
    # _, folder, use_task, use_circuit_dict = sys.argv

    plt.style.use("fivethirtyeight")
    sns.set_context("paper")
    results_path = pathlib.Path(results_path)

    # load data
    rel_cv = ml_plots.get_rel_cv(cv_stats, gene_ids)
    cut = ml_plots.get_cut_point(rel_cv)
    gene_symbols_dict = ml_plots.get_symbol_dict()
    circuit_dict = ml_plots.get_circuit_dict()

    ## Median relevance
    for use_symb_dict in [True, False]:
        ml_plots.plot_median(rel_cv, use_symb_dict, gene_symbols_dict, results_path)

    ml_plots.save_median_df(rel_cv, cut, gene_symbols_dict, results_path)

    ## Relevance distribution
    for symb_dict in [None, gene_symbols_dict]:
        ml_plots.plot_relevance_distribution(rel_cv, cut, symb_dict, results_path)

    ## ML stats
    ml_plots.plot_stats(cv_stats, circuit_ids, circuit_dict, results_path)

    for fname in ["shap_summary.tsv", "shap_selection.tsv"]:
        ml_plots.convert_frame_ids(fname, results_path, circuit_dict, gene_symbols_dict)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    hord()
