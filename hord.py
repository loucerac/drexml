#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Carlos Loucera <carlos.loucera@juntadeandalucia.es>
Author: Maria Pena Chilet <maria.pena.chilet.ext@juntadeandalucia.es>
Author: Marina Esteban <marina.estebanm@gmail.com>

Entry CLI point for HORD multi-task framework.
"""

import os
import warnings
from pathlib import Path

import click
import numpy as np
import pandas as pd
import shap
from dotenv import find_dotenv, load_dotenv
from sklearn import metrics
import joblib
from sklearn.model_selection import RepeatedKFold
import json

from src.datasets import get_disease_data
from src.learn import AutoMorf
from src.explain import run_stability, compute_shap
from src import ml_plots
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib


def warn(*args, **kwargs):
    pass


warnings.warn = warn

dotenv_filepath = find_dotenv()
load_dotenv(dotenv_filepath)
project_path = os.path.dirname(dotenv_filepath)

DATA_PATH = Path(os.environ.get("DATA_PATH"))
NUM_CPUS = int(os.getenv("NUM_CPUS"))
USE_GPU = bool(os.getenv("USE_GPU"))
OUT_PATH = Path(os.environ.get("OUT_PATH"))

warnings.filterwarnings("ignore", category=DeprecationWarning, module="sklearn")


@click.command()
@click.option("--disease", default="fanconi", help="which disease to test")
@click.option("--mlmodel", default="morf", help="ML model")
@click.option("--opt", default="hyperopt", help="HP-optimization method.")
@click.option("--seed", default=42, type=int, help="Random seed")
@click.option("--mode", default="train", help="Train and evaluate or evaluate")
@click.option("--pathways", default=None, help="Pathways filter", multiple=True)
@click.option("--gset", default="target", help="Set of genes to use")
def hord(disease, mlmodel, opt, seed, mode, pathways, gset):
    """HORD multi-task module.

    Parameters
    ----------
    disease : str
        Disease to train/test.
    mlmodel : str
        Which ML model to use.
    opt : str
        Select optimization mode ["sko", "hpo", None]
    seed : int
        Seed for random number generator.
    mode : bool
        Train or load a pre-trained model.
    pathways : str list
        Which pathways to use as the ML target.
    gset : str
        Which set of genes must be selected as the ML input.
    """

    print("Working on disease {}".format(disease))

    run_(disease, mlmodel, opt, seed, mode, pathways, gset)

    print(get_out_path(disease, mlmodel, opt, seed, mode, pathways, gset))


def get_out_path(disease, mlmodel, opt, seed, mode, pathways, gset):
    """Construct the path where the model must be saved.

    Returns
    -------
    pathlib.Path
        The desired path.
    """

    env_possible = Path(disease)

    if env_possible.exists() and (env_possible.suffix == ".env"):
        print("Working with experiment {}".format(env_possible.stem))
        out_path = env_possible.parent.joinpath("ml", mlmodel + "_" + mode)
    else:
        if not len(pathways):
            name = ["all"]
        else:
            name = pathways
        name = "_".join(name)

        out_path = OUT_PATH.joinpath(disease, name, gset, mlmodel, opt, mode, str(seed))

    out_path.mkdir(parents=True, exist_ok=True)
    print("Storage folder: {}".format(out_path))

    return out_path


def run_(disease, mlmodel, opt, seed, mode, pathways, gset):
    """Select the training mode."""
    if mode in ["train", "test"]:
        run_full(disease, mlmodel, opt, seed, mode, pathways, gset)


def get_data(disease, mode, pathways, gset):
    """Load disease data and metadata."""
    gene_xpr, pathvals, circuits, genes, clinical = get_disease_data(
        disease, pathways, gset
    )

    print(gene_xpr.shape, pathvals.shape)

    if mode == "test":
        gene_xpr = gene_xpr.iloc[:100, :]
        pathvals = pathvals.iloc[:100, :]

    return gene_xpr, pathvals, circuits, genes, clinical


def run_full(disease, mlmodel, opt, seed, mode, pathways, gset):
    """Full model training, with hyperparametr optimization, unbiased CV
    performance estimation and relevance computation.
    """

    output_folder = get_out_path(disease, mlmodel, opt, seed, mode, pathways, gset)

    # Load data
    gene_xpr, pathvals, circuits, genes, clinical = get_data(
        disease, mode, pathways, gset
    )

    # Get ML model
    model = get_model(mlmodel, opt, mode)

    # Optimize and fit using the whole data
    model.fit(gene_xpr, pathvals)
    model.save(output_folder, fmt="json")

    estimator = model.best_model

    # Save global relevances
    print("Computing global relevance.")
    relevance_fname = "model_global_relevance.tsv"
    rel_fpath = output_folder.joinpath(relevance_fname)
    rel = estimator.feature_importances_
    rel_df = pd.DataFrame({"relevance": rel}, index=gene_xpr.columns)
    rel_df.to_csv(rel_fpath, sep="\t")

    # CV with optimal hyperparameters (unbiased performance)
    cv_stats = perform_cv(gene_xpr, pathvals, estimator, seed, mode, clinical.tissue)

    # Save results
    stats_fname = "cv_stats.pkl"
    stats_fpath = output_folder.joinpath(output_folder, stats_fname)
    with open(stats_fpath, "wb") as f:
        joblib.dump(cv_stats, f)
    print("Unbiased CV stats saved to: {}".format(stats_fpath))

    # Compute shap relevances
    shap_summary, fs = compute_shap(estimator, gene_xpr, pathvals)
    
    # Save results
    shap_summary_fname = "shap_summary.tsv"
    shap_summary_fpath = output_folder.joinpath(output_folder, shap_summary_fname)
    shap_summary.to_csv(shap_summary_fpath, sep="\t")
    print("Shap summary results saved to: {}".format(shap_summary_fpath))
    # Save results
    fs_fname = "shap_selection.tsv"
    fs_fpath = output_folder.joinpath(output_folder, fs_fname)
    fs.to_csv(fs_fpath, sep="\t")
    print("Shap selection results saved to: {}".format(shap_summary_fpath))

    plot(output_folder, gene_xpr.columns, pathvals.columns, cv_stats)

    # Stability Analysys
    if mode == "test":
        joblib.dump(estimator, "/home/cloucera/results/hord/estimator.jbl")
        joblib.dump(gene_xpr, "/home/cloucera/results/hord/gene_xpr.jbl")
        joblib.dump(pathvals, "/home/cloucera/results/hord/pathvals.jbl")

    stability_results = run_stability(estimator, gene_xpr, pathvals, alpha=0.05)
    # Save results
    stability_results_fname = "stability_results.json"
    stability_results_fpath = output_folder.joinpath(
        output_folder, stability_results_fname
    )
    with open(stability_results_fpath, "w") as f:
        json.dump(stability_results, f)
    print("Stability results saved to: {}".format(stability_results_fpath))
    print(stability_results["stability_score"])
    print(stability_results["stability_error"])


def get_model(mlmodel, opt, mode):
    """Get an instace of an AutoMorf model."""
    name = "_".join([mlmodel, opt])
    if mlmodel == "morf":
        if mode == "train":
            model = AutoMorf(
                name=name, framework=opt, n_jobs=NUM_CPUS, cv=5, n_calls=10 ** 3
            )
        elif mode == "test":
            model = AutoMorf(name=name, framework=opt, n_jobs=NUM_CPUS, cv=2, n_calls=5)
    else:
        raise NotImplementedError()

    return model


def perform_cv(X, y, estimator, seed, mode, tissue=None):
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

    n_repeats = 5 if mode == "test" else 10
    n_splits = 2 if mode == "test" else 10
    skf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    iter_ = 0
    for train_index, test_index in skf.split(X):
        iter_ = iter_ + 1
        print("RepeatedKFold iteration {}. ".format(iter_))

        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index, :], y.iloc[test_index, :]

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
    #_, folder, use_task, use_circuit_dict = sys.argv

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
