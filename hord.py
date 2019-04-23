#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Carlos Loucera <carlos.loucera@juntadeandalucia.es>
Author: Maria Pena Chilet <maria.pena.chilet.ext@juntadeandalucia.es>
Author: Marina Esteban <marina.estebanm@gmail.com>

Entry CLI point for HORD multi-task framework.
"""

import warnings
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.metrics import average_precision_score
import sklearn
import numpy as np
from src.learn import AutoMorf
from src.datasets import get_disease_data, load_clinical_data
import traceback
import pickle
import click
from datetime import datetime
from pathlib import Path
import os


def warn(*args, **kwargs):
    pass


warnings.warn = warn


dotenv_filepath = find_dotenv()
load_dotenv(dotenv_filepath)
project_path = os.path.dirname(dotenv_filepath)

DATA_PATH = Path(os.environ.get("DATA_PATH"))
NUM_CPUS = int(os.getenv("NUM_CPUS"))
USE_GPU = bool(os.getenv("USE_GPU"))
warnings.filterwarnings(
    'ignore', category=DeprecationWarning, module='sklearn')


@click.command()
@click.option('--disease', default="fanconi", help='which disease to test')
@click.option('--mlmodel', default="morf", help='ML model')
@click.option('--opt', default="hyperopt", help='Train/test mode')
@click.option('--seed', default=42, type=int, help='Random seed')
@click.option("--mode", default="train", help="Train and evaluate or evaluate")
def hord(disease, mlmodel, opt, seed, mode):
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
    """

    print("Working on disease {}".format(disease))

    run_(disease, mlmodel, opt, seed, mode)

    exit(0)


def get_out_path(disease, mlmodel, opt, seed, mode):

    out_path = DATA_PATH.joinpath("out", disease, mlmodel, opt, mode, str(seed))
    if mode == "train":
        ok = False
    elif mode == "test":
        ok = True
    out_path.mkdir(parents=True, exist_ok=ok)

    return out_path


def run_(disease, mlmodel, opt, seed, mode):
    if mode in ["train", "test"]:
        run_full(disease, mlmodel, opt, seed, mode)

def get_data(disease, mode):
    gene_xpr, pathvals, circuits, genes, clinical = get_disease_data(disease)

    if mode == "test":
        gene_xpr = gene_xpr.iloc[:100, :]
        pathvals = pathvals.iloc[:100, :]

    return gene_xpr, pathvals, circuits, genes, clinical


def run_full(disease, mlmodel, opt, seed, mode):
    from sklearn.model_selection import RepeatedStratifiedKFold

    output_folder = get_out_path(disease, mlmodel, opt, seed, mode)

    # Load data
    gene_xpr, pathvals, circuits, genes, clinical = get_data(disease, mode)

    # Get ML model
    model = get_model(mlmodel, opt, mode)

    # Optimize and fit using the whole data
    model.fit(gene_xpr, pathvals)
    model.save(output_folder)

    # CV with optimal hyperparameters (unbiased performance)
    estimator = model.best_model
    cv_stats = perform_cv(gene_xpr, pathvals, estimator, seed, clinical.tissue)

    # Save results
    stats_fname = "cv_stats.pkl"
    stats_fpath = os.path.join(output_folder, stats_fname)
    with open(stats_fpath, "wb") as f:
        joblib.dump(cv_stats, f)
    print("saved to: {}".format(output_folder))


def get_model(mlmodel, opt, mode):
    name = "_".join([mlmodel, opt])
    if mlmodel == "morf":
        if mode == "train":
            model = AutoMorf(
                name=name, 
                framework=opt,
                n_jobs=NUM_CPUS,
                cv=5,
                n_calls=10**3)
        elif mode == "test":
            model = AutoMorf(
                name=name, 
                framework=opt,
                n_jobs=NUM_CPUS,
                cv=2,
                n_calls=10)

    return model


def perform_cv(X, y, estimator, seed, tissue):
    from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
    from sklearn import metrics
    from collections import defaultdict

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
        "relevance": []
    }

    skf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=seed)
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
            y_train,
            y_train_hat,
            multioutput="raw_values")
        stats["evs_mo"]["train"].append(evs_mo_train)

        evs_mo_test = metrics.explained_variance_score(
            y_test,
            y_test_hat,
            multioutput="raw_values"
        )
        stats["evs_mo"]["test"].append(evs_mo_test)

        evs_ua_train = metrics.explained_variance_score(
            y_train,
            y_train_hat,
            multioutput="uniform_average"
        )
        stats["evs_ua"]["train"].append(evs_ua_train)

        evs_ua_test = metrics.explained_variance_score(
            y_test,
            y_test_hat,
            multioutput="uniform_average"
        )
        stats["evs_ua"]["test"].append(evs_ua_test)

        # MAE

        mae_mo_train = metrics.mean_absolute_error(
            y_train,
            y_train_hat,
            multioutput="raw_values")
        stats["mae_mo"]["train"].append(mae_mo_train)

        mae_mo_test = metrics.mean_absolute_error(
            y_test,
            y_test_hat,
            multioutput="raw_values"
        )
        stats["mae_mo"]["test"].append(mae_mo_test)

        mae_ua_train = metrics.mean_absolute_error(
            y_train,
            y_train_hat,
            multioutput="uniform_average"
        )
        stats["mae_ua"]["train"].append(mae_ua_train)

        mae_ua_test = metrics.mean_absolute_error(
            y_test,
            y_test_hat,
            multioutput="uniform_average"
        )
        stats["mae_ua"]["test"].append(mae_ua_test)

        # MSE

        mse_mo_train = metrics.mean_squared_error(
            y_train,
            y_train_hat,
            multioutput="raw_values")
        stats["mse_mo"]["train"].append(mse_mo_train)

        mse_mo_test = metrics.mean_squared_error(
            y_test,
            y_test_hat,
            multioutput="raw_values"
        )
        stats["mse_mo"]["test"].append(mse_mo_test)

        mse_ua_train = metrics.mean_squared_error(
            y_train,
            y_train_hat,
            multioutput="uniform_average"
        )
        stats["mse_ua"]["train"].append(mse_ua_train)

        mse_ua_test = metrics.mean_squared_error(
            y_test,
            y_test_hat,
            multioutput="uniform_average"
        )
        stats["mse_ua"]["test"].append(mse_ua_test)

        # MSLE

        msle_mo_train = metrics.mean_squared_log_error(
            y_train,
            y_train_hat,
            multioutput="raw_values")
        stats["msle_mo"]["train"].append(msle_mo_train)

        msle_mo_test = metrics.mean_squared_log_error(
            y_test,
            y_test_hat,
            multioutput="raw_values"
        )
        stats["msle_mo"]["test"].append(msle_mo_test)

        msle_ua_train = metrics.mean_squared_log_error(
            y_train,
            y_train_hat,
            multioutput="uniform_average"
        )
        stats["msle_ua"]["train"].append(msle_ua_train)

        msle_ua_test = metrics.mean_squared_log_error(
            y_test,
            y_test_hat,
            multioutput="uniform_average"
        )
        stats["msle_ua"]["test"].append(msle_ua_test)

        # r2

        r2_mo_train = metrics.r2_score(
            y_train,
            y_train_hat,
            multioutput="raw_values")
        stats["r2_mo"]["train"].append(r2_mo_train)

        r2_mo_test = metrics.r2_score(
            y_test,
            y_test_hat,
            multioutput="raw_values"
        )
        stats["r2_mo"]["test"].append(r2_mo_test)

        r2_ua_train = metrics.r2_score(
            y_train,
            y_train_hat,
            multioutput="uniform_average"
        )
        stats["r2_ua"]["train"].append(r2_ua_train)

        r2_ua_test = metrics.r2_score(
            y_test,
            y_test_hat,
            multioutput="uniform_average"
        )
        stats["r2_ua"]["test"].append(r2_ua_test)

        if hasattr(estimator, "feature_importances_"):
            stats["relevance"].append(estimator.feature_importances_)
        if hasattr(estimator, "coef_"):
            stats["relevance"].append(estimator.coef_)

    return stats


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    hord()
