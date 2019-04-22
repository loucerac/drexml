#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Carlos Loucera <carlos.loucera@juntadeandalucia.es>
Author: Maria Pena Chilet <maria.pena.chilet.ext@juntadeandalucia.es>
Author: Marina Esteban <marina.estebanm@gmail.com>

Entry CLI point for HORD multi-task framework.
"""

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os
from pathlib import Path
from datetime import datetime
import click
import pickle
import traceback
from src.datasets import get_disease_data, load_clinical_data
from src.learn import BoMorf
import numpy as np
import sklearn
from sklearn.metrics import average_precision_score
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split

from dotenv import find_dotenv, load_dotenv


dotenv_filepath = find_dotenv()
load_dotenv(dotenv_filepath)
project_path = os.path.dirname(dotenv_filepath)

DATA_PATH = Path(os.environ.get("DATA_PATH"))
NUM_CPUS = int(os.getenv("NUM_CPUS"))
USE_GPU = bool(os.getenv("USE_GPU"))
warnings.filterwarnings('ignore', category=DeprecationWarning, module='sklearn')

@click.command()
@click.option('--disease', default="fanconi", help='which disease to test')
@click.option('--model', default="morf", help='ML model')
@click.option('--opt', deafult="hyperopt", help='Train/test mode')
@click.option('--seed', deafult=42, type=int, help='Random seed')
@click.option("--train", is_flag=True, help="Train and evaluate or evaluate")
def hord(disease, model, opt, seed, train):
    """HORD multi-task module.
    
    Parameters
    ----------
    disease : str
        Disease to train/test.
    model : str
        Which ML model to use.
    opt : str
        Select optimization mode ["sko", "hpo", None]
    seed : int
        Seed for random number generator.
    train : bool
        Train or load a pre-trained model.
    """

    print("Working on disease {}".format(disease))

    run_(disease, model, opt, seed, train)    
 
    exit(0)

def get_out_path(disease, model, opt, seed):

    out_path = DATA_PATH.joinpath("out", disease, model, opt, str(seed))
    out_path.mkdir(parents=True, exist_ok=False)

    return out_path

def run_(disease, model, opt, seed, train):
    if train:
        run_full(disease, model, opt, seed)


def run_full(disease, model, opt, seed):
    from sklearn.model_selection import RepeatedStratifiedKFold

    output_folder = get_out_path(disease, model, opt, seed)
    
    # Load data
    gene_xpr, pathvals, circuits, genes, clinical = get_disease_data(disease)

    # Optimize and fit using the whole data
    model.fit(gene_xpr, pathvals)
    model.save(output_folder)

    # CV with optimal hyperparameters
    estimator = model.best_model
    cv_stats = perform_cv(gene_xpr, pathvals, estimator, seed, clinical.tissue)

    # Save results
    stats_fname = "cv_stats"
    stats_fpath = os.path.join(output_folder, stats_fname)
    with open(stats_fpath, "wb") as f:
        joblib.dump(cv_stats, f)

    

def perform_cv(X, y, estimator, seed, tissue):
    from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
    from sklearn import metrics
    from collections import defaultdict
    
    stats = {
        "evs_mo":{"train": [], "test": []},
        "evs_ua":{"train": [], "test": []},
        "mae_mo":{"train": [], "test": []},
        "mae_ua":{"train": [], "test": []},
        "mse_mo":{"train": [], "test": []},
        "mse_ua":{"train": [], "test": []},
        "msle_mo":{"train": [], "test": []},
        "msle_ua":{"train": [], "test": []},
        "r2_mo":{"train": [], "test": []},
        "r2_ua":{"train": [], "test": []},
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
