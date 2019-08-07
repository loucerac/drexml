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
from sklearn.externals import joblib
from sklearn.model_selection import RepeatedKFold

from src.datasets import get_disease_data
from src.learn import AutoMorf


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

warnings.filterwarnings(
    'ignore', category=DeprecationWarning, module='sklearn')


@click.command()
@click.option('--disease', default="fanconi", help='which disease to test')
@click.option('--mlmodel', default="morf", help='ML model')
@click.option('--opt', default="hyperopt", help='Train/test mode')
@click.option('--seed', default=42, type=int, help='Random seed')
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
        out_path = env_possible.parent.joinpath("ml", mlmodel)
    else:
        if not len(pathways):
            name = ["all"]
        else:
            name = pathways
        name = "_".join(name)

        out_path = OUT_PATH.joinpath(
            disease,
            name,
            gset,
            mlmodel,
            opt,
            mode,
            str(seed)
        )

    if mode == "train":
        ok = True
    elif mode == "test":
        ok = True

    out_path.mkdir(parents=True, exist_ok=ok)
    print("Storage folder: {}".format(out_path))

    return out_path


def run_(disease, mlmodel, opt, seed, mode, pathways, gset):
    """Select the training mode.
    """
    if mode in ["train", "test"]:
        run_full(disease, mlmodel, opt, seed, mode, pathways, gset)


def get_data(disease, mode, pathways, gset):
    """Load disease data and metadata.
    """
    gene_xpr, pathvals, circuits, genes, clinical = get_disease_data(
        disease,
        pathways,
        gset
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
        disease,
        mode,
        pathways,
        gset
    )

    # Get ML model
    model = get_model(mlmodel, opt, mode)

    # Optimize and fit using the whole data
    model.fit(gene_xpr, pathvals)
    model.save(output_folder)

    estimator = model.best_model

    # Save global relevances
    print("Computing global relevance.")
    relevance_fname = "model_global_relevance.tsv"
    rel_fpath = output_folder.joinpath(relevance_fname)
    rel = estimator.feature_importances_
    rel_df = pd.DataFrame({"relevance": rel}, index=gene_xpr.columns)
    rel_df.to_csv(rel_fpath, sep="\t")

    # Compute shap relevances
    print("Computing task relevances.")
    compute_shap_relevance(estimator, gene_xpr, pathvals, output_folder, True)

    # CV with optimal hyperparameters (unbiased performance)
    cv_stats = perform_cv(gene_xpr, pathvals, estimator, seed, clinical.tissue)

    # Save results
    stats_fname = "cv_stats.pkl"
    stats_fpath = output_folder.joinpath(output_folder, stats_fname)
    with open(stats_fpath, "wb") as f:
        joblib.dump(cv_stats, f)
    print("Unbiased CV stats saved to: {}".format(stats_fpath))


def compute_shap_relevance(estimator, gene_xpr, pathvals, output_folder, task):
    """Compute the model relevance with SHAP.

    Parameters
    ----------
    estimator : scikit-learn estimator
        A fitted estimator.
    gene_xpr : array-like, shape = (n_samples, n_features)
        Gene expression dataset.
    pathvals : array-like, shape = (n_samples, n_tasks)
        Pathvals dataset.
    output_folder : str, pathlib.Path, or file object.
            The path where the model must be stored in '.gz' format.
    task : bool
        If True compute the Per task relevance, if False comute SHAP global
        relevance.
    """
    if not task:
        # Compute global shap relevances
        explainer = shap.TreeExplainer(estimator)
    else:
        X_summary = shap.kmeans(gene_xpr, 50)
        # Decorate prediction function
        predict = lambda x: estimator.predict(x)
        explainer = shap.KernelExplainer(predict, X_summary.data)

    shap_values = explainer.shap_values(gene_xpr)
    if task:
        shap_values_fname = "shap_values_task.pkl"
    else:
        shap_values_fname = "shap_values_global.pkl"
    shap_values_fpath = output_folder.joinpath(shap_values_fname)
    with open(shap_values_fpath, "wb") as f:
        joblib.dump(shap_values, f)
    print("Shap values saved to: {}".format(shap_values_fpath))

    if task:
        # Per task relevance
        n_tasks = pathvals.shape[1]
        relevances = []
        for i_task in range(n_tasks):
            name = pathvals.columns[i_task]
            print("Computing {} Shap relevance.".format(name))
            relevances.append(np.abs(shap_values[i_task]).mean(0))

        relevance = pd.DataFrame(
            np.vstack(relevances).T,
            columns=pathvals.columns,
            index=gene_xpr.columns
        )

        shap_values_fname = "shap_values_task_relevance.tsv"
    else:
        # Global relevance
        global_shap_values = np.abs(shap_values).mean(0)
        relevance = pd.DataFrame(
            {"relevance": global_shap_values},
            index=gene_xpr.columns
        )
        shap_values_fname = "shap_values_global_relevance.tsv"

    shap_values_fpath = output_folder.joinpath(shap_values_fname)
    relevance.to_csv(shap_values_fpath, sep="\t")
    print("Shap relevances saved to: {}".format(shap_values_fpath))


def get_model(mlmodel, opt, mode):
    """Get an instace of an AutoMorf model.
    """
    name = "_".join([mlmodel, opt])
    if mlmodel == "morf":
        if mode == "train":
            model = AutoMorf(
                name=name,
                framework=opt,
                n_jobs=NUM_CPUS,
                cv=5,
                n_calls=10 ** 3)
        elif mode == "test":
            model = AutoMorf(
                name=name,
                framework=opt,
                n_jobs=NUM_CPUS,
                cv=2,
                n_calls=5)

    return model


def perform_cv(X, y, estimator, seed, tissue):
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
