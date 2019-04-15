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
from src.datasets import get_disease_data
from src.learn import BoMorf
import numpy as np
import sklearn
from sklearn.metrics import average_precision_score
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
@click.option('--mode', deafult="split", help='Train/test mode')
@click.option('--seed', deafult=42, type=int, help='Random seed')
def hord(disease, model, mode, seed):
    """HORD multi-task module.
    
    Parameters
    ----------
    disease : str
        Disease to train/test.
    model : str
        Which ML model to use.
    mode : str
        Select mode for train/test from ["split"]
    seed : int
        Seed for random number generator.
    """

    print("Working on disease {}".format(disease))

    X_train, X_test, Y_train, Y_test = get_mode_data(disease, mode, seed)
    output_folder = get_output_folder(disease, mode, model)

    model, name = get_model(model)
    print("Training {} for {} model with {} model".format(disease, mode, model))
    start = timer()
    model.fit(X_train.values, Y_train.values)
    end = timer()
    print("training finished.")
    print("Training time {}".format(end - start))

    print("Saving model...")
    save_model(model, disease, mode, name)
    print("model saved.")

    # set plotting context and plot
    print("Saving analysis...")
    plt.style.use("ggplot")
    sns.set_context("paper")
    plot_model(model, X_train, y_train, mode, which, "train")
    plot_model(model, X_test, y_test, mode, which, "test")
    print("analysis saved.")
    
    exit(0)


def get_mode_data(disease, mode, seed):
    """Helper function to load Achilles data based on a preset mode.
    
    Parameters
    ----------
    which : str
        Cell line which.
    mode : str
        Achilles testing mode (either Versus (vs) or LOCO)
    
    Raises
    ------
    NotImplementedError
        To be implemented.
    
    Returns
    -------
    List of pandas.Dataframe
        Train and test data.
    """

    print("Perform analysis data for {} mode".format(mode))

    try:
        expr, pathvals, path_metadata, gene_metadata = get_disease_data(disease)
    except Exception as e:
        print("No available data for {} disease".format(disease))
        print(traceback.format_exc(e))
        exit("Abort ...")

    if mode.lower() == "split":
        try:
            X_train, X_test, Y_train, Y_test = train_test_split(
                expr,
                pathvals,
                test_size=0.33,
                random_state=seed
            )
        except:
            print("No available data for {} mode".format(mode))
            exit("Abort ...")
    else:
        raise NotImplementedError("{} not yet implemented".format(mode))

    print("Train shape: ", X_train.shape, "Test shape: ", X_test.shape)
    print("Train shape: ", Y_train.shape, "Test shape: ", Y_test.shape)
      
    if not X_test.shape[0]:
        raise IOError("No testing data for {}".format(mode))

    if not X_train.shape[0]:
        raise IOError("No training data for {}".format(mode))

    return X_train, X_test, Y_train, Y_test


def get_output_folder(disease, mode, model):

    folder_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_path = DATA_PATH.joinpath("out", mode, model, folder_name)
    out_path.mkdir(parents=True, exist_ok=False)

    return out_path


def get_model(name: str):
    """Helper function to construct a ML model that folows the sklearn API.
    
    Parameters
    ----------
    name : str
        Machine Learning model name: lightgbm, xgb or bbag.   
    
    Returns
    -------
    sklearn.base.ClassifierMixin
        A trained model that follows the sklearn API.
    str
        Model name.
    """

    if name.lower() == "bomorf":
        model = BoMorf(
            n_estimators=NUM_CPUS,
            n_jobs=NUM_CPUS
        )
    
    model.name = name

    return model, name


def save_model(model, which, mode, name):
    """Helper function to save a sklearn-based ML model.
    
    Parameters
    ----------
    model : sklearn.base.ClassifierMixin
        A Machine Learning model that follows the sklearn API.
    which : str
        Cell line which.
    mode : str
        Achilles testing mode (either Versus (vs) or LOCO)
    name : str
        Machine Learning model name: lightgbm, xgb or bbag.    
    """

    out_path = get_out_path(which, mode, name)
    model_name = "achilles_{}_{}_{}.pkl".format(which, mode, name)
    model_path = os.path.join(out_path, model_name)
    with open(model_path, 'wb') as fout:
        if hasattr(model, "clf"):
            pickle.dump(model.clf, fout)
        else:
            pickle.dump(model, fout)


def load_model(which, mode, name):
    """Helper function to load a trained model.
    
    Parameters
    ----------
    which : str
        Cell line which.
    mode : str
        Achilles testing mode (either Versus (vs) or LOCO)
    name : str
        Machine Learning model name: lightgbm, xgb or bbag.

    Returns
    -------
    sklearn.base.ClassifierMixin
        A trained model that follows the sklearn API.
    """

    out_path = get_out_path(which, mode, name)
    model_name = "achilles_{}_{}_{}.pkl".format(which, mode, name)
    model_path = os.path.join(out_path, model_name)
    with open(model_path, 'rb') as fin:
        try:
            model = pickle.load(fin)
        except Exception as e:
            print("Unable to load trained model: {}".format(model_path))
            print(traceback.format_exc(e))

    return model


def run_full(disease, model, sample_metadata, seed):
    from sklearn.model_selection import RepeatedStratifiedKFold
    # Load whole data
    gene_xpr, pathvals, path_metadata, gene_metadata = get_disease_data(disease)

    # Optimize and fit and the whole data
    model.fit(gene_xpr, pathvals)

    # CV with optimal hyperparameters
    estimator = model.best_model
    perform_cv(X, y, estimator, seed, sample_metadata.tissue)

    # Save results

    pass

def perform_cv(X, y, estimator, seed, tissue):
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn import metrics

    skf = RepeatedStratifiedKFold(n_splits=10, n_repeats=100, random_state=seed)
    for train_index, test_index in skf.split(X, tissue):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        estimator.fit(X_train, y_train)

        y_train_hat = estimator.predict(X_train)
        y_test_hat = estimator.predict(X_test)

        # metrics computation
        evs_mo_train = metrics.explained_variance_score(
            y_train,
            y_train_hat,
            multioutput="raw_values")

        evs_mo_test = metrics.explained_variance_score(
            y_test,
            y_test_hat,
            multioutput="raw_values"
        )

        evs_ua_train = metrics.explained_variance_score(
            y_train,
            y_train_hat,
            multioutput="uniform_average"
        )

        evs_ua_test = metrics.explained_variance_score(
            y_test,
            y_test_hat,
            multioutput="uniform_average"
        )



if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    hord()