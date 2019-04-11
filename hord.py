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
import click
import pickle
import traceback
from src.datasets import get_disease_data
from src.learn import TabularClassifier
from src.plot import plot_model
from src.autoxgb  import OptimizedXGB
from src.utils import get_out_path, get_tissue_cell_line
import numpy as np
import sklearn
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from timeit import default_timer as timer

from dotenv import find_dotenv, load_dotenv


dotenv_filepath = find_dotenv()
load_dotenv(dotenv_filepath)
project_path = os.path.dirname(dotenv_filepath)

DATA_PATH = os.getenv("DATA_PATH")
NUM_CPUS = int(os.getenv("NUM_CPUS"))
USE_GPU = bool(os.getenv("USE_GPU"))
warnings.filterwarnings('ignore', category=DeprecationWarning, module='sklearn')

@click.command()
@click.option('--disease', default="fanconi", help='which disease to test')
@click.option('--model', default="morf", help='ML model')
@click.option('--train', is_flag=True, help='train')
def hord(disease, model, train):
    """HORD multi-task module.
    
    Parameters
    ----------
    disease : str
        Disease to train/test.
    model : str
        Which ML model to use.
    train : bool
        Flag (Defaults to False. If true, train a model).
    """

    gene_expression, path_vals, metadata = get_disease_data(disease)

    print("Working on disease {}".format(disease))

    X_train, X_test, y_train, y_test = load_data(which, mode)

    if train:
        model, name = get_model(name)

        get_out_path(which, mode, name)
        
        print("Training {} for {}".format(name, which))
        start = timer()
        model.fit(X_train.values, y_train.values.ravel())
        end = timer()
        print("training finished.")
        print("{} mode {} training time {}".format(which, mode, end - start))

        print("Saving model...")
        save_model(model, which, mode, name)
        print("model saved.")
    else:
        print("Loading model...")
        model = load_model(name, which, mode)
        print("model loaded.")

    # set plotting context and plot
    print("Saving analysis...")
    plt.style.use("ggplot")
    sns.set_context("paper")
    plot_model(model, X_train, y_train, mode, which, "train")
    plot_model(model, X_test, y_test, mode, which, "test")
    print("analysis saved.")
    
    exit(0)


def load_data(which, mode):
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

    print("Loading {} data for {} mode".format(which, mode))

    if mode in ["versus", "vs"]:
        try:
            X_train, X_test, y_train, y_test = load_versus_dataset(which)
        except:
            print("No available data for {} which".format(which))
            exit("Abort ...")
    elif mode.lower() == "loco":
        try:
            tissue = "_".join(which.split("_")[1:])
            print(tissue)
            X_train, X_test, y_train, y_test = build_loco_dataset_single(
                tissue,
                which
            )
        except Exception as e:
            print("No available data for {} which".format(which))
            print(traceback.format_exc(e))
            exit("Abort ...")
    else:
        raise NotImplementedError("{} not yet implemented".format(mode))

    print("Train shape: ", X_train.shape, "Test shape: ", X_test.shape)
    print("Train shape: ", y_train.shape, "Test shape: ", y_test.shape)
      
    if not X_test.shape[0]:
        raise IOError("No testing data for {}".format(which))

    if not X_train.shape[0]:
        raise IOError("No training data for {}".format(which))

    return X_train, X_test, y_train, y_test


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

    if name.lower() == "lgbm":
        model = TabularClassifier(verbose=1)
    elif name.lower() in ["xgb", "xgboost"]:
        model = OptimizedXGB()
    elif name.lower() in ["bag", "bbag", "balanced"]:
        model = BalancedBaggingClassifier(
            n_estimators=NUM_CPUS,
            n_jobs=NUM_CPUS
        )

        name = "bbag"
    
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


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    achilles()