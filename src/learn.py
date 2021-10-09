#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Carlos Loucera <carlos.loucera@juntadeandalucia.es>
Author: Maria Pena Chilet <maria.pena.chilet.ext@juntadeandalucia.es>
Author: Marina Esteban <marina.estebanm@gmail.com>

Learning module for HORD multi-task framework.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hpsklearn import HyperoptEstimator, random_forest_regression
from hyperopt import tpe
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_is_fitted, check_X_y
import json


def plot_feature_importances(
    clf,
    X_train,
    y_train=None,
    top_n=10,
    figsize=(8, 8),
    print_table=False,
    title="Feature Importances",
):
    """Plot feature importances of a tree-based sklearn estimator.

    Parameters
    ----------
    clf : scikit-learn estimator
        ML model, if not fitted the method tries to fit the model to the given
        data.
    X_train : DataFrame
        Train features.
    y_train : array like, shape [n_saples,] or [n_samples, n_tasks]
        Training targets, by default None
    top_n : int, optional
        Maximum number of features to plot (in descending order of importance),
        by default 10
    figsize : tuple, optional
        Figure size, by default (8, 8)
    print_table : bool, optional
        Print relevance Data Frame, by default False
    title : str, optional
        Figure title, by default "Feature Importances"

    Returns
    -------
    DataFrame, shape [n_features, 1]
        Relevance DataFrame.

    Raises
    ------
    AttributeError
        An error is raised if the estimator lacks a feature_importances_
        attribute.
    """

    try:
        if not hasattr(clf, "feature_importances_"):
            clf.fit(X_train.values, y_train.values.ravel())

            if not hasattr(clf, "feature_importances_"):
                raise AttributeError(
                    "{} does not have feature_importances_ attribute".format(
                        clf.__class__.__name__
                    )
                )

    except ValueError:
        clf.fit(X_train.values, y_train.values.ravel())

    feat_imp = pd.DataFrame({"importance": clf.feature_importances_})
    feat_imp["feature"] = X_train.columns
    feat_imp.sort_values(by="importance", ascending=False, inplace=True)
    feat_imp = feat_imp.iloc[:top_n]

    feat_imp.sort_values(by="importance", inplace=True)
    feat_imp = feat_imp.set_index("feature", drop=True)
    feat_imp.plot.barh(title=title, figsize=figsize)
    plt.xlabel("Feature Importance Score")
    plt.show()

    if print_table:
        from IPython.display import display

        print("Top {} features in descending order of importance".format(top_n))
        display(feat_imp.sort_values(by="importance", ascending=False))

    return feat_imp


class AutoMorf(BaseEstimator, RegressorMixin):
    """Automated Multi Output Random Forest. Multi task learning class via
    Random Forest with only one set of hyperparameters which are automagically
    tunned with either Tree-structured Parzen Estimator (TPE) or Bayesian
    Optimization (BO).
    """

    def __init__(
        self,
        name,
        framework="hyperopt",
        n_jobs=1,
        cv=10,
        n_calls=100,
        copy_X_train=True,
        random_state=42,
    ):
        self.name = name
        self.framework = framework
        self.n_jobs = n_jobs
        self.copy_X_train = (copy_X_train,)
        self.random_state = random_state
        self.cv = cv
        self.n_calls = n_calls
        self.opt = None
        self.best_model = None

    def fit(self, X, y=None):
        """Fit estimator.

        A suitable set of hyperparameters is found via either Tree-structured
        Parzen Estimator (TPE) or Bayesian Optimization (BO).

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            The input samples. Use ``dtype=np.float32`` for maximum
            efficiency. Sparse matrices are also supported, use sparse
            ``csc_matrix`` for maximum efficiency.
        y : array-like, [n_samples, n_outputs]
            The target (continuous) values for regression.

        Returns
        -------
        self : object
        """

        # validate X, y
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)

        self.fit_(X, y)

    def fit_(self, X, y):
        """Select the optimization method basen on the `framework` arguments and
        fit the model to data.

        """
        if self.framework == "hyperopt":
            self.fit_hyperopt(X, y)

        self.best_model.fit(X, y)
        self.best_model.copy_X_train = self.copy_X_train
        self.best_model.cv = self.cv

        self.best_model.X_train_ = np.copy(X) if self.copy_X_train else X
        self.best_model.y_train_ = np.copy(y) if self.copy_X_train else y

        self.X_train_ = self.best_model.X_train_
        self.y_train_ = self.best_model.X_train_

    def fit_hyperopt(self, X, y):
        """Tree-structured Parzen Estimator (TPE)-based hyperparameter search."""
        estimator = random_forest_regression(
            self.name, n_jobs=self.n_jobs, random_state=self.random_state
        )

        self.opt = HyperoptEstimator(
            regressor=estimator,
            algo=tpe.suggest,
            max_evals=self.n_calls,
            trial_timeout=None,
            seed=self.random_state,
        )

        self.opt.fit(X, y)

        self.opt.cv = self.cv
        self.opt.n_calls = self.n_calls

        self.best_model = self.opt.best_model()["learner"]

    def predict(self, X):
        """Predict multiple regression targets for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        y : [n_samples, n_outputs]
            The predicted values.
        """
        return self.best_model.predict(X)

    def score(self, X, y):
        """Returns the coefficient of determination R^2 of the prediction.
        The coefficient R^2 is defined as (1 - u/v), where u is the residual
        sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
        sum of squares ((y_true - y_true.mean()) ** 2).sum().
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always
        predicts the expected value of y, disregarding the input features,
        would get a R^2 score of 0.0.
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples. For some estimators this may be a
            precomputed kernel matrix instead, shape = (n_samples,
            n_samples_fitted], where n_samples_fitted is the number of
            samples used in the fitting for the estimator.
        y : (n_samples, n_outputs)
            True values for X.

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """
        return self.best_model.score(X, y)

    @property
    def feature_importances_(self):
        """Return the feature importances (the higher, the more important the
           feature).
        Returns
        -------
        feature_importances_ : array, shape = [n_features]
        """

        check_is_fitted(self.best_model, "estimators_")

        return self.best_model.feature_importances_

    def save(self, out, fmt="json"):
        """Save the model to specified folder.

        Due to serialization limitations the saved model is split into
        two separte files: one file is used for the best estimator while the
        other file is used for the (resumable) optimization history.

        Parameters
        ----------
        out : str, pathlib.Path, or file object.
            The path where the model must be stored in '.gz' format.
        """
        out = self.get_output_folder(out)
        if fmt == "json":
            json_fname = "model_hyperparameters.json"
            json_fpath = out.joinpath(json_fname)
            model_hp = self.best_model.get_params()
            with open(json_fpath, "w") as fjson:
                json.dump(model_hp, fjson, indent=4)
        else:
            opt_path = out.joinpath(self.get_opt_fname(self.name))
            if self.framework == "hyperopt":
                joblib.dump(self.opt, opt_path)
            estimator_path = out.joinpath(self.get_estimator_fname(self.name))
            joblib.dump(self.best_model, estimator_path)

    @classmethod
    def load(cls, out, name):
        """Alternative AutoMorf cosntruction method. Load an stored AutoMorf
        model.

        Parameters
        ----------
        out : str, pathlib.Path, or file object.
            The path where the model is stored.
        name : str
            Model name.

        Returns
        -------
        AutoMorf instance.
            A fitted AutoMorf model.
        """
        out = cls.get_output_folder(out)
        # load ML model
        estimator_path = out.joinpath(cls.get_estimator_fname(name))
        with open(estimator_path, "rb") as f:
            estimator = joblib.load(f)

        # load optimization result
        opt_path = out.joinpath(cls.get_opt_fname(name))

        if "hyperopt" in name:
            with open(opt_path, "rb") as f:
                opt = joblib.load(f)

        bomorf = AutoMorf(
            name,
            n_jobs=estimator.n_jobs,
            cv=opt.cv,
            n_calls=opt.n_calls,
            copy_X_train=estimator.copy_X_train,
            random_state=estimator.random_state,
        )
        bomorf.best_model = estimator
        bomorf.opt = opt

        return bomorf

    @staticmethod
    def build_model_from_sko(opt, n_jobs=1, random_state=42):
        """Parse a scikit-optimization history file to construct the
        scikit-learn estimator.
        """
        hyperparameters = {
            "max_depth": opt.x[0],
            "max_features": opt.x[1],
            "min_samples_split": opt.x[2],
            "min_samples_leaf": opt.x[3],
            "n_estimators": opt.x[4],
        }

        model = RandomForestRegressor(
            n_jobs=n_jobs, random_state=random_state, **hyperparameters
        )

        return model

    @staticmethod
    def get_opt_fname(name):
        """Construct the optimization history file name."""
        return "{}_opt.pkl".format(name)

    @staticmethod
    def get_estimator_fname(name):
        """Construct the estimator file name."""
        return "{}_estimator.pkl".format(name)

    @staticmethod
    def get_output_folder(out):
        """Pathlib check and conversion of output folder."""
        if out:
            out = Path(out)
        return out
