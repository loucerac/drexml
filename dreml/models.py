# -*- coding: utf-8 -*-
"""
Model definition.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor


class AutoMORF(RandomForestRegressor):
    def __init__(
        self,
        n_estimators_min=50,
        n_estimators_max=1000,
        tol=1e-3,
        patience=50,
        **kwargs
    ):
        """Extension of RandomForestRegressor that automatically selects the number of
        trees using an early stopping criterion and warm starting. Note that we want to
        minimize the model memory fingerprint when using SHAP, GPUs and big datasets.

        Parameters
        ----------
        mtry_min : int, optional
            _description_, by default 2
        mtry_max : int, optional
            _description_, by default 100
        tol : _type_, optional
            _description_, by default 1e-3
        patience : int, optional
            _description_, by default 50
        """
        super().__init__(**kwargs)
        self.n_estimators_min = n_estimators_min
        self.n_estimators_max = n_estimators_max
        self.tol = tol
        self.patience = patience

    def fit(self, X, y, sample_weight=None, X_val=None, y_val=None):
        """_summary_

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        y : array-like of shape (n_samples,)
            Target values (strings or integers in classification, real numbers
            in regression)
            For classification, labels must correspond to classes.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.ptional
            _description_, by default None
        X_val : array-like of shape (n_samples, n_features), optional
            feature validation matrix, by default None
        y_val : array-like of shape (n_samples, n_targets), optional
            Target validation matrix, by default None

        Returns
        -------
        self : object
            Fitted estimator.
        """
        super().fit(X, y, sample_weight)

        if X_val is not None:
            estimators = self.estimators_
            error_rate = [0]
            diffs = [0]
            n_ok = 0
            for i in range(self.n_estimators_min, self.n_estimators_max):
                self.n_estimators = i
                self.estimators_ = estimators[0:i]

                # basic early stopping after `patience` iterations under tol
                error_rate.append(1 - self.score(X_val, y_val))
                diffs.append(np.abs(error_rate[-1] - error_rate[-2]))
                if diffs[-1] < self.tol:
                    n_ok += 1
                else:
                    n_ok = 0
                if n_ok > self.patience:
                    break

        self.warm_start = False
        return self


def get_model(n_features, n_targets, n_jobs, debug, n_iters=None):
    """Create a model.

    Parameters
    ----------
    n_features : int
        Number of features (KDTs).
    n_targets : int
        Number of targets (circuits).
    n_jobs : int
        The number of jobs to run in parallel.
    debug : bool
        Debug flag.
    n_iters : int, optional
        Number of ietrations for hyperparatemer optimization, by default None

    Returns
    -------
    sklearn.ensemble.RandomForestRegressor
        The model to be fitted.
    """
    mtry = int(np.sqrt(n_features) + 20)
    if debug:
        n_estimators = 20
        n_estimators_min = 10
        n_estimators_max = 20
        patience = 5
    else:
        n_estimators = int(1.5 * (n_features + n_targets))
        n_estimators_min = 50
        n_estimators_max = n_estimators
        patience = 50

    model = AutoMORF(
        n_jobs=n_jobs,
        n_estimators=n_estimators,
        n_estimators_min=n_estimators_min,
        n_estimators_max=n_estimators_max,
        patience=patience,
        max_depth=8,
        max_features=mtry,
    )

    return model
