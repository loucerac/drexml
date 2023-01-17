# -*- coding: utf-8 -*-
"""
Model definition.
"""

import copy

import numpy as np
from sklearn.ensemble import RandomForestRegressor


class AutoMORF(RandomForestRegressor):
    def __init__(
        self,
        n_estimators_min=50,
        n_estimators_max=1000,
        tol=1e-3,
        patience=50,
        n_estimators=100,
        *,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
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
        super().__init__(
            n_estimators=100,
            criterion="squared_error",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=1.0,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=None,
            random_state=None,
            verbose=0,
            warm_start=False,
            ccp_alpha=0.0,
            max_samples=None,
        )

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

        if X_val.shape[1] > 1e6:
            estimators = copy.deepcopy(self.estimators_)
            error_rate = [0]
            diffs = [0]
            n_ok = 0
            for i in range(self.n_estimators_min, self.n_estimators_max):
                self.n_estimators = i
                self.estimators_ = copy.deepcopy(estimators[0:i])

                # basic early stopping after `patience` iterations under tol
                error_rate.append(1 - self.score(X_val, y_val))
                diffs.append(np.abs(error_rate[-1] - error_rate[-2]))
                if diffs[-1] < self.tol:
                    n_ok += 1
                else:
                    n_ok = 0
                if n_ok > self.patience:
                    break

        # self.n_estimators = len(self.estimators_)
        self.warm_start = False
        print(len(self.estimators_), self.n_estimators)
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
    this_seed = 275
    mtry = int(np.sqrt(n_features) + 20)
    if debug:
        n_estimators = 2
        n_estimators_min = 10
        n_estimators_max = 20
        patience = 5
    else:
        n_estimators = max(201, int((n_features + n_targets) * 201 / 700))
        # n_estimators = int(1.5 * (n_features + n_targets))
        # n_estimators = np.log2()
        n_estimators_min = 100
        n_estimators_max = n_estimators
        patience = 100

    model = AutoMORF(
        n_jobs=n_jobs,
        n_estimators=n_estimators,
        n_estimators_min=n_estimators_min,
        n_estimators_max=n_estimators_max,
        patience=patience,
        max_depth=8,
        max_features=mtry,
    )

    model = RandomForestRegressor(
        n_jobs=n_jobs,
        n_estimators=200,
        max_depth=8,
        max_features=mtry,
        random_state=this_seed,
    )

    print(n_estimators)

    return model
