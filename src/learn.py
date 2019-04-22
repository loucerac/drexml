import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

from lightgbm.sklearn import LightGBMError
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array, check_random_state
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import FLOAT_DTYPES, check_X_y, check_is_fitted
from xgboost.core import XGBoostError

from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt import dump, load

from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

from gplearn.genetic import SymbolicTransformer
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.metrics import geometric_mean_score

from hpsklearn import HyperoptEstimator, random_forest_regression
from hyperopt import tpe


def plot_feature_importances(clf, X_train, y_train=None,
                             top_n=10, figsize=(8,8), print_table=False,
                             title="Feature Importances"):
    '''
    plot feature importances of a tree-based sklearn estimator

    Note: X_train and y_train are pandas DataFrames

    Note: Scikit-plot is a lovely package but I sometimes have issues
              1. flexibility/extendibility
              2. complicated models/datasets
          But for many situations Scikit-plot is the way to go
          see https://scikit-plot.readthedocs.io/en/latest/Quickstart.html

    Parameters
    ----------
        clf         (sklearn estimator) if not fitted, this routine will fit it

        X_train     (pandas DataFrame)

        y_train     (pandas DataFrame)  optional
                                        required only if clf has not already been fitted

        top_n       (int)               Plot the top_n most-important features
                                        Default: 10

        figsize     ((int,int))         The physical size of the plot
                                        Default: (8,8)

        print_table (boolean)           If True, print out the table of feature importances
                                        Default: False

    Returns
    -------
        the pandas dataframe with the features and their importance
    '''

    __name__ = "plot_feature_importances"

    try:
        if not hasattr(clf, 'feature_importances_'):
            clf.fit(X_train.values, y_train.values.ravel())

            if not hasattr(clf, 'feature_importances_'):
                raise AttributeError("{} does not have feature_importances_ attribute".
                                    format(clf.__class__.__name__))

    except (XGBoostError, LightGBMError, ValueError):
        clf.fit(X_train.values, y_train.values.ravel())

    feat_imp = pd.DataFrame({'importance':clf.feature_importances_})
    feat_imp['feature'] = X_train.columns
    feat_imp.sort_values(by='importance', ascending=False, inplace=True)
    feat_imp = feat_imp.iloc[:top_n]

    feat_imp.sort_values(by='importance', inplace=True)
    feat_imp = feat_imp.set_index('feature', drop=True)
    feat_imp.plot.barh(title=title, figsize=figsize)
    plt.xlabel('Feature Importance Score')
    plt.show()

    if print_table:
        from IPython.display import display
        print("Top {} features in descending order of importance".format(top_n))
        display(feat_imp.sort_values(by='importance', ascending=False))

    return feat_imp

class BoMorf(BaseEstimator, RegressorMixin):

    def __init__(self, name, framework="hyperopt", n_jobs=1, cv=10, n_calls=100, out=None, copy_X_train=True, random_state=42):
        self.name = name
        self.framework = framework
        self.n_jobs = n_jobs
        self.copy_X_train=copy_X_train,
        self.random_state=random_state
        self.cv = cv
        self.n_calls = n_calls
        self.out = self.get_output_folder(out)
        self.opt = None
        self.best_model = None

    def fit(self, X, y=None):
        # validate X, y
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)

        self.fit_(X, y)

    def fit_(self, X, y):
        if self.framework == "sko":
            self.fit_sko(X, y)
        elif self.framework == "hyperopt":
            self.fit_hyperopt(X, y)

        self.best_model.fit(X, y)
        self.best_model.copy_X_train = self.copy_X_train
        self.best_model.cv = self.cv

        self.best_model.X_train_ = np.copy(X) if self.copy_X_train else X
        self.best_model.y_train_ = np.copy(y) if self.copy_X_train else y

        self.X_train_ = self.best_model.X_train_
        self.y_train_ = self.best_model.X_train_

    def fit_hyperopt(self, X, y):

        estimator = random_forest_regression(
            self.name,
            n_jobs=self.n_jobs,
            random_state=self.random_state)

        self.opt = HyperoptEstimator(
            regressor = estimator,
            algo=tpe.suggest,
            max_evals=self.n_calls,
            trial_timeout=None,
            seed=self.random_state)

        self.opt.fit(X, y)

        self.opt.cv = self.cv
        self.opt.n_calls = self.n_calls

        self.best_model = self.opt.best_model()["learner"]

    def fit_sko(self, X, y):
        from skopt.callbacks import VerboseCallback

        estimator, space, objective = self.get_optimizer(
            X,
            y,
            self.n_jobs,
            self.cv,
            self.random_state
        )

        self.opt =  gp_minimize(
            objective,
            space,
            acq_optimizer="lbfgs",
            n_jobs=self.n_jobs,
            n_calls=self.n_calls,
            random_state=self.random_state,
            callback=VerboseCallback(n_total=1)
            )

        self.opt.cv = self.cv

        self.best_model = self.build_model_from_sko(self.opt)

    def predict(self, X):
        return self.best_model.predict(X)

    def score(self, X, y):
        return self.best_model.score(X, y)

    @property
    def feature_importances_(self):
        """Return the feature importances (the higher, the more important the
           feature).
        Returns
        -------
        feature_importances_ : array, shape = [n_features]
        """

        check_is_fitted(self.best_model, 'estimators_')

        return self.best_model.feature_importances_

    def save(self, out):
        out = self.get_output_folder(out)
        opt_path = out.joinpath(self.get_opt_fname(self.name))
        if self.framework == "sko":
            estimator, space, objective = self.get_optimizer(
                self.X_train_,
                self.y_train_,
                self.n_jobs,
                self.cv,
                self.random_state
            )
            joblib.dump(self.opt, opt_path)
        elif self.framework == "hyperopt":
            joblib.dump(self.opt, opt_path)
        estimator_path = out.joinpath(self.get_estimator_fname(self.name))
        joblib.dump(self.best_model, estimator_path)

    @classmethod
    def load(cls, out, name):
        out = cls.get_output_folder(out)
        # load ML model
        estimator_path = out.joinpath(cls.get_estimator_fname(name))
        with open(estimator_path, "rb") as f:
            estimator = joblib.load(f)

        # load optimization result
        opt_path = out.joinpath(cls.get_opt_fname(name))

        if "sko" in name:
            print(
                estimator.X_train_.shape, estimator.y_train_.shape,
                estimator.n_jobs, estimator.cv, estimator.random_state
            )
            estimator, space, objective = cls.get_optimizer(
                estimator.X_train_,
                estimator.y_train_,
                estimator.n_jobs,
                estimator.cv,
                estimator.random_state
            )

            print(estimator, space, objective)

            opt = load(opt_path)
        elif "hyperopt" in name:
            with open(opt_path, "rb") as f:
                opt = joblib.load(f)

        bomorf = cls(
            name,
            n_jobs=estimator.n_jobs,
            cv=opt.cv,
            n_calls=opt.n_calls,
            out=out,
            copy_X_train=estimator.copy_X_train,
            random_state=estimator.random_state)
        bomorf.best_model = estimator
        bomorf.opt = opt

        return bomorf

    @staticmethod
    def build_model_from_sko(opt, n_jobs=1, random_state=42):
        hyperparameters  = {
            'max_depth':opt.x[0],
            'max_features':opt.x[1],
            'min_samples_split':opt.x[2],
            'min_samples_leaf':opt.x[3],
            'n_estimators':opt.x[4]
        }

        model = RandomForestRegressor(
            n_jobs=n_jobs,
            random_state=random_state,
            **hyperparameters
        )

        return model

    @staticmethod
    def get_opt_fname(name):
        return "{}_opt.pkl".format(name)

    @staticmethod
    def get_estimator_fname(name):
        return "{}_estimator.pkl".format(name)

    @staticmethod
    def get_output_folder(out):
        if out:
            out = Path(out)
        return out

    @staticmethod
    def get_optimizer(X, y, n_jobs, cv, random_state):
        estimator = RandomForestRegressor(
            n_jobs=n_jobs,
            random_state=random_state
            )

        # n_features = y.shape[1]
        space = [
            Integer(1, 10, name='max_depth'),
            Real(0.1, 1.0, name="max_features"),
            Integer(2, 100, name='min_samples_split'),
            Integer(1, 100, name='min_samples_leaf'),
            Categorical([10**2, 500, 10**3, 5000, 10**4], name="n_estimators")]

        @use_named_args(space)
        def objective(**params):
            estimator.set_params(**params)

            cv_scores = cross_val_score(
                estimator,
                X,
                y,
                cv=cv,
                n_jobs=n_jobs,
                scoring="r2"
            )

            return -np.mean(cv_scores)

        return estimator, space, objective
