#!/usr/bin/env python
# coding: utf-8

# In[1]:


import joblib
import dotenv
import pathlib
import matplotlib.pyplot as plt
plt.style.use("ggplot")


import shap
import seaborn as sns


def matcorr(O, P):    
    (n, t) = O.shape      # n traces of t samples
    (n_bis, m) = P.shape  # n predictions for each of m candidates

    DO = O - (np.einsum("nt->t", O, optimize='optimal') / np.double(n)) # compute O - mean(O)
    DP = P - (np.einsum("nm->m", P, optimize='optimal') / np.double(n)) # compute P - mean(P)

    cov = np.einsum("nm,nt->mt", DP, DO, optimize='optimal')

    varP = np.einsum("nm,nm->m", DP, DP, optimize='optimal')
    varO = np.einsum("nt,nt->t", DO, DO, optimize='optimal')
    tmp = np.einsum("m,t->mt", varP, varO, optimize='optimal')

    return cov / np.sqrt(tmp)








from sklearn.model_selection import cross_validate, ShuffleSplit, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import MultiTaskLasso, RidgeCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels
import numpy as np


# In[11]:


import pandas as pd


# In[12]:


import sys
import time


# In[13]:




# In[14]:




# In[15]:


from joblib import Parallel, delayed
from scipy.stats import pearsonr
import joblib
import stability as stab


# In[16]:


from sklearn.metrics import r2_score


# In[17]:


from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.linear_model import MultiTaskLasso, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.cross_decomposition import PLSRegression


# In[18]:




# In[19]:


def compute_shap_fs(relevances, q=0.95, by_circuit=False):

    by_circuit_frame = relevances.abs().apply(lambda x: x > np.quantile(x, q), axis=1)

    if by_circuit:
        res = by_circuit_frame
    else:
        res = by_circuit_frame.any().values

    return res


# In[20]:



def compute_shap_values(estimator, background, new, approximate=True, check_additivity=False):
    explainer = shap.GPUTreeExplainer(estimator, background)
    shap_values = np.array(explainer.shap_values(new))    

    return shap_values


# In[21]:


def compute_shap_relevance(shap_values, X, Y):

    feature_names = X.columns
    task_names = Y.columns

    n_features = len(feature_names)
    n_tasks = len(task_names)
    
    c = lambda x, y: np.sign(np.diag(matcorr(x, y)))

    corr_sign = lambda x, y: np.sign(pearsonr(x, y)[0])
    signs = Parallel(n_jobs=-1)(
        delayed(c)(X.values, shap_values[y_col, :, :])
        for y_col in range(n_tasks)
    )

    signs = np.array(signs).reshape((n_tasks, n_features), order="F")
    signs = pd.DataFrame(signs, index=Y.columns, columns=X.columns)

    #shap_values = np.array(shap_values)

    shap_relevance = pd.DataFrame(
        np.abs(shap_values).mean(axis=(1)), index=task_names, columns=feature_names
    )

    shap_relevance = shap_relevance * signs
    shap_relevance = shap_relevance.fillna(0.0)

    return shap_relevance


# In[22]:



def build_stability_dict(z_mat, errors, alpha=0.05):

    support_matrix = np.squeeze(z_mat)
    scores = np.squeeze(1 - errors)

    stab_res = stab.confidenceIntervals(support_matrix, alpha=alpha)
    stability = stab_res["stability"]
    stability_error = stab_res["stability"] - stab_res["lower"]

    res = {
        "scores": scores.tolist(),
        "stability_score": stability,
        "stability_error": stability_error,
        "alpha": alpha,
    }

    return res


# In[23]:


from sklearn.base import clone

def run_stability(model, X, Y, n_bootstraps=100, alpha=0.05, approximate=True, check_additivity=False):
    n_samples, n_variables = X.shape
    sample_fraction = 0.5
    n_subsamples = np.floor(sample_fraction * n_samples).astype(int)

    q = 0.95

    # lambda: quantile selected

    Z = np.zeros((n_bootstraps, n_variables), dtype=np.int8)
    errors = np.zeros(n_bootstraps)

    stability_cv = ShuffleSplit(
        n_splits=n_bootstraps, train_size=n_subsamples, random_state=0
    )

    def stab_i(model, X, Y, n_split, split, q=0.95):
        print(n_split)
        train, test = split
        X_train = X.iloc[train, :]
        Y_train = Y.iloc[train, :]
        X_test = X.iloc[test, :]
        Y_test = Y.iloc[test, :]

        X_learn, X_val, Y_learn, Y_val = train_test_split(
            X_train, Y_train, test_size=0.3, random_state=n_split
        )

        model_ = clone(model)
        model_.set_params(**{"random_state": n_split})
        model_.fit(X_learn, Y_learn)

        # FS using shap relevances
        shap_values = compute_shap_values(
            model_,
            X_learn,
            X_val
        )
        shap_relevances = compute_shap_relevance(shap_values, X_val, Y_val)
        filt_i = compute_shap_fs(shap_relevances, q=q, by_circuit=False)

        X_train_filt = X_train.loc[:, filt_i]
        X_test_filt = X_test.loc[:, filt_i]

        sub_model = clone(model_)
        sub_model.set_params(**{"max_depth": 32})
        # sub_model.set_params(max_features=1.0)
        sub_model.fit(X_train_filt, Y_train)
        Y_test_filt_preds = sub_model.predict(X_test_filt)

        r2_loss = 1.0 - r2_score(Y_test, Y_test_filt_preds)
        mo_r2_loss = 1.0 - r2_score(
            Y_test, Y_test_filt_preds, multioutput="raw_values"
        )

        return (filt_i, r2_loss, mo_r2_loss)
    
    stab_values = []
    for n_split, split in enumerate(stability_cv.split(X, Y)):
        stab_values.append(stab_i(model, X, Y, n_split, split))

    for n_split, values in enumerate():
        Z[n_split, :] = values[0]
        errors[n_split] = values[1]

    res = build_stability_dict(Z, errors, alpha)
    print(res["stability_score"])

    return res


# In[24]:

X_learn = x.iloc[split[0], :]
Y_learn = y.iloc[split[0], :]

X_val = x.iloc[split[1], :]
Y_val = y.iloc[split[1], :]

shap_values = compute_shap_values(
    model,
    X_learn,
    X_val
)

shap_relevances = compute_shap_relevance(shap_values, X_val, Y_val)
filt_i = compute_shap_fs(shap_relevances, q=0.95, by_circuit=False)

# return gpu id to queue
q.put(gpu)

return filt_i