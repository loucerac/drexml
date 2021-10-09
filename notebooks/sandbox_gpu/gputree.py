#!/usr/bin/env python
# coding: utf-8

# In[1]:


import multiprocessing
import os
import pathlib
from multiprocessing import Manager, Process

import dotenv
import joblib
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.metrics import r2_score

plt.style.use("ggplot")


import shap


def matcorr(O, P):
    (n, t) = O.shape  # n traces of t samples
    (n_bis, m) = P.shape  # n predictions for each of m candidates

    DO = O - (
        np.einsum("nt->t", O, optimize="optimal") / np.double(n)
    )  # compute O - mean(O)
    DP = P - (
        np.einsum("nm->m", P, optimize="optimal") / np.double(n)
    )  # compute P - mean(P)

    cov = np.einsum("nm,nt->mt", DP, DO, optimize="optimal")

    varP = np.einsum("nm,nm->m", DP, DP, optimize="optimal")
    varO = np.einsum("nt,nt->t", DO, DO, optimize="optimal")
    tmp = np.einsum("m,t->mt", varP, varO, optimize="optimal")

    return cov / np.sqrt(tmp)


import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit, train_test_split

import stability as stab

# In[11]:


# In[12]:


# In[13]:


# In[14]:


# In[15]:


# In[16]:


# In[17]:


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


def compute_shap_values(
    estimator, background, new, approximate=True, check_additivity=False
):
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
        delayed(c)(X.values, shap_values[y_col, :, :]) for y_col in range(n_tasks)
    )

    signs = np.array(signs).reshape((n_tasks, n_features), order="F")
    signs = pd.DataFrame(signs, index=Y.columns, columns=X.columns)

    # shap_values = np.array(shap_values)

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


import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor

# In[24]:


plt.style.use("ggplot")


# In[32]:


# In[34]:


# In[35]:


# In[39]:

# In[42]:


# In[44]:


import os

from joblib import Parallel, delayed

from sklearn.base import clone

def run_stability(model, X, Y, cv, fs, alpha=0.05):
    n_bootstraps = len(fs)
    n_samples, n_features = X.shape

    # lambda: quantile selected

    Z = np.zeros((n_bootstraps, n_features), dtype=np.int8)
    errors = np.zeros(n_bootstraps)

    def stab_i(model, X, Y, n_split, split):
        print(n_split)
        learn, val, test = split
        X_learn = X.iloc[learn, :]
        Y_learn = Y.iloc[learn, :]
        X_val = X.iloc[val, :]
        Y_val = Y.iloc[val, :]
        X_test = X.iloc[test, :]
        Y_test = Y.iloc[test, :]
        X_train = pd.concat((X_learn, X_val), axis=0)
        Y_train = pd.concat((Y_learn, Y_val), axis=0)

        filt_i = fs[n_split]

        X_train_filt = X_train.loc[:, filt_i]
        X_test_filt = X_test.loc[:, filt_i]

        sub_model = clone(model)
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
    for n_split, split in enumerate(cv):
        stab_values.append(stab_i(model, X, Y, n_split, split))

    for n_split, values in enumerate(stab_values):
        Z[n_split, :] = values[0] * 1
        errors[n_split] = values[1]

    res = build_stability_dict(Z, errors, alpha)
    print(res["stability_score"])

    return res


def runner(data_folder, gpu_id_list, i):
    cpu_name = multiprocessing.current_process().name
    cpu_id = int(cpu_name[cpu_name.find("-") + 1 :]) - 1
    gpu = gpu_id_list[cpu_id]
    # print(gpu0, gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(os.environ["CUDA_VISIBLE_DEVICES"])

    cv = joblib.load("cv.jbl")
    split = cv[i]

    X_fpath = data_folder.joinpath("features_covid.pkl")
    X = joblib.load(X_fpath)

    Xat_fpath = data_folder.joinpath("features.pkl")
    Xat = joblib.load(Xat_fpath)

    X = X[Xat.columns]

    Y_fpath = data_folder.joinpath("target_covid.pkl")
    Y = joblib.load(Y_fpath)

    X_learn = X.iloc[split[0], :]
    Y_learn = Y.iloc[split[0], :]

    X_val = X.iloc[split[1], :]
    Y_val = Y.iloc[split[1], :]

    model = joblib.load(f"model_{i}.jbl")

    shap_values = compute_shap_values(model, X_learn, X_val)

    shap_relevances = compute_shap_relevance(shap_values, X_val, Y_val)
    filt_i = compute_shap_fs(shap_relevances, q=0.95, by_circuit=False)
    joblib.dump(filt_i, f"filt_{i}.jbl")

    # return gpu id to queue
    #q.put(os.environ["CUDA_VISIBLE_DEVICES"])

    return filt_i


# Change loop

if __name__ == "__main__":

    project_folder = pathlib.Path(dotenv.find_dotenv()).parent.absolute()
    data_folder = project_folder.joinpath("notebooks", "data")

    X_fpath = data_folder.joinpath("features_covid.pkl")
    X = joblib.load(X_fpath)

    Xat_fpath = data_folder.joinpath("features.pkl")
    Xat = joblib.load(Xat_fpath)

    X = X[Xat.columns]

    Y_fpath = data_folder.joinpath("target_covid.pkl")
    Y = joblib.load(Y_fpath)

    n = 3
    cv = ShuffleSplit(n_splits=n, train_size=0.5, random_state=0)
    cv = list(cv.split(X, Y))
    cv = [(*train_test_split(cv[i][0], test_size=0.5), cv[i][1]) for i in range(n)]

    joblib.dump(cv, "cv.jbl")

    model = RandomForestRegressor(
        n_estimators=100, n_jobs=-1, max_features=0.1, max_depth=8
    )

    # In[43]:

    train_models = [
        clone(model)
        .set_params(random_state=i)
        .fit(X.iloc[cv[i][0], :], Y.iloc[cv[i][0], :])
        for i, split in enumerate(cv)
    ]

    for i, _ in enumerate(train_models):
        joblib.dump(train_models[i], f"model_{i}.jbl")

    # Define number of GPUs available
    N_GPU = 3

    # Put indices in queue

    #filts = Parallel(n_jobs=N_GPU, backend="multiprocessing")(
    #    delayed(runner)(n_split, data_folder) for n_split in range(n))

    input_list = [1, 1, 1, 1]
    r = lambda x: runner(x, data_folder)

    from functools import partial

    gpu_id_list = list(range(N_GPU))

    # create a new function that multiplies by 2
    r = partial(runner, data_folder, gpu_id_list)
    with multiprocessing.Pool(N_GPU) as pool:
        fs = pool.map(r, range(n))
        
    res = run_stability(model, X, Y, cv, fs)
    
    import json
    with open("stability.json", "w") as fjson:
        json.dump(res, fjson, indent=4)
                  
    # filts = Parallel(n_jobs=N_GPU, backend="multiprocessing")(
    #    delayed(r)(n_split) for n_split in range(n))

    #def send2gpu(q):
    #    gpu = q.get()
    #    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    #m = Manager()
    #q = m.Queue(maxsize=N_GPU)
    #for i in range(N_GPU):
    #    q.put(i)
    #r = partial(runner, data_folder, gpu_id_list)
    #p = Process(target=send2gpu, args=(q,))
    #p.start()
    #Parallel(n_jobs=N_GPU)(delayed(r)(i) for i in range(n))
    #q.put(None)
    #p.join()
