# coding: utf-8

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from timeit import default_timer as timer
import feather
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

import shap
from hpsklearn import HyperoptEstimator, random_forest_regression
from hyperopt import tpe

from src.datasets import get_disease_data


data_dir = "."
version = 2
inp = "targets"
out = "pathways"
SEED = 42
DISEASE = "fanconi"
MODE = "gobal"
out_dir = os.path.join(".", "rf", DISEASE, version)

name = "hypmorf_{}_{:02d}".format(DISEASE, version)

train = True

out_folder = out_dir

# X_train = feather.read_dataframe(os.path.join(data_dir, "X_train.feather"))
# X_test = feather.read_dataframe(os.path.join(data_dir, "X_test.feather"))
# Y_train = feather.read_dataframe(os.path.join(data_dir, "Y_train.feather"))
# Y_test = feather.read_dataframe(os.path.join(data_dir, "Y_test.feather"))

# X_train.set_index("index", drop=True, inplace=True)
# X_test.set_index("index", drop=True, inplace=True)
# Y_train.set_index("index", drop=True, inplace=True)
# Y_test.set_index("index", drop=True, inplace=True)

print("loading data")
# X_train = feather.read_dataframe(os.path.join(data_dir, "X_train.feather"))
# X_test = feather.read_dataframe(os.path.join(data_dir,"X_test.feather"))
# Y_train = feather.read_dataframe(os.path.join(data_dir,"Y_train.feather"))
# Y_test = feather.read_dataframe(os.path.join(data_dir,"Y_test.feather"))

# X_train.set_index("index", drop=True, inplace=True)
# X_test.set_index("index", drop=True, inplace=True)
# Y_train.set_index("index", drop=True, inplace=True)
# Y_test.set_index("index", drop=True, inplace=True)

X, Y, circuits, genes = get_disease_data(DISEASE)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_satet=42)

print("Data loaded for {} disease.".format(DISEASE))

n_pathways = Y_train.columns.str.split(".").str[1].unique().size

model_fpath = os.path.join(
    out_folder, 
    "{}_from_{}_to_{}.pkl".format(name, inp, out))

np_fpath = os.path.join(
    out_folder, 
    "{}_from_{}_to_{}_preds.npy".format(name, inp, out))

if train:
#     model = RandomForestRegressor(
#         n_estimators=10**4,
#         n_jobs=22
#     )

    np.random.seed(42)

    # Instantiate a HyperoptEstimator with the search space and number of evaluations

    estim = HyperoptEstimator(
        regressor=random_forest_regression('morf', n_jobs=24),
        algo=tpe.suggest,
        max_evals=1000,
        trial_timeout=120)

    start = timer()

#     model.fit(X_train, Y_train)
    estim.fit(X_train.values, Y_train.values)

    end = timer()
    print("Training time: {}".format(end - start))
    
    model = estim.best_model()["learner"]
    
    np.save(np_fpath, model.predict(X_test))

    with open(model_fpath, 'wb') as f:
        pickle.dump(model, f, -1)

    print("Model saved in path: %s" % model_fpath)
else:
    import pickle 

    with open(model_fpath, "rb") as f:
        rf = pickle.load(f)
    
    with open(model_fpath, 'wb') as f:
        pickle.dump(rf, f, -1)

# Y_train_hat = model.predict(X_train)
# Y_test_hat = model.predict(X_test)
    
# # Score
# r2_raw_train = r2_score(Y_train, mean_train, multioutput="raw_values")
# r2_raw_test = r2_score(Y_test, mean_test, multioutput="raw_values")

# # Relevance

# explainer = shap.KernelExplainer(model.predict, X_train)
# shap_values_train = explainer.shap_values(X_train)   
# shap_values_test = explainer.shap_values(X_test) 
    
# for i in (range(Y_test.shape[1])):

# 	fpath = os.path.join(
# 	    out_folder, 
# 	    "{}_from_{}_to_{}_relevance".format(name, inp, out, Y_test.columns[i])
# 	)

# 	relevance_i = get_relevance(
# 	    shap_values,
# 	    X_test.columns.tolist(),
# 	    i,
# 	    r2_raw_train,
# 	    r2_raw_test,
# 	    Y_test.columns.tolist(),
# 	    plot_path=fpath)

# 	relevance_i.to_csv(
# 	    "{}.tsv".format(fpath),
# 	    sep="\t",
# 	    index_label="circuit"
# 	)
