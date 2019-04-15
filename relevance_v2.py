# coding: utf-8

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from timeit import default_timer as timer
import feather
import shap
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pickle

from src.datasets import get_disease_data


data_dir = "."
version = 2
inp = "targets"
out = "pathways"

SEED = 42
DISEASE = "fanconi"

MODE = "model"

out_dir = os.path.join(".", "rf", DISEASE, "{:02d}".format(version))

name = "hypmorf_{}_{:02d}".format(DISEASE, version)

# model: avereage of tree FI
# global: shap global via tree explainer
# task: shap kernel explainer, one relevance dataset per task
# precomputed: use precomputed shap values


def get_relevance(shap_values, feature_names, ind, r2_train, r2_test, columns, plot_path=None, max_display=20):
    """
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.style.use("ggplot")
    sns.set_context("paper")

    # see https://github.com/slundberg/shap/blob/master/shap/plots/summary.py

    feature_order = np.argsort(np.sum(np.abs(shap_values[ind]), axis=0))
    feature_order = feature_order[-min(max_display, len(feature_order)):]

    feature_inds = feature_order[:max_display]
    y_pos = np.arange(len(feature_inds))
    global_shap_values = np.abs(shap_values[ind]).mean(0)
    
    feature_names_relevant = [feature_names[i] for i in feature_inds]
    relevance_score = global_shap_values[feature_inds]
    
    relevance = pd.DataFrame(
        {                        
            "r2_train": r2_train[ind],
            "r2_test": r2_test[ind]
        },
        index=[columns[ind]]
    )

    for i in range(max_display):
        col_name = "var_{:02d}".format(i)
        relevance[col_name] = feature_names_relevant[i]
        col_name = "rel_{:02d}".format(i)
        relevance[col_name] = relevance_score[i]
    
    if plot_path:
        plt.figure(figsize=(8, 10))
        plt.barh(y_pos, global_shap_values[feature_inds], 0.7, align='center', color="cornflowerblue")
        plt.yticks(y_pos, fontsize=13)
        plt.gca().set_yticklabels(feature_names_relevant)
        plt.title("Feature relevance for circuit {}".format(columns[ind]))
        plt.tight_layout()
        plt.savefig("{}.pdf".format(plot_path))
        plt.savefig("{}.png".format(plot_path), dpi=300)
        plt.close()

    return relevance

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

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=SEED)

print("Data loaded for {} disease.".format(DISEASE))

model_fpath = os.path.join(out_dir, "{}_from_{}_to_{}.pkl".format(name, inp, out))

if MODE == "model":
    with open(model_fpath, "rb") as f:
        model = pickle.load(f)

    rel_fpath = os.path.join(
        out_dir, 
        "{}_from_{}_to_{}_global_relevance.tsv".format(name, inp, out)
    )

    rel = model.feature_importances_

    pd.DataFrame({"relevance":rel}, index=X_train.columns).to_csv(rel_fpath, sep="\t")

    exit(0)

elif MODE == "precomputed":
    #TODO define as constant
    split = "test" 

    with open(os.path.join(out_dir, "r2_raw_train.pkl"), "rb") as f:
        r2_raw_train = pickle.load(f)

    with open(os.path.join(out_dir, "r2_raw_test.pkl"), "rb") as f:
        r2_raw_test = pickle.load(f)

    with open(os.path.join(out_dir, "shap.pkl"), "rb") as f:
        shap_values = pickle.load(f)

else:
    print("Loading model...")
    with open(model_fpath, "rb") as f:
        model = pickle.load(f)
    print("Model loaded.")

    Y_train_hat = model.predict(X_train)
    Y_test_hat = model.predict(X_test)

    # Score
    r2_raw_train = r2_score(Y_train, Y_train_hat, multioutput="raw_values")
    r2_raw_test = r2_score(Y_test, Y_test_hat, multioutput="raw_values")

    with open("rf/r2_raw_train.pkl", "wb") as f:
        pickle.dump(r2_raw_train, f, -1)

    with open("rf/r2_raw_test.pkl", "wb") as f:
        pickle.dump(r2_raw_test, f, -1)
        # Relevance

    split = "test"

    if MODE == "task":
        print("{} Computing relevence by task...".format(name))
        ## Compute relevance    
        X_train_summary = shap.kmeans(X_train, 10)

        # Decorate prediction function
        predict = lambda x: model.predict(x)

        explainer = shap.KernelExplainer(predict, X_train_summary.data)
        # shap_values_train = explainer.shap_values(X_train)   

    elif MODE == "global":
        print("{} Computing global relevance...".format(name))
        explainer = shap.TreeExplainer(model)
    
    start = timer()
    shap_values = explainer.shap_values(X_test)
    end = timer()
    print("time: ", end - start)
    print("Relevance computed.")

    shap_fpath = os.path.join(
        out_dir, 
        "{}_{}_{}_from_{}_to_{}_shap.pkl".format(
            split, 
            name, 
            MODE,
            inp,
            out
        )
    )

    with open(shap_fpath, "wb") as f:
        pickle.dump(shap_values, f, -1)

if MODE == "task":
    for i in (range(Y_test.shape[1])):
        print(i, Y_test.columns[i])
        results_fpath = os.path.join(
            out_dir, 
            "{}_{}_{}_from_{}_to_{}_{}_relevance".format(
                split, 
                name, 
                MODE,
                inp,
                out, 
                Y_test.columns[i]
            )
        )

        relevance_i = get_relevance(
            shap_values,
            X_test.columns.tolist(),
            i,
            r2_raw_train,
            r2_raw_test,
            Y_test.columns.tolist(),
            plot_path=results_fpath
        )

        relevance_i.to_csv(
            "{}.tsv".format(results_fpath),
            sep="\t",
            index_label="circuit"
        )

