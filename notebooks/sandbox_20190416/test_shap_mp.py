from sklearn.externals import joblib
import os
import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import shap
from sklearn.ensemble import RandomForestRegressor

from dotenv import find_dotenv, load_dotenv
os.system("taskset -p 0xff %d" % os.getpid())

dotenv_filepath = find_dotenv()
load_dotenv(dotenv_filepath)
project_path = os.path.dirname(dotenv_filepath)

# add the 'src' directory as one where we can import modules
src_dir = os.path.join(project_path)
sys.path.append(src_dir)

from src.datasets import get_disease_data

DISEASE = "fanconi"
SEED = 42


def main():
    X, Y, circuits, genes, clinical = get_disease_data(DISEASE)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=SEED)

    print("Load model.")

    model = RandomForestRegressor(n_estimators=100, n_jobs=-1)

    print("Model Loaded.")

    print("Fit model.")
    model.fit(X_train, Y_train)
    print("Model fitted.")

    print("Computing SHAP values.")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    print("SHAP values computed.")

    with open("shap_values.pkl", "wb") as f:
        joblib.dump(shap_values, f)


if __name__ == "__main__":
    main()