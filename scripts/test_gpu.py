from sklearn.ensemble import RandomForestRegressor

import numpy as np
import shap

# simulate some binary data and a linear outcome with an interaction term
# note we make the features in X perfectly independent of each other to make
# it easy to solve for the exact SHAP values
N = 2000
X = np.zeros((N,5))
X[:1000,0] = 1
X[:500,1] = 1
X[1000:1500,1] = 1
X[:250,2] = 1
X[500:750,2] = 1
X[1000:1250,2] = 1
X[1500:1750,2] = 1
X[:,0:3] -= 0.5
y = 2*X[:,0] - 3*X[:,1]

# train a model with single tree
model = RandomForestRegressor(n_estimators=1000).fit(X, y)
print("Model error =", np.linalg.norm(y-model.predict(X)))

# make sure the SHAP values add up to marginal predictions
explainer = shap.GPUTreeExplainer(model)
shap_values = explainer.shap_values(X)
