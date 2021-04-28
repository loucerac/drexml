import sklearn
import numpy as np
import shap
from shap.utils import assert_import
from sklearn.ensemble import RandomForestRegressor


rs = np.random.RandomState(15921)  # pylint: disable=no-member
n = 100
m = 20
my = 10
datasets = {'regression': (rs.randn(n, m), rs.randn(n, my)),
            'binary': (rs.randn(n, m), rs.binomial(1, 0.5, n)),
            'multiclass': (rs.randn(n, m), rs.randint(0, 5, n))}

X, y = datasets["regression"]

model = RandomForestRegressor(n_jobs=-1, max_depth=8)
model.fit(X, y)
gpu_ex = shap.GPUTreeExplainer(model, X)
gpu_shap = gpu_ex.shap_values(X, check_additivity=True)
s = np.array(gpu_shap)
print(s.shape)
print(s.any())
