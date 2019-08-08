from sklearn.externals import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

plt.style.use('fivethirtyeight')

use_circuit_dict = False

model_rel_fpath = "model_global_relevance.tsv"
model_rel = pd.read_csv(model_rel_fpath, sep="\t", index_col=0)

path = "shap_values_task_relevance.tsv"
task_rel = pd.read_csv(path, sep="\t", index_col=0)

circuit_ids = task_rel.columns
gene_ids = task_rel.index

cv_stats_fpath = "cv_stats.pkl"
cv_stats = joblib.load(cv_stats_fpath)

rel_cv = pd.DataFrame(cv_stats["relevance"], columns=gene_ids)

top_n = 50
query_top = rel_cv.mean().sort_values(ascending=False).index[:top_n]

to_plot = rel_cv.loc[:, query_top].copy()
to_plot.columns = model_rel.loc[rel_cv.loc[:, query_top].columns, "gene"].values

plt.figure()
ax = to_plot.plot(kind="box", figsize=(16, 9))
ax.set_xticklabels(ax.get_xticklabels(), rotation=90);
ax.set_ylabel("Relevance")
plt.tight_layout()
fname = "cv_relevance_distribution"
plt.savefig(fname + ".png", dpi=300)
plt.savefig(fname + ".pdf")
plt.savefig(fname + ".svg")
plt.close()

for stat in cv_stats.keys():
    if "_mo" in stat:
        dfs = []
        for i, split in enumerate(cv_stats[stat].keys()):
            df = pd.DataFrame(cv_stats[stat][split], columns=circuit_ids)
            dfs.append(df)

to_plot2 = (dfs[0] - (1- dfs[1])/2).copy()

if use_circuit_dict:
    fname = "name_circuits.tsv"
    circuit_names = pd.read_csv(fname, sep="\t", index_col=1)
    to_plot2.columns = circuit_names.loc[to_plot2.columns, "NAME"]

dfs[0].columns = to_plot2.columns
dfs[1].columns = to_plot2.columns

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
to_plot2.plot(kind="box", ax=axes[0])
axes[0].set_title("Train")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=70);
axes[0].set_ylabel(r"$R_{2}$")
dfs[1].plot(kind="box", ax=axes[1])
axes[1].set_title("Test")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=70);
# plt.suptitle("Performance score distribution")
plt.tight_layout()
fname = "cv_performance_dsitribution"
plt.savefig(fname + ".png", dpi=300)
plt.savefig(fname + ".pdf")
plt.savefig(fname + ".svg")
plt.close()
