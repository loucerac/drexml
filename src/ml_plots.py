from sklearn.externals import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import sys

_, folder, use_task, use_circuit_dict = sys.argv
results_path = Path(folder)
use_task = int(use_task)
use_circuit_dict = int(use_circuit_dict)

print(results_path, use_task, use_circuit_dict)

plt.style.use('fivethirtyeight')

# load data
features_fname = "features.pkl"
features_fpath = results_path.joinpath(results_path, features_fname)
features = pd.read_pickle(features_fpath)

target_fname = "target.pkl"
target_fpath = results_path.joinpath(results_path, target_fname)
target = pd.read_pickle(target_fpath)

circuit_ids = target.columns
gene_ids = features.columns

if use_task:
    path = results_path.joinpath("shap_values_task_relevance.tsv")
    task_rel = pd.read_csv(path, sep="\t", index_col=0)

# CV relevance
model_rel_fpath = results_path.joinpath("model_global_relevance.tsv")
model_rel = pd.read_csv(model_rel_fpath, sep="\t", index_col=0)

cv_stats_fpath = results_path.joinpath(results_path, "cv_stats.pkl")
cv_stats = joblib.load(cv_stats_fpath)

rel_cv = pd.DataFrame(cv_stats["relevance"], columns=gene_ids)

top_n = 50
query_top = rel_cv.median().sort_values(ascending=False).index[:top_n]

to_plot = rel_cv.loc[:, query_top].copy()
if use_circuit_dict:
	to_plot.columns = model_rel.loc[rel_cv.loc[:, query_top].columns, "gene"].values

plt.figure()
ax = to_plot.plot(kind="box", figsize=(16, 9))
ax.set_xticklabels(ax.get_xticklabels(), rotation=90);
ax.set_ylabel("Relevance")
plt.tight_layout()
fname_base = "cv_relevance_distribution"
fpath = results_path.joinpath(results_path, fname_base + ".png")
plt.savefig(fpath, dpi=300)
fpath = results_path.joinpath(results_path, fname_base + ".pdf")
plt.savefig(fpath)
fpath = results_path.joinpath(results_path, fname_base + ".svg")
plt.savefig(fpath)
fpath = results_path.joinpath(results_path, fname_base + ".eps")
plt.savefig(fpath, format="eps")
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

sns.set_context("paper")

fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
dfs[0].plot(kind="box", ax=axes[0])
axes[0].set_title("Train")
new_labels = to_plot2.columns.str.split(".").str[1]
axes[0].set_xticklabels(new_labels, rotation=90);
# axes[0].set_ylabel(r"$R_{2}$")
dfs[1].plot(kind="box", ax=axes[1])
axes[1].set_title("Test")
axes[1].set_xticklabels(new_labels, rotation=90)

fig.text(0.01, 0.5, r"$R_{2}$", ha="center", va="center", rotation=90)

# plt.suptitle("Performance score distribution")
plt.tight_layout()

fname_base = "cv_performance_distribution"
fpath = results_path.joinpath(results_path, fname_base + ".png")
plt.savefig(fpath, dpi=300)
fpath = results_path.joinpath(results_path, fname_base + ".pdf")
plt.savefig(fpath)
fpath = results_path.joinpath(results_path, fname_base + ".svg")
plt.savefig(fpath)
fpath = results_path.joinpath(results_path, fname_base + ".eps")
plt.savefig(fpath, format="eps")

plt.close()
