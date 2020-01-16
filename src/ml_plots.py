import numpy as np
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import sys
from sklearn.preprocessing import minmax_scale

translate_folder = Path("/mnt/lustre/scratch/CBRA/research/projects/holrd/")

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

plt.figure()
target.plot(kind="box", figsize=(16, 9))
fpath = results_path.joinpath(results_path, "task_distribution" + ".png")
plt.savefig(fpath, dpi=300)

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

gene_names = pd.read_csv(
    results_path.parent.parent.parent.parent.joinpath("entrez_sym-table.tsv"),
    #translate_folder.joinpath("gene_names.tsv"),
    sep=",",
    dtype={"entrez":str})
gene_names.set_index("entrez", drop=True, inplace=True)

d = gene_names.loc[gene_ids].copy()
q = d.isnull().values
d[q] = d[q].index.tolist()

gene_symbols = d.values.ravel()
rel_cv = pd.DataFrame(cv_stats["relevance"])

cut = rel_cv.median().mean() + 0.1 * rel_cv.median().std()

plt.figure()
rel_cv.median().sort_values(ascending=False).plot(figsize=(16, 9))
plt.title("No selected: {} of {}".format((rel_cv.median() > cut).sum(), rel_cv.median().size))
plt.axhline(cut, color="k", linestyle="--")
fnz = np.nonzero(rel_cv.median().sort_values(ascending=False).values.ravel() < cut)[0][0] - 1
fnz = rel_cv.median().sort_values(ascending=False).index[fnz]
plt.axvline(fnz, color="k", linestyle="--")
fpath = results_path.joinpath(results_path, "median_entrez" + ".png")
plt.savefig(fpath, dpi=300)

sel = pd.DataFrame(
    rel_cv.median().loc[rel_cv.median() > cut].values,
    index=rel_cv.median().loc[rel_cv.median() > cut].index,
    columns=["median_rel"],
    )
sel.sort_values(by="median_rel", ascending=False, inplace=True)
sel.index.name = "entrez"
sel.to_csv(results_path.joinpath(results_path, "median_sel_entrez" + ".csv"))

query_top = rel_cv.columns[rel_cv.median() > cut]
to_plot = rel_cv.loc[:, query_top].copy()
to_plot = to_plot.loc[:, to_plot.median().sort_values(ascending=False).index]

sns.set_context("poster")
plt.figure()
ax = to_plot.plot(kind="box", figsize=(16, 9))
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_ylabel("Relevance")
plt.tight_layout()
fname_base = "cv_relevance_distribution_entrez"
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
    fpath = translate_folder.joinpath("circuit_names.tsv")
    circuit_names = pd.read_csv(fpath, sep="\t", index_col=0, header=None, names=["NAME"])
    circuit_names.index = circuit_names.index.str.replace(r"-| ", ".")
    to_plot2.columns = circuit_names.loc[to_plot2.columns, "NAME"]

dfs[0].columns = to_plot2.columns
dfs[1].columns = to_plot2.columns

sns.set_context("paper")

fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
dfs[0].plot(kind="box", ax=axes[0])
axes[0].set_title("Train")
new_labels = to_plot2.columns.str.split(".").str[1]
axes[0].set_xticklabels(new_labels, rotation=90)
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
