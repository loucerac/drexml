import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
import dotenv
import pathlib

dotenv_filepath = dotenv.find_dotenv()
project_path = pathlib.Path(dotenv_filepath).parent
resources_folder = project_path.joinpath("resources")
exts = ["pdf", "png", "eps", "svg"]


def get_symbol_dict():
    fname = "entrez_sym-table.tsv"
    fpath = resources_folder.joinpath(fname)

    gene_names = pd.read_csv(fpath, sep="\t", dtype={"entrez": str})
    gene_names.set_index("entrez", drop=True, inplace=True)

    gene_symbols_dict = gene_names.to_dict()["symbol"]

    return gene_symbols_dict


def get_circuit_dict():
    fname = "circuit_names.tsv"
    fpath = resources_folder.joinpath(fname)
    circuit_names = pd.read_csv(
        fpath, sep="\t", index_col=0, header=None, names=["NAME"]
    )
    circuit_names.index = circuit_names.index.str.replace(r"-| ", ".")

    return circuit_names["NAME"].to_dict()


def plot_median(rel_cv, use_symb, symb_dict, pdir, extensions=exts):

    name = "median"
    if use_symb:
        name = f"{name}_symbol"
        df_plot = rel_cv.rename(columns=symb_dict).copy()
    else:
        name = f"{name}_entrez"
        df_plot = rel_cv.copy()

    df_plot = df_plot.median().sort_values(ascending=False)
    cut = get_cut_point(rel_cv)

    plt.figure()
    df_plot.plot(figsize=(16, 9), rot=90)
    plt.axhline(cut, color="k", linestyle="--")
    fnz = np.nonzero(df_plot.values.ravel() < cut)[0][0] - 1
    plt.axvline(fnz, color="k", linestyle="--")
    plt.xlabel("Gene")
    plt.ylabel("Median Relevance")
    plt.tight_layout()
    for ext in extensions:
        fname = f"{name}.{ext}"
        fpath = pdir.joinpath(fname)
        plt.savefig(fpath, dpi=300, bbox_inches="tight", pad_inches=0)


def plot_task_distribution(target, pdir, pdict, extensions=exts):
    name = "task_distribution"
    df_plot = target.apply(np.log1p).rename(columns=pdict)

    plt.figure()
    df_plot.plot(kind="box", figsize=(16, 9))
    plt.tight_layout()
    for ext in extensions:
        fname = f"{name}.{ext}"
        fpath = pdir.joinpath(fname)
        plt.savefig(fpath, dpi=300, bbox_inches="tight", pad_inches=0)


def get_shap_relevance(results_path):
    fname = "shap_values_task_relevance.tsv"
    fpath = results_path.joinpath(fname)
    task_rel = pd.read_csv(fpath, sep="\t", index_col=0)

    return task_rel


def get_cv_stats(results_path):
    fname = "cv_stats.pkl"
    fpath = results_path.joinpath(results_path, fname)
    cv_stats = joblib.load(fpath)

    return cv_stats


def get_rel_cv(cv_stats, gene_ids):
    rel_cv = pd.DataFrame(cv_stats["relevance"], columns=gene_ids)

    return rel_cv


def get_cut_point(rel_cv):
    return rel_cv.median().mean() + 0.1 * rel_cv.median().std()


def save_median_df(rel_cv, cut, symbol_dict, results_path):
    entrez_list = rel_cv.median().index.values.ravel()
    symbol_list = [
        symbol_dict[entrez] if entrez in symbol_dict.keys() else None
        for entrez in entrez_list
    ]
    is_selected = (rel_cv.median() > cut).values.ravel()
    median_rel = rel_cv.median().values.ravel()

    sel = pd.DataFrame(
        {
            "entrez": entrez_list,
            "symbol": symbol_list,
            "is_selected": is_selected,
            "median_rel": median_rel,
        }
    )

    sel.sort_values(by="median_rel", ascending=False, inplace=True)
    sel.to_csv(results_path.joinpath("median_relevance" + ".csv"), index=False)


def plot_relevance_distribution(rel_cv, cut, symb_dict, pdir, extensions=exts):
    query_top = rel_cv.columns[rel_cv.median() > cut]
    to_plot = rel_cv.loc[:, query_top].copy()
    to_plot = to_plot.loc[:, to_plot.median().sort_values(ascending=False).index]

    if symb_dict is not None:
        to_plot.rename(columns=symb_dict, inplace=True)

    # sns.set_context("poster")
    plt.figure()
    ax = to_plot.plot(kind="box", figsize=(16, 9), rot=90)
    ax.set_ylabel("Relevance")
    plt.tight_layout()

    name = "cv_relevance_distribution"
    if symb_dict is None:
        name = f"{name}_entrez"
    else:
        name = f"{name}_symbol"

    for ext in extensions:
        fname = f"{name}.{ext}"
        fpath = pdir.joinpath(fname)
        plt.savefig(fpath, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def plot_stats(cv_stats, circuit_ids, circuit_dict, pdir, extensions=exts):
    stat = "r2_mo"
    sns.set_theme(context="paper", style="whitegrid")

    d = pd.DataFrame(cv_stats[stat]["test"], columns=circuit_ids)
    d = d.rename(columns=circuit_dict)
    d = d.melt(value_name="score", var_name="Circuit")
    d["score"] = 1 - d["score"]
    plt.figure(figsize=(10, 20))
    g = sns.boxplot(x="score", y="Circuit", data=d, color="lightgray")
    g.set_xlabel(
        "10 times 10-fold $1 - R^{2}$ Cross-Validation Distribution", fontsize=12
    )
    g.set_ylabel("Circuit", fontsize=12)
    sns.despine(left=True, bottom=True)

    name = f"{stat}_cv_performance_distribution"
    for ext in extensions:
        fname = f"{name}.{ext}"
        fpath = pdir.joinpath(fname)
        plt.savefig(fpath, dpi=300, bbox_inches="tight", pad_inches=0)


def convert_frame_ids(fname, results_path, circuit_dict, gene_symbols_dict):
    fpath = results_path.joinpath(fname)
    frame = pd.read_csv(fpath, sep="\t", index_col=0)
    frame = frame.rename(index=circuit_dict)
    frame = frame.rename(columns=gene_symbols_dict)

    fname_out = fname.replace(".tsv", "_symbol.tsv")
    fpath_out = results_path.joinpath(fname_out)
    frame.to_csv(fpath_out, sep="\t")

    return frame
