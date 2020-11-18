import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib

exts = ["pdf", "png", "eps", "svg"]


def get_symbol_dict(results_path):
    fname = "entrez_sym-table.tsv"
    fpath = results_path.parent.parent.parent.parent.joinpath(fname)

    gene_names = pd.read_csv(fpath, sep=",", dtype={"entrez": str})
    gene_names.set_index("entrez", drop=True, inplace=True)

    gene_symbols_dict = gene_names.to_dict()["symb"]

    return gene_symbols_dict


def get_circuit_dict(translate_folder):
    fname = "circuit_names.tsv"
    fpath = translate_folder.joinpath(fname)
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


def get_features(results_path):
    features_fname = "features.pkl"
    features_fpath = results_path.joinpath(features_fname)
    features = pd.read_pickle(features_fpath)

    feature_ids = features.columns

    return features, feature_ids


def get_target(results_path):
    target_fname = "target.pkl"
    target_fpath = results_path.joinpath(target_fname)
    target = pd.read_pickle(target_fpath)

    target_ids = target.columns

    return target, target_ids


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
    for stat in cv_stats.keys():
        if "_mo" in stat:
            dfs = []
            for split in cv_stats[stat].keys():
                df = pd.DataFrame(cv_stats[stat][split], columns=circuit_ids)
                df.rename(columns=circuit_dict, inplace=True)
                dfs.append(df)

    sns.set_context("paper")

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    dfs[0].plot(kind="box", ax=axes[0], rot=90)
    axes[0].set_title("Train")

    dfs[1].plot(kind="box", ax=axes[1], rot=90)
    axes[1].set_title("Test")

    SMALL_SIZE = 8
    plt.rc("xtick", labelsize=SMALL_SIZE)
    fig.text(0.01, 0.5, r"$R_{2}$", ha="center", va="center", rotation=90)

    # plt.suptitle("Performance score distribution")
    plt.tight_layout()

    name = "cv_performance_distribution"
    for ext in extensions:
        fname = f"{name}.{ext}"
        fpath = pdir.joinpath(fname)
        plt.savefig(fpath, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()

    plt.close()


if __name__ == "__main__":

    _, folder, use_task, use_circuit_dict = sys.argv
    results_path = Path(folder)
    use_task = int(use_task)
    use_circuit_dict = int(use_circuit_dict)

    plt.style.use("fivethirtyeight")

    translate_folder = results_path.parent.parent.parent.parent.parent

    # load data
    target, circuit_ids = get_target(results_path)
    features, gene_ids = get_features(results_path)
    cv_stats = get_cv_stats(results_path)
    rel_cv = get_rel_cv(cv_stats, gene_ids)
    cut = get_cut_point(rel_cv)
    gene_symbols_dict = get_symbol_dict(results_path)
    circuit_dict = get_circuit_dict(translate_folder)

    if use_task:
        task_rel = get_shap_relevance(results_path)

    ## Median relevance
    for use_symb_dict in [False]:
        plot_median(rel_cv, use_symb_dict, gene_symbols_dict, results_path)

    save_median_df(rel_cv, cut, gene_symbols_dict, results_path)

    ## Relevance distribution
    for symb_dict in [None]:
        plot_relevance_distribution(rel_cv, cut, symb_dict, results_path)

    ## ML stats
    plot_stats(cv_stats, circuit_ids, circuit_dict, results_path)

    plt.close("all")
