#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Entry CLI point for stab.
"""

import multiprocessing
import time

import joblib
import numpy as np
import shap
from sklearn.model_selection import train_test_split

from drexml.explain import compute_shap_fs, compute_shap_relevance, compute_shap_values_
from drexml.models import get_model
from drexml.utils import parse_stab

if __name__ == "__main__":
    import sys

    data_folder, n_iters, n_gpus, n_cpus, n_splits, debug = parse_stab(sys.argv)

    queue = multiprocessing.Queue()

    n_devices = n_gpus if n_gpus > 0 else n_cpus
    use_gpu = n_gpus > 0
    print(use_gpu, n_gpus, n_devices)
    fs_lst = []
    data_path = data_folder

    for gpu_ids in range(n_devices):
        queue.put(gpu_ids)

    features_orig_fpath = data_folder.joinpath("features.jbl")
    features_orig = joblib.load(features_orig_fpath)

    targets_orig_fpath = data_folder.joinpath("target.jbl")
    targets_orig = joblib.load(targets_orig_fpath)

    n_features = features_orig.shape[1]
    n_targets = targets_orig.shape[1]

    this_model = get_model(n_features, n_targets, n_cpus, debug, n_iters)

    X_learn, X_val, Y_learn, Y_val = train_test_split(
        features_orig, targets_orig, test_size=0.3, random_state=42
    )

    this_model.fit(X_learn, Y_learn)

    chunk_size = len(X_val) // (n_devices) + 1

    def runner(explainer, new, check_add):

        gpu_id = queue.get()
        # explainer = shap.GPUTreeExplainer(estimator, background)
        values = compute_shap_values_(new, explainer, check_add, gpu_id)
        queue.put(gpu_id)

        return values

    if use_gpu:
        this_explainer = shap.GPUTreeExplainer(this_model, X_learn)
    else:
        this_explainer = shap.TreeExplainer(this_model, X_learn)

    # bkg = shap.sample(features_learn, nsamples=1000, random_state=42)
    t = time.time()
    with joblib.parallel_backend("multiprocessing", n_jobs=n_devices):
        shap_values = joblib.Parallel()(
            joblib.delayed(runner)(
                explainer=this_explainer,
                new=gb,
                check_add=True,
            )
            for _, gb in X_val.groupby(np.arange(len(X_val)) // chunk_size)
        )

    # shape: (n_tasks, n_samples, n_features)
    shap_values = np.concatenate(shap_values, axis=1)
    elapsed = time.time() - t
    print(f"time {elapsed}")

    shap_summary = compute_shap_relevance(shap_values, X_val, Y_val)
    fs_df = compute_shap_fs(
        shap_summary, model=this_model, X=X_val, Y=Y_val, q="r2", by_circuit=True
    )
    fs_df = fs_df * 1

    # Save results
    shap_summary_fname = "shap_summary.tsv"
    shap_summary_fpath = data_folder.joinpath(shap_summary_fname)
    shap_summary.to_csv(shap_summary_fpath, sep="\t")
    print(f"Shap summary results saved to: {shap_summary_fpath}")

    # Save results
    fs_fname = "shap_selection.tsv"
    fs_fpath = data_folder.joinpath(fs_fname)
    fs_df.to_csv(fs_fpath, sep="\t")
    print(f"Shap selection results saved to: {fs_fpath}")
