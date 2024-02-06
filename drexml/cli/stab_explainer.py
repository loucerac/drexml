#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Entry CLI point for stab.
"""

import multiprocessing
import time
import warnings

import joblib
import numpy as np
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", module="shap", message="IPython could not be loaded!"
    )
    warnings.filterwarnings("ignore", module="shap", category=NumbaDeprecationWarning)
    warnings.filterwarnings(
        "ignore", module="shap", category=NumbaPendingDeprecationWarning
    )
    import shap

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from drexml.explain import compute_shap_fs, compute_shap_relevance, compute_shap_values_
from drexml.models import extract_estimator, get_model
from drexml.utils import convert_names, parse_stab

if __name__ == "__main__":
    import sys

    data_folder, n_iters, n_gpus, n_cpus, n_splits, debug, add = parse_stab(sys.argv)
    this_seed = 82
    queue = multiprocessing.Queue()

    n_devices = n_gpus if n_gpus > 0 else n_cpus
    gpu = n_gpus > 0
    print(gpu, n_gpus, n_devices)
    fs_lst = []
    data_path = data_folder

    for gpu_ids in range(n_devices):
        queue.put(gpu_ids)

    features_fpath = data_path.joinpath("features.jbl")
    features = joblib.load(features_fpath)

    targets_fpath = data_path.joinpath("target.jbl")
    targets = joblib.load(targets_fpath)

    if n_splits > 1:
        print(f"loading cv {n_splits=}")
        cv_fname = "cv.jbl"
        cv_fpath = data_path.joinpath(cv_fname)
        cv_gen = joblib.load(cv_fpath)
    else:
        ids = np.arange(features.shape[0])
        learn_ids, val_ids = train_test_split(
            ids, test_size=0.3, random_state=this_seed
        )

    for i_split in range(n_splits):
        print(n_splits, i_split)

        if n_splits > 1:
            sample_ids = cv_gen[i_split]
            learn_ids = sample_ids[0]
            val_ids = sample_ids[1]

        print(f"dataset sizes {learn_ids.size=} {val_ids.size=}")

        # gpu_id = queue.get()
        filt_i = None
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # print("device", os.environ["CUDA_VISIBLE_DEVICES"])
        features_learn = features.iloc[learn_ids, :].copy()
        targets_learn = targets.iloc[learn_ids, :].copy()

        features_val = features.iloc[val_ids, :].copy()
        targets_val = targets.iloc[val_ids, :].copy()

        if n_splits == 1:
            print(f"fit final model {i_split=} {n_splits=}")
            use_imputer = features.isna().any(axis=None)
            this_model = get_model(
                features.shape[1], targets.shape[1], n_cpus, debug, n_iters, use_imputer
            )
            if isinstance(this_model, Pipeline):
                # set final estimator params if pipeline
                this_model[-1].set_params(n_jobs=n_cpus, random_state=this_seed)
            else:
                this_model.set_params(n_jobs=n_cpus, random_state=this_seed)
            this_model.fit(features_learn, targets_learn)
        else:
            model_fname = f"model_{i_split}.jbl"
            model_fpath = data_path.joinpath(model_fname)
            this_model = joblib.load(model_fpath)

        n_chunks = max(1, n_devices)
        chunk_size = len(features_val) // (n_chunks) + 1

        print(f"{add=}")

        def runner(model, bkg, new, check_add, use_gpu):
            gpu_id = queue.get()

            if use_gpu:
                explainer = shap.GPUTreeExplainer(model, bkg)
            else:
                explainer = shap.TreeExplainer(model, bkg)

            # explainer = shap.GPUTreeExplainer(estimator, background)
            values = compute_shap_values_(new, explainer, check_add, gpu_id)
            queue.put(gpu_id)

            return values

        # bkg = shap.sample(features_learn, nsamples=1000, random_state=42)
        t = time.time()
        features_bkg = features_learn.copy()
        if features_learn.shape[0] > 1000:
            features_bkg = features_learn.sample(n=1000, random_state=42)
        else:
            features_bkg = features_learn

        estimator = extract_estimator(this_model)

        with joblib.parallel_backend("multiprocessing", n_jobs=n_devices):
            shap_values = joblib.Parallel()(
                joblib.delayed(runner)(
                    model=estimator,
                    bkg=features_bkg,
                    new=gb,
                    check_add=add,
                    use_gpu=gpu,
                )
                for _, gb in features_val.groupby(
                    np.arange(len(features_val)) // chunk_size
                )
            )

        # shape: (n_tasks, n_samples, n_features)
        shap_values = np.concatenate(shap_values, axis=1)
        elapsed = time.time() - t

        print(i_split, elapsed)
        shap_relevances = compute_shap_relevance(shap_values, features_val, targets_val)
        filt_i = compute_shap_fs(
            shap_relevances,
            model=this_model,
            X=features_val,
            Y=targets_val,
            q="r2",
            by_circuit=True,
        )

        if n_splits > 1:
            fs_fname = f"filt_{i_split}.jbl"
            fs_fpath = data_path.joinpath(fs_fname)

            joblib.dump(filt_i, fs_fpath)
            fs_lst.append(filt_i)

    if n_splits > 1:
        joblib.dump(fs_lst, data_folder.joinpath("fs.jbl"))
    else:
        # Save results
        shap_summary_fname = "shap_summary.tsv"
        shap_summary_fpath = data_folder.joinpath(shap_summary_fname)
        shap_relevances.to_csv(shap_summary_fpath, sep="\t")
        print(f"Shap summary results saved to: {shap_summary_fpath}")

        shap_summary_renamed = convert_names(
            shap_relevances,
            ["circuits", "genes"],
            axis=[0, 1],
        )
        shap_summary_renamed.to_csv(
            shap_summary_fpath.absolute().parent.joinpath(
                f"{shap_summary_fpath.stem}_symbol.tsv"
            ),
            sep="\t",
            index_label="circuit_name",
        )

        # Save results
        fs_fname = "shap_selection.tsv"
        fs_fpath = data_folder.joinpath(fs_fname)
        (filt_i * 1).to_csv(fs_fpath, sep="\t")
        print(f"Shap selection results saved to: {fs_fpath}")

        fs_renamed = convert_names(filt_i, ["circuits", "genes"], axis=[0, 1])
        fs_renamed.to_csv(
            fs_fpath.absolute().parent.joinpath(f"{fs_fpath.stem}_symbol.tsv"),
            sep="\t",
            index_label="circuit_name",
        )
