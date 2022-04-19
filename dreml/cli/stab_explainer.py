#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Entry CLI point for stab.
"""

import multiprocessing
import os
import pathlib

import joblib

from dreml.explain import compute_shap_fs, compute_shap_relevance, compute_shap_values

if __name__ == "__main__":
    import sys

    # client = Client('127.0.0.1:8786')

    _, data_folder, n_iters, n_gpus, n_cpus, debug = sys.argv
    n_iters = int(n_iters)
    data_folder = pathlib.Path(data_folder)
    n_gpus = int(n_gpus)
    use_gpu = n_gpus > 0
    n_cpus = int(n_cpus)
    debug = bool(int(debug))

    n_devices = n_gpus if n_gpus > 0 else n_cpus
    device_list = list(range(n_devices))
    n_splits = 5 if debug else 100

    queue = multiprocessing.Queue()

    def runner(data_path, gpu_flag, split_id):
        """Run instance."""
        gpu_id = queue.get()
        filt_i = None
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            print("device", os.environ["CUDA_VISIBLE_DEVICES"])

            cv_fname = "cv.jbl"
            cv_fpath = data_path.joinpath(cv_fname)
            cv_gen = joblib.load(cv_fpath)
            sample_ids = cv_gen[split_id]

            features_fpath = data_path.joinpath("features.jbl")
            features = joblib.load(features_fpath)

            targets_fpath = data_path.joinpath("target.jbl")
            targets = joblib.load(targets_fpath)

            features_learn = features.iloc[sample_ids[0], :]

            features_val = features.iloc[sample_ids[1], :]
            targets_val = targets.iloc[sample_ids[1], :]

            model_fname = f"model_{split_id}.jbl"
            model_fpath = data_path.joinpath(model_fname)
            this_model = joblib.load(model_fpath)

            shap_values = compute_shap_values(
                this_model, features_learn, features_val, gpu_flag
            )
            shap_relevances = compute_shap_relevance(
                shap_values, features_val, targets_val
            )
            filt_i = compute_shap_fs(
                shap_relevances,
                model=this_model,
                X=features_val,
                Y=targets_val,
                q="r2",
                by_circuit=True,
            )

            model_fname = f"filt_{split_id}.jbl"
            model_fpath = data_path.joinpath(model_fname)
            joblib.dump(filt_i, model_fpath)
            print(gpu_id, filt_i.sum(axis=1).value_counts())

        finally:
            queue.put(gpu_id)

        return filt_i

    # Put indices in queue
    for gpu_ids in range(n_devices):
        queue.put(gpu_ids)

    with joblib.parallel_backend("multiprocessing", n_jobs=n_devices):
        fs_lst = joblib.Parallel()(
            joblib.delayed(runner)(
                data_path=data_folder,
                gpu_flag=use_gpu,
                split_id=i_split,
            )
            for i_split in range(n_splits)
        )
    joblib.dump(fs_lst, data_folder.joinpath("fs.jbl"))
