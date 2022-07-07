#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Entry CLI point for stab.
"""


import joblib

from dreml.explain import compute_shap_fs, compute_shap_relevance, compute_shap_values
from dreml.models import AutoMORF
from dreml.utils import parse_stab

if __name__ == "__main__":
    import sys

    data_folder, n_iters, n_gpus, n_cpus, n_splits, debug = parse_stab(sys.argv)

    n_devices = n_gpus if n_gpus > 0 else n_cpus
    gpu = n_gpus > 0
    print(gpu, n_gpus, n_devices)
    fs_lst = []
    data_path = data_folder

    cv_fname = "cv.jbl"
    cv_fpath = data_path.joinpath(cv_fname)
    cv_gen = joblib.load(cv_fpath)

    features_fpath = data_path.joinpath("features.jbl")
    features = joblib.load(features_fpath)

    targets_fpath = data_path.joinpath("target.jbl")
    targets = joblib.load(targets_fpath)

    for i_split in range(n_splits):
        print(n_splits, i_split)
        sample_ids = cv_gen[i_split]

        # gpu_id = queue.get()
        filt_i = None
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # print("device", os.environ["CUDA_VISIBLE_DEVICES"])
        features_learn = features.iloc[sample_ids[0], :].copy()
        y_learn = targets.iloc[sample_ids[0], :].copy()

        features_val = features.iloc[sample_ids[1], :].copy()
        targets_val = targets.iloc[sample_ids[1], :].copy()

        model_fname = f"model_{i_split}.jbl"
        model_fpath = data_path.joinpath(model_fname)
        this_model = joblib.load(model_fpath)
        if isinstance(this_model, AutoMORF):
            pass

        # shap_values = compute_shap_values(
        #     estimator=this_model,
        #     background=features_learn,
        #     new=features_val,
        #     gpu=n_devices>1,
        #     n_devices=n_devices
        # )

        shap_values = compute_shap_values(
            estimator=this_model,
            background=features_learn,
            new=features_val,
            gpu=gpu,
            n_devices=n_devices,
        )

        shap_relevances = compute_shap_relevance(shap_values, features_val, targets_val)
        filt_i = compute_shap_fs(
            shap_relevances,
            model=this_model,
            X=features_val,
            Y=targets_val,
            q="r2",
            by_circuit=True,
        )

        fs_fname = f"filt_{i_split}.jbl"
        fs_fpath = data_path.joinpath(fs_fname)

        joblib.dump(filt_i, fs_fpath)
        fs_lst.append(filt_i)

    joblib.dump(fs_lst, data_folder.joinpath("fs.jbl"))
