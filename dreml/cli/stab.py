#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Carlos Loucera <carlos.loucera@juntadeandalucia.es>
Author: Maria Pena Chilet <maria.pena.chilet.ext@juntadeandalucia.es>
Author: Marina Esteban <marina.estebanm@gmail.com>

Entry CLI point for stab.
"""

import multiprocessing
import os
from functools import partial

import click
import joblib
from sklearn.base import clone
from sklearn.model_selection import ShuffleSplit, train_test_split

from dreml.explain import (compute_shap_fs, compute_shap_relevance,
                           compute_shap_values)
from dreml.models import get_model
from dreml.utils import get_number_cuda_devices, get_out_path, get_version


def run_stab(data_folder, n_iters, n_gpus, n_cpus, debug):
    """[summary]

    Parameters
    ----------
    data_folder : [type]
        [description]
    n_iters : [type]
        [description]
    n_gpus : [type]
        [description]
    n_cpus : [type]
        [description]
    """

    X_fpath = data_folder.joinpath("features.jbl")
    X = joblib.load(X_fpath)

    Y_fpath = data_folder.joinpath("target.jbl")
    Y = joblib.load(Y_fpath)

    n_splits = 100
    cv = ShuffleSplit(n_splits=n_splits, train_size=0.5, random_state=0)
    cv = list(cv.split(X, Y))
    cv = [
        (*train_test_split(cv[i][0], test_size=0.3), cv[i][1]) for i in range(n_splits)
    ]

    fname = f"cv.jbl"
    fpath = data_folder.joinpath(fname)
    joblib.dump(cv, fpath)
    n_features = X.shape[1]
    n_targets = Y.shape[1]

    model = get_model(n_features, n_targets, n_cpus, n_iters, debug)

    for i, split in enumerate(cv):
        with joblib.parallel_backend("multiprocessing", n_jobs=n_cpus):
            model_ = clone(model)
            model_.set_params(random_state=i)
            model_.fit(X.iloc[split[0], :], Y.iloc[split[0], :])
            fname = f"model_{i}.jbl"
            fpath = data_folder.joinpath(fname)
            joblib.dump(model_, fpath)
            print(f"50-50 split {i}")

    n_devices = n_gpus if n_gpus > 0 else n_cpus

    # filts = Parallel(n_jobs=N_GPU, backend="multiprocessing")(
    #    delayed(runner)(n_split, data_folder) for n_split in range(n))

    device_list = list(range(n_devices))

    queue = multiprocessing.Queue()
    # initialize the queue with the GPU ids

    def runner(data_folder, gpu, gpu_id_list, i):
        """Run instance."""
        cpu_name = multiprocessing.current_process().name
        cpu_id = int(cpu_name[cpu_name.find("-") + 1 :]) - 1
        # print(gpu0, gpu)
        gpu_id = queue.get()
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        # print(os.environ["CUDA_VISIBLE_DEVICES"])
        filt_i = None
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            print("device", os.environ["CUDA_VISIBLE_DEVICES"])

            fname = f"cv.jbl"
            fpath = data_folder.joinpath(fname)
            cv = joblib.load(fpath)
            split = cv[i]

            X_fpath = data_folder.joinpath("features.jbl")
            X = joblib.load(X_fpath)

            Y_fpath = data_folder.joinpath("target.jbl")
            Y = joblib.load(Y_fpath)

            X_learn = X.iloc[split[0], :]
            Y_learn = Y.iloc[split[0], :]

            X_val = X.iloc[split[1], :]
            Y_val = Y.iloc[split[1], :]

            fname = f"model_{i}.jbl"
            fpath = data_folder.joinpath(fname)
            model = joblib.load(fpath)

            shap_values = compute_shap_values(model, X_learn, X_val, gpu)
            shap_relevances = compute_shap_relevance(shap_values, X_val, Y_val)
            filt_i = compute_shap_fs(
                shap_relevances, model=model, X=X_val, Y=Y_val, q="r2", by_circuit=True
            )

            fname = f"filt_{i}.jbl"
            fpath = data_folder.joinpath(fname)
            joblib.dump(filt_i, fpath)
            print(gpu_id, filt_i.sum(axis=1).value_counts())

        finally:
            queue.put(gpu_id)

        return filt_i

    # Put indices in queue
    for gpu_ids in range(n_devices):
        queue.put(gpu_ids)

    r = partial(runner, data_folder, n_gpus, device_list)

    with joblib.parallel_backend("multiprocessing", n_jobs=n_devices):
        fs = joblib.Parallel()(
            joblib.delayed(r)(i_split) for i_split in range(n_splits)
        )
    joblib.dump(fs, data_folder.joinpath("fs.jbl"))


@click.command()
@click.option(
    "--debug/--no-debug", is_flag=True, default=False, help="Flag to run in debug mode."
)
@click.option(
    "--n-iters",
    default=0,
    help="Number of Optimization iterations. 0 means use sensible hyperparameters.",
)
@click.option(
    "--n-gpus",
    is_flag=True,
    default=-1,
    help="Number of CUDA-enabled GPUs, -1 use all decices.",
)
@click.option("--n-cpus", default=-1, help="Number of jobs, -1 uses all devices.")
@click.argument("disease-path", type=click.Path(exists=True))
@click.version_option(get_version())
def stability(disease_path, debug, n_iters, n_gpus, n_cpus):
    """[summary]"""

    print(f"running DREML orchestrate v {get_version()}")
    output_folder = get_out_path(disease_path)
    data_folder = output_folder.joinpath("tmp")

    if n_gpus < 0:
        n_gpus = get_number_cuda_devices()

    if n_cpus < 0:
        n_cpus = multiprocessing.cpu_count()
