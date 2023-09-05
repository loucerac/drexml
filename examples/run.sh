#!/usr/bin/env bash

mkdir -p results

conda create -y -p ./.venv python=3.10 jupyterlab -c conda-forge

