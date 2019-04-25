#!/usr/bin/env bash
BATCH_FILE="hord.sbatch"
DISEASE="fanconi"
MLMODEL="morf"
OPT="hyperopt"
SEED=42
MODE="train"
PATHWAY1="hsa03460m"
PATHWAY2="hsa04110"

out = $(python hord.py --disease ${DISEASE} --mlmodel ${MLMODEL} --opt ${OPT} --seed ${SEED} --mode ${MODE})

git archive -o latest.zip HEAD
mv lastest.zip ${out}
