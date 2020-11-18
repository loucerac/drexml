#!/bin/bash
#SBATCH --export=ALL
#SBATCH --mail-user=carlos.loucera@juntadeandalucia.es
#SBATCH --mail-type=ALL
#SBATCH --mem=200g


module load anaconda2
conda activate ./.venv

python hord.py --disease ${EXPERIMENT_ENV_FILE} --mlmodel ${MLMODEL} --opt ${OPT} --seed ${SEED} --mode ${MODE}
