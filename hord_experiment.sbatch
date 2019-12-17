#!/bin/bash
#SBATCH --export=ALL
#SBATCH --mail-user=carlos.loucera@juntadeandalucia.es
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --mem=0


# module load anaconda2
# conda activate holrd

module load python/3.7.3
module load python373-matplotlib
module load python373-pandas
module load python373-scikit-learn
module load python373-scipy
module load python373-pyarrow
module load python373-shap
module load python373-xlrd
module load python373-xlwt

source ~/.pyenv/holrd/bin/activate

# Avoid multithreading issues while computing relevances with SHAP.
OMP_NUM_THREADS=1

python hord.py --disease ${EXPERIMENT_ENV_FILE} --mlmodel ${MLMODEL} --opt ${OPT} --seed ${SEED} --mode ${MODE}
