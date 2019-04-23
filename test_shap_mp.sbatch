#!/bin/bash 
#SBATCH --export=ALL
#SBATCH --mail-user=carlos.loucera@juntadeandalucia.es
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --mem=0
#SBATCH -o jobs/%A_%a.out # Standard output
#SBATCH -e jobs/%A_%a.err # Standard error

#source ~/.bashrc
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
python notebooks/sandbox_20190416/test_shap_mp.py 

#python hord.py --disease fanconi --mlmodel morf --opt hyperopt --seed 42 --mode train  

# python rf.py
# python relevance_v2.py model
# python relevance_v2.py global
# python relevance_v2.py task
