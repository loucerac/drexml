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

BASE="mnt/lustre/scratch/CBRA/research/projects/holrd/experiments/"

declare -a DISEASES
DISEASES=("ALB" "EVC" "H" "RP" "WS")
declare -a EXPERIMENTS
EXPERIMENTS=("AT" "AT_C")

## now loop through the above array
for DISEASE in "${DISEASES[@]}"
do
    for EXPERIMENT in "${EXPERIMENTS[@]}"
    do
        MLPATH=$BASE/${DISEASE}/${EXPERIMENT}/ml/morf_train
        python src/ml_plots.py $MLPATH 0 1
    done
   # or do whatever with individual element of the array
done
