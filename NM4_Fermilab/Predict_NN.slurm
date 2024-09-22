#!/usr/bin/env bash
#SBATCH -p standard
#SBATCH --output=result_%a.out
#SBATCH --error=result_%a.err
#SBATCH -c 16
#SBATCH -t 48:00:00
#SBATCH -A spinquest_standard

module purge
module load anaconda
source activate tf-2.7
conda install scikit-learn
export LD_LIBRARY_PATH=~/.conda/envs/tf-2.7/lib:$LD_LIBRARY_PATH

python Predict_NN.py $1 $2