#!/usr/bin/env bash
#SBATCH -p standard
#SBATCH --output=result_%a.out
#SBATCH --error=result_%a.err
#SBATCH -c 10
#SBATCH -t 01:00:00
#SBATCH -A spinquest_standard

module purge
module anaconda
source activate tf-2.7
export LD_LIBRARY_PATH=~/.conda/envs/tf-2.7/lib:$LD_LIBRARY_PATH

python Create_Sample_Data.py $1
