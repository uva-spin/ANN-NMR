#!/usr/bin/env bash
#SBATCH --output=result_%a.out
#SBATCH --error=result_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 6:00:00
#SBATCH -A spinquest_standard

module load miniforge/24.3.0-py3.11
module load apptainer tensorflow/2.17.0
apptainer run --nv $CONTAINERDIR/tensorflow-2.17.0.sif Training.py