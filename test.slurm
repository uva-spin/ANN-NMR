#!/bin/bash
#SBATCH --job-name=test_job            # Job name
#SBATCH --output=test_job.out           # Output file
#SBATCH --error=test_job.err            # Error file
#SBATCH --time=00:05:00                 # Time limit (hh:mm:ss)
#SBATCH --partition=ptgroup              # Partition (queue) name
#SBATCH --ntasks=1                      # Number of tasks (processes)
#SBATCH --cpus-per-task=1               # Number of CPU cores per task
#SBATCH --mem=1G                        # Memory per node


hostname

echo "SLURM test job completed"
