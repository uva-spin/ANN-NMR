#!/bin/bash
#SBATCH --job-name=test_job            # Job name
#SBATCH --output=test_job.out           # Output file
#SBATCH --error=test_job.err            # Error file
#SBATCH --time=00:05:00                 # Time limit (hh:mm:ss)
#SBATCH --partition=ptgroup              # Partition (queue) name
#SBATCH --ntasks=1                      # Number of tasks (processes)
#SBATCH --cpus-per-task=1               # Number of CPU cores per task
#SBATCH --mem=1G                        # Memory per node

# Load any necessary modules (if needed)
# module load python/3.8

# Display the hostname of the node that runs the job
hostname

# Optional: Print completion message
echo "SLURM test job completed"
