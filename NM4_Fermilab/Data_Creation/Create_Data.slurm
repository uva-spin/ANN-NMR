#!/usr/bin/env bash
#SBATCH -p standard
#SBATCH --job-name=data_generation
#SBATCH --output=job_%A_%a.out
#SBATCH --error=job_%A_%a.err
#SBATCH -c 16
#SBATCH -t 12:00:00
#SBATCH -A spinquest_standard
#SBATCH --array=1-1000     # Generate 1000 sets of data in parallel
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@example.edu

# ===== Global Configuration =====
# Set these variables to configure the job
PROJECT_DIR="/project/ptgroup/Devin/Neural_Network"
OUTPUT_DIR="${PROJECT_DIR}/Testing_Data_v5"
OUTPUT_FILENAME="Sample_Data_1M.csv"  # Global variable for output filename
MODE="deuteron"                       # Options: deuteron, proton
NUM_SAMPLES=1000                      # Number of samples per task 
ADD_NOISE=1                           # 0=no noise, 1=add noise

# Print job details for logging
echo "Starting job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Configuration:"
echo "  - Output directory: ${OUTPUT_DIR}"
echo "  - Output filename: ${OUTPUT_FILENAME}"
echo "  - Mode: ${MODE}"
echo "  - Samples per job: ${NUM_SAMPLES}"
echo "  - Add noise: ${ADD_NOISE}"

# Load required modules
module purge
module load miniforge/24.3.0-py3.11

# Set working directory and create output directories
SCRIPT_DIR="${PROJECT_DIR}/scripts"

mkdir -p ${OUTPUT_DIR}
cd ${SCRIPT_DIR}

# ===== Part 1: Generate Signal Data =====
echo "Generating signal data..."

# Run signal generator with array task ID as the job identifier
python signal_generator.py ${SLURM_ARRAY_TASK_ID} ${MODE} ${NUM_SAMPLES} ${ADD_NOISE}

# Check if generation was successful
if [ $? -ne 0 ]; then
    echo "Error: Signal generation failed. Exiting."
    exit 1
fi

echo "Signal generation complete for task ${SLURM_ARRAY_TASK_ID}"

# ===== Part 2: Combine CSV Files (only run by the last array job) =====
# We'll use a job dependency to trigger this after all array jobs finish
if [ "${SLURM_ARRAY_TASK_ID}" -eq "${SLURM_ARRAY_TASK_MAX}" ]; then
    echo "This is the last array job. Submitting the merge job."
    
    # Create the merge job script with the OUTPUT_FILENAME variable passed through
    MERGE_SCRIPT="${SCRIPT_DIR}/merge_job.sh"
    cat > ${MERGE_SCRIPT} << EOF
#!/usr/bin/env bash
#SBATCH -p standard
#SBATCH --job-name=merge_csv
#SBATCH --output=merge_%j.out
#SBATCH --error=merge_%j.err
#SBATCH -c 16
#SBATCH -t 12:00:00
#SBATCH -A spinquest_standard
#SBATCH --mem=128G          # Large memory for merging 1000 files

# Global variables passed from parent script
OUTPUT_DIR="${OUTPUT_DIR}"
OUTPUT_FILENAME="${OUTPUT_FILENAME}"

# Load required modules
module purge
module load miniforge/24.3.0-py3.11

# Set working directory
PROJECT_DIR="/project/ptgroup/Devin/Neural_Network"

echo "Starting CSV merge job at \$(date)"
echo "Output file will be: \${OUTPUT_DIR}/\${OUTPUT_FILENAME}"

# Create the merge script file
cat > merge.py << PYEOF
import pandas as pd
import glob
import os
import sys

# Get output directory and filename from command line arguments
output_dir = sys.argv[1]
output_filename = sys.argv[2]

os.chdir(output_dir)
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

print(f"Found {len(all_filenames)} CSV files to combine")

# Load and combine all CSV files
print("Loading and combining files...")
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])

# Save the combined data
print(f"Saving combined data to {output_filename}...")
combined_csv.to_csv(output_filename, index=False)

print(f"Successfully combined {len(all_filenames)} files into {output_filename}")
print(f"Total samples: {len(combined_csv):,}")
PYEOF

# Run the merger script with parameters
echo "Running merge script..."
python merge.py "\${OUTPUT_DIR}" "\${OUTPUT_FILENAME}"

# Check if merge was successful
if [ \$? -ne 0 ]; then
    echo "Error: CSV merge failed. Exiting."
    exit 1
fi

echo "CSV merge completed successfully at \$(date)"
echo "Output file: \${OUTPUT_DIR}/\${OUTPUT_FILENAME}"

# Print file size
echo "File size: \$(du -h \${OUTPUT_DIR}/\${OUTPUT_FILENAME} | cut -f1)"

echo "Job complete."
EOF

    # Make the merge script executable
    chmod +x ${MERGE_SCRIPT}
    
    # Submit the merge job with a dependency on the entire array job
    sbatch --dependency=afterok:${SLURM_ARRAY_JOB_ID} ${MERGE_SCRIPT}
else
    echo "This is not the last array job. Skipping merge step."
fi

echo "Job ${SLURM_JOB_ID}, task ${SLURM_ARRAY_TASK_ID} completed at $(date)"
exit 0