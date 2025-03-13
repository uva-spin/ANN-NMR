#!/usr/bin/env bash
#SBATCH -p standard
#SBATCH --job-name=signal_processing
#SBATCH --output=job_%A_%a.out
#SBATCH --error=job_%A_%a.err
#SBATCH -c 16
#SBATCH -t 12:00:00
#SBATCH -A spinquest_standard
#SBATCH --array=1-1000     # Generate an N number of instances of the script. The outputted amount of events will be N * NUM_SAMPLES
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@example.edu

# Print job details for logging
echo "Starting job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"

# Load required modules
module purge
module load miniforge/24.3.0-py3.11

# Set working directory and create output directories
PROJECT_DIR="/project/ptgroup/Devin/NMR_Final/Testing" ### Change this to your own directory
SCRIPT_DIR="${PROJECT_DIR}/scripts"
OUTPUT_DIR="${PROJECT_DIR}/Training_Data_$(date +%Y%m%d)"

mkdir -p ${OUTPUT_DIR}
cd ${SCRIPT_DIR}

# ===== Part 1: Generate Signal Data =====
echo "Generating signal data..."

# Configure sample generation parameters
MODE="deuteron"        # Options: deuteron, proton
NUM_SAMPLES=1000       # Number of samples per task (reduced from 10000 to prevent overloading)
ADD_NOISE=1            # 0=no noise, 1=add noise

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
    
    # Create the merge job script
    MERGE_SCRIPT="${SCRIPT_DIR}/merge_job.sh"
    cat > ${MERGE_SCRIPT} << 'EOF'
#!/usr/bin/env bash
#SBATCH -p standard
#SBATCH --job-name=merge_csv
#SBATCH --output=merge_%j.out
#SBATCH --error=merge_%j.err
#SBATCH -c 16
#SBATCH -t 12:00:00        # Increased time for merging 1000 files
#SBATCH -A spinquest_standard
#SBATCH --mem=64G          # Increased memory for large merge operation

# Load required modules
module purge
module load miniforge/24.3.0-py3.11

# Set working directory
PROJECT_DIR="/project/ptgroup/Devin/Neural_Network"
SCRIPT_DIR="${PROJECT_DIR}/scripts"
OUTPUT_DIR="${PROJECT_DIR}/Training_Data_$(date +%Y%m%d)"

cd ${SCRIPT_DIR}

echo "Starting CSV merge job at $(date)"

# Run the CSV combiner with smaller chunk size for memory efficiency
python csv_combiner.py ${OUTPUT_DIR} "Combined_Dataset_1M.csv" "Sample_*.csv"

# Check if merge was successful
if [ $? -ne 0 ]; then
    echo "Error: CSV merge failed. Exiting."
    exit 1
fi

echo "CSV merge completed successfully at $(date)"

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