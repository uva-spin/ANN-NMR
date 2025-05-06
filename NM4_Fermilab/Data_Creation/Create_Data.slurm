#!/usr/bin/env bash
#SBATCH -p standard
#SBATCH --job-name=data_generation
#SBATCH --output=job_logs/job_%A_%a.out  # Redirect output to job_logs subdirectory
#SBATCH --error=job_logs/job_%A_%a.err   # Redirect error to job_logs subdirectory
#SBATCH -c 16
#SBATCH -t 1:00:00
#SBATCH -A spinquest_standard
#SBATCH --array=1-2     # Generate SLURM_ARRAY_TASK_MAX sets of data in parallel

# ===== Global Configuration =====
# Set these variables to configure the job
PROJECT_DIR="..."  ### Change this to your own directory
OUTPUT_DIR="${PROJECT_DIR}/Test"
OUTPUT_FILENAME="Deuteron_1_60_With_Noise_500K.parquet"  # Global variable for output filename
MODE="deuteron"                       # Options: deuteron, proton
NUM_SAMPLES=10                      # Number of samples per task ### TOTAL NUMBER OF SAMPLES = NUM_SAMPLES * SLURM_ARRAY_TASK_MAX
ADD_NOISE=1                           # 0=no noise, 1=add noise

OVERSAMPLING=0                        # 0=no oversampling, 1=enable oversampling
OVERSAMPLED_VALUE=0.0005              # Value to oversample around
OVERSAMPLING_UPPER_BOUND=0.0006       # Upper bound for oversampling range
OVERSAMPLING_LOWER_BOUND=0.0004       # Lower bound for oversampling range
UPPER_BOUND=0.6                       # Upper bound for P value (not oversampled)
LOWER_BOUND=0.01                       # Lower bound for P value (not oversampled)
P_MAX=0.6                             # Maximum polarization value
ALPHA=0.3                             # Decay rate for power law distribution
BASELINE=1                         # 0=no baseline, 1=add baseline
NOISE_LEVEL=0.00002                  # Standard deviation of Gaussian noise
SHIFTING=0                         # 0=no shifting, 1=enable shifting
BOUND=0.08                            # Bound for shifting

# Create job_logs directory if it doesn't exist
mkdir -p job_logs

echo "Starting job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Configuration:"
echo "  - Output directory: ${OUTPUT_DIR}"
echo "  - Output filename: ${OUTPUT_FILENAME}"
echo "  - Mode: ${MODE}"
echo "  - Samples per job: ${NUM_SAMPLES}"
echo "  - Add noise: ${ADD_NOISE}"
echo "  - Oversampling: ${OVERSAMPLING}"
echo "  - Oversampled value: ${OVERSAMPLED_VALUE}"
echo "  - Oversampling upper bound: ${OVERSAMPLING_UPPER_BOUND}"
echo "  - Oversampling lower bound: ${OVERSAMPLING_LOWER_BOUND}"
echo "  - Upper bound: ${UPPER_BOUND}"
echo "  - Lower bound: ${LOWER_BOUND}"
echo "  - P_max: ${P_MAX}"
echo "  - Alpha: ${ALPHA}"
echo "  - Baseline: ${BASELINE}"
echo "  - Noise level: ${NOISE_LEVEL}"
echo "  - Shifting: ${SHIFTING}"
echo "  - Bound: ${BOUND}"

module purge
echo "Modules purged"
module spider miniforge/24.3.0-py3.11
echo "Miniforge module spidered"
module load apptainer tensorflow/2.17.0
echo "TensorFlow module loaded"

mkdir -p ${OUTPUT_DIR}
chmod 755 ${OUTPUT_DIR}
echo "Output directory created: ${OUTPUT_DIR}"


# ===== Part 1: Generate Signal Data =====
echo "Generating signal data..."

# Create a Python wrapper script to handle the additional parameters
cat > run_signal_generator.py << EOF
import sys
import os
import subprocess

# Get parameters from command line
job_id = sys.argv[1]
mode = sys.argv[2]
num_samples = sys.argv[3]
add_noise = sys.argv[4]
oversampling = sys.argv[5]
oversampled_value = sys.argv[6]
oversampling_upper_bound = sys.argv[7]
oversampling_lower_bound = sys.argv[8]
upper_bound = sys.argv[9]
lower_bound = sys.argv[10]
p_max = sys.argv[11]
alpha = sys.argv[12]
baseline = sys.argv[13]
noise_level = sys.argv[14]
shifting = sys.argv[15]
bound = sys.argv[16]

output_dir = "${OUTPUT_DIR}"
print(f"Output directory: {output_dir}")
print(f"Current working directory: {os.getcwd()}")

# Construct the command to run the signal generator
cmd = [
    "python", "Create_Training_Data.py",
    "--job_id", job_id,
    "--mode", mode,
    "--num_samples", num_samples,
    "--add_noise", add_noise,
    "--oversampling", oversampling,
    "--oversampled_value", oversampled_value,
    "--oversampling_upper_bound", oversampling_upper_bound,
    "--oversampling_lower_bound", oversampling_lower_bound,
    "--upper_bound", upper_bound,
    "--lower_bound", lower_bound,
    "--p_max", p_max,
    "--alpha", alpha,
    "--baseline", baseline,
    "--noise_level", noise_level,
    "--shifting", shifting,
    "--bound", bound,
    "--output_dir", output_dir
]

# Print the command for debugging
print(f"Running command: {' '.join(cmd)}")

# Run the command
result = subprocess.run(cmd)
sys.exit(result.returncode)
EOF

# Run the wrapper script with all parameters
echo "Running signal generation script using apptainer..."
apptainer run --nv $CONTAINERDIR/tensorflow-2.17.0.sif run_signal_generator.py \
    ${SLURM_ARRAY_TASK_ID} \
    ${MODE} \
    ${NUM_SAMPLES} \
    ${ADD_NOISE} \
    ${OVERSAMPLING} \
    ${OVERSAMPLED_VALUE} \
    ${OVERSAMPLING_UPPER_BOUND} \
    ${OVERSAMPLING_LOWER_BOUND} \
    ${UPPER_BOUND} \
    ${LOWER_BOUND} \
    ${P_MAX} \
    ${ALPHA} \
    ${BASELINE} \
    ${NOISE_LEVEL} \
    ${SHIFTING} \
    ${BOUND}

# Check if generation was successful
if [ $? -ne 0 ]; then
    echo "Error: Signal generation failed. Exiting."
    exit 1
fi

echo "Signal generation complete for task ${SLURM_ARRAY_TASK_ID}"

# ===== Part 2: Create the merge.py script directly =====
echo "Creating the merge.py script..."

cat > merge.py << 'EOF'
import pandas as pd
import glob
import os
import sys

# Get output directory and filename from command line arguments
output_dir = sys.argv[1]
output_filename = sys.argv[2]

print(f"Merging files in {output_dir} to {output_filename}")

os.chdir(output_dir)
print(f"Changed working directory to {os.getcwd()}")

# Print directory contents for debugging
print(f"Contents of {output_dir}:")
for f in glob.glob('*'):
    print(f"  {f}")

extension = 'parquet'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
print(f"Found {len(all_filenames)} Parquet files to combine")

if not all_filenames:
    print("ERROR: No Parquet files found. Exiting.")
    sys.exit(1)

# Load and combine all Parquet files
print("Loading and combining files...")
combined_csv = pd.concat([pd.read_parquet(f) for f in all_filenames])

# Save the combined data
print(f"Saving combined data to {output_filename}...")
combined_csv.to_parquet(output_filename, engine='pyarrow', compression='fastparquet')

print(f"Successfully combined {len(all_filenames)} files into {output_filename}")
print(f"Total samples: {len(combined_csv):,}")
EOF

# ===== Part 3: Combine Parquet Files (only run by the last array job) =====
# We'll use a job dependency to trigger this after all array jobs finish
echo "Checking if this is the last array job..."
if [ "${SLURM_ARRAY_TASK_ID}" -eq "${SLURM_ARRAY_TASK_MAX}" ]; then
    echo "This is the last array job. Submitting the merge job."
    
    # Create the merge job script with the OUTPUT_FILENAME variable passed through
    MERGE_SCRIPT="merge_job.sh"
    cat > ${MERGE_SCRIPT} << EOF
#!/usr/bin/env bash
#SBATCH -p standard
#SBATCH --job-name=merge_parquet
#SBATCH --output=job_logs/merge_%j.out  # Redirect output to job_logs subdirectory
#SBATCH --error=job_logs/merge_%j.err   # Redirect error to job_logs subdirectory
#SBATCH -c 16
#SBATCH -t 2:00:00
#SBATCH -A spinquest_standard
#SBATCH --mem=32G          # Large memory for merging 1000 files

# Global variables passed from parent script
OUTPUT_DIR="${OUTPUT_DIR}"
OUTPUT_FILENAME="${OUTPUT_FILENAME}"

module purge
module load miniforge/24.3.0-py3.11
module load apptainer tensorflow/2.17.0

# Run the merge script inside apptainer container
apptainer run --nv $CONTAINERDIR/tensorflow-2.17.0.sif merge.py \
    ${OUTPUT_DIR} ${OUTPUT_FILENAME}
EOF

    sbatch ${MERGE_SCRIPT}
    echo "Merge job submitted."
else
    echo "This is not the last array job. Skipping merge step."
fi

echo "Job ${SLURM_JOB_ID} complete at $(date)"