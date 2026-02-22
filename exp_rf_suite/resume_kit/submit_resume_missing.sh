#!/bin/bash
#SBATCH -J rf_resume
#SBATCH -p main
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -o rf_resume_%A_%a.out
#SBATCH -e rf_resume_%A_%a.err

set -euo pipefail
umask 077

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

source /scratch/lustre/home/$USER/.venv_rf_suite/bin/activate

# Stable joblib temp folder (avoid /dev/shm memmapping issues)
JOBLIB_TMP=/scratch/lustre/home/$USER/exp_rf_suite/joblib_tmp_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$JOBLIB_TMP"
export JOBLIB_TEMP_FOLDER="$JOBLIB_TMP"
export TMPDIR="$JOBLIB_TMP"

CFG=/scratch/lustre/home/$USER/config_rf_suite.yaml
SCRIPT=/scratch/lustre/home/$USER/run_rf_suite.py
MISS=/scratch/lustre/home/$USER/exp_rf_suite/resume_kit/missing_tasks.txt

TASK_ID="$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$MISS")"
echo "Running missing task_id=$TASK_ID"
python -u "$SCRIPT" --config "$CFG" --fold 0 --task-id "$TASK_ID"
