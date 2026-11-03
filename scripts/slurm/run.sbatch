#!/bin/bash -l
# Run one of our Python scripts on an Alex compute node.
#
#   sbatch scripts/slurm/run.sbatch scripts/dead_ball/dead_check.py [args...]
#   sbatch scripts/slurm/run.sbatch scripts/make_submission.py baseline50_dead
#
# Submit from the repository root: the job changes to the folder it was submitted from, and the
# log path below is relative to it (logs/ must exist: Slurm does not create it). Everything
# after the script name is passed to the script.
# Heavy runs go through here because the login node is shared and slow for large arrays; on a
# compute node the features take 1-2 s. Jobs usually start within a minute, at worst ~10.
#
# Output goes to logs/<jobname>-<jobid>.out (gitignored), e.g. logs/mtah-123456.out. Its first
# line is "node <hostname>, <date>, commit <short hash>", with "+dirty" appended when tracked
# files have unstaged changes, so every result can be traced to the code that produced it; the
# rest is the script's own output. One A40 comes with 16 cores; the GPU is not used by the numpy
# steps, but Alex only hands out nodes with GPUs. The time limit is 1 h.
#SBATCH --job-name=mtah
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --export=NONE

# The job starts with a clean environment (--export=NONE); unset this so srun steps inherit it.
unset SLURM_EXPORT_ENV
set -euo pipefail

# Work in the submit folder with the project's conda environment, used by absolute path (no
# conda activate needed). Python warnings are silenced with -W ignore; numpy uses every core.
cd "$SLURM_SUBMIT_DIR"
PYTHON=/home/atuin/b180dc/b180dc59/software/private/conda/envs/mtah2027/bin/python
export XDG_CACHE_HOME=$WORK/.cache
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

echo "node $(hostname), $(date), commit $(git rev-parse --short HEAD)$(git diff --quiet || echo '+dirty')"
"$PYTHON" -W ignore "$@"
