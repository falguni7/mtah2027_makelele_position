#!/bin/bash -l
# Run one of our Python scripts on an Alex compute node.
#
#   sbatch scripts/slurm/run.sbatch scripts/dead_ball/dead_check.py [args...]
#
# Output goes to logs/<jobname>-<jobid>.out (gitignored). One A40 comes with 16 cores; the GPU
# is not used by the numpy steps, but Alex only hands out nodes with GPUs.
#SBATCH --job-name=mtah
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
PYTHON=/home/atuin/b180dc/b180dc59/software/private/conda/envs/mtah2027/bin/python
export XDG_CACHE_HOME=$WORK/.cache
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

echo "node $(hostname), $(date), commit $(git rev-parse --short HEAD)$(git diff --quiet || echo '+dirty')"
"$PYTHON" -W ignore "$@"
