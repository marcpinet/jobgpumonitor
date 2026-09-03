#!/bin/bash
# Smoke test for jobgpumonitor on a Slurm + enroot/pyxis cluster. ~1 minute, 1 GPU.
#
#   git clone https://github.com/marcpinet/jobgpumonitor ~/jobgpumonitor
#   sbatch ~/jobgpumonitor/examples/slurm_smoke.sbatch
#
#SBATCH --job-name=jgm-smoke
#SBATCH --partition=short
#SBATCH --qos=normal
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

CONTAINER_IMAGE="${CONTAINER_IMAGE:-dockerproxy.repos.tech.orange/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime}"
SRC="${JGM_SRC:-$HOME/jobgpumonitor}"
MOUNTS="$HOME/.jobgpumonitor:/jgm,$SRC:/jgm-src"

mkdir -p "$HOME/.jobgpumonitor"
echo "job=$SLURM_JOB_ID node=$SLURM_NODELIST end_time=${SLURM_JOB_END_TIME:-unset} home=$HOME src=$SRC"

echo "=== control step: hostname inside the container ==="
srun --container="$CONTAINER_IMAGE" --container-mounts="$MOUNTS" hostname
echo "control step exit code: $?"

echo "=== smoke test ==="
srun --container="$CONTAINER_IMAGE" --container-mounts="$MOUNTS" bash /jgm-src/examples/smoke_inner.sh
echo "srun exit code: $?"

echo "=== events written ==="
find "$HOME/.jobgpumonitor/runs" -name "*.jsonl" -exec wc -l {} + 2>/dev/null || echo none
