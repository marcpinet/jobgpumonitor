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

mkdir -p "$HOME/.jobgpumonitor"
export JGM_DIR=/jgm            # path of the event directory as seen from inside the container
export JGM_DEBUG=1             # internal diagnostics on stderr, useful for a first run
export PYTHONUNBUFFERED=1
export PYTHONPATH=/jgm-src/src # pure-Python, zero dependency: no pip install needed

echo "job=$SLURM_JOB_ID node=$SLURM_NODELIST end_time=${SLURM_JOB_END_TIME:-unset} home=$HOME src=$SRC"
set -x

srun --container="$CONTAINER_IMAGE" \
     --container-mounts="$HOME/.jobgpumonitor:/jgm,$SRC:/jgm-src" \
     --container-workdir=/jgm-src \
     bash -c '
        exec > >(tee -a /jgm/smoke_$SLURM_JOB_ID.log) 2>&1
        echo "=== inside container: $(hostname) $(date) ==="
        echo "=== doctor (inside container) ==="
        python -m jobgpumonitor.cli doctor
        echo "=== wrapped run, clean exit ==="
        python -m jobgpumonitor.cli run -- python examples/gpu_smoke.py --seconds 40
        echo "=== wrapped run, intentional crash ==="
        python -m jobgpumonitor.cli run -- python examples/gpu_smoke.py --seconds 5 --crash
        echo "wrapper exit code for the crash run: $?"
     '
echo "srun exit code: $?"
set +x
echo "events written:"; find "$HOME/.jobgpumonitor/runs" -name "*.jsonl" -exec wc -l {} +
