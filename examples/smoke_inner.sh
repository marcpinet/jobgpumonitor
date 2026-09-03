#!/bin/bash
# Runs inside the container (see slurm_smoke.sbatch). Repo is mounted at /jgm-src, events at /jgm.
export PYTHONPATH=/jgm-src/src JGM_DIR=/jgm JGM_DEBUG=1 PYTHONUNBUFFERED=1
cd /jgm-src || exit 1
echo "=== inside container: $(hostname) $(date) python=$(command -v python) ==="
echo "=== scheduler env ==="
env | grep -E "^(SLURM_(JOB|ARRAY|RESTART|CLUSTER|NTASKS|PROCID|MEM|CPUS|GPUS|STEP)|RANK|WORLD_SIZE|LOCAL_RANK|CUDA_VISIBLE|NVIDIA_)" | sort
echo "=== doctor ==="
python -m jobgpumonitor.cli doctor
echo "=== wrapped run, clean exit (40 s) ==="
python -m jobgpumonitor.cli run -- python examples/gpu_smoke.py --seconds 40
echo "wrapper exit code: $?"
echo "=== wrapped run, intentional crash (5 s) ==="
python -m jobgpumonitor.cli run -- python examples/gpu_smoke.py --seconds 5 --crash
echo "wrapper exit code for the crash run: $?"
