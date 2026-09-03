<p align="center">
  <img src="docs/assets/logo.svg" alt="jgm" width="220">
</p>

<h1 align="center">jobgpumonitor</h1>

<p align="center">
  <b>Your GPU job tells you what it is doing. You stop ssh-ing in to check.</b><br>
  One import &nbsp;·&nbsp; zero dependencies &nbsp;·&nbsp; Slurm, OAR, PBS, LSF or none
</p>

<p align="center">
  <a href="https://github.com/marcpinet/jobgpumonitor/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/marcpinet/jobgpumonitor/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://pypi.org/project/jobgpumonitor/"><img alt="PyPI" src="https://img.shields.io/pypi/v/jobgpumonitor?color=blue"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue">
  <img alt="Dependencies" src="https://img.shields.io/badge/dependencies-0-success">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey"></a>
</p>

---

Your job waited four hours in the queue, ran for two, and died at 3 a.m.
You found out at 9, from a `.err` file.

`jobgpumonitor` makes the job report on itself while it runs: where it is, how fast it goes,
whether the GPU is actually busy, how close it is to the memory limit, what the traceback was,
and how it ended. Everything is streamed as plain JSONL events to your shared filesystem,
so a notifier, a dashboard or a one-line `grep` can read it. No network from the compute node.

## 30-second start

```bash
pip install jobgpumonitor
```

```python
import jobgpumonitor.auto
```

That is the whole integration. No code at all? Wrap the command instead:

```bash
jgm run -- python train.py
```

Events land in `~/.jobgpumonitor/runs/<cluster>/<job>/<restart>/`. Run `jgm doctor` on a node
to see what gets detected.

## What the job tells you

| Event | The question it answers |
|---|---|
| `run.start` | Which commit, which GPUs, which node, and *when the scheduler will kill it* |
| `progress.update` | tqdm ETA compared to the job deadline, hours before a time-out |
| `resource.sample` | Is the GPU idle? How close to the cgroup memory limit, before the OOM kill |
| `run.exception` | The full traceback, secrets masked, OOM flagged |
| `run.end` | Status, exit code, GPU utilisation and peak memory summary |
| `run.heartbeat` | Still alive, latest metrics, latest progress |

Add your own with `jobgpumonitor.log(loss=0.12, step=10)` or `jgm emit stage name=eval` from bash.

## Designed for real clusters

- **No network needed from the job.** Compute nodes rarely have internet; the shared filesystem always works. `jgm forward` on the login node ships the files to a server over HTTPS, through the site proxy, and resumes where it left off.
- **Never touches your program.** Hooks are chained, not replaced. Nothing raises. Forked workers and DDP ranks other than 0 stay quiet. Writes happen on a background thread, so a slow NFS never stalls a training step.
- **Works inside containers.** Reads environment variables and `/proc` only, no `squeue` required. Point `JGM_DIR` at a mounted path and you are done.
- **Honest about what it cannot see.** A SIGKILL leaves no trace from inside. Run `jgm scheduler` on the login node: it watches `squeue`/`sacct` and writes the scheduler's verdict (OOM, time-out, preemption, cancel), the queue state and the real `.out` path into the same run directories.

## Slurm in one screen

```bash
#!/bin/bash
#SBATCH --gpus-per-node=1 --time=04:00:00
export PYTHONUNBUFFERED=1
srun jgm run -- python train.py
```

Inside an enroot or Apptainer container, mount the event directory and export `JGM_DIR`:

```bash
mkdir -p "$HOME/.jobgpumonitor" && export JGM_DIR=/jgm
srun --container-mounts="$HOME/.jobgpumonitor:/jgm" jgm run -- python train.py
```

## Command line

`jgm` ships with the package (short for **j**ob**g**pu**m**onitor); `python -m jobgpumonitor.cli` is equivalent.

```
jgm run [--name N] -- CMD...   run a command under monitoring, forward signals, keep the stderr tail
jgm scheduler [--once]         login-node probe: squeue/sacct (or oarstat) -> scheduler.state events
jgm forward --url U --token T  login-node relay: ship the event files to a jobgpumonitor-server (POST /ingest)
jgm emit TYPE k=v ...          emit one event from a shell script
jgm doctor [--json]            show what is detected on this node
jgm ls [--dir D]               list runs found in the event directory
```

Keep the probe alive on the login node with `tmux` or `systemd --user`, or run
`jgm scheduler --once` from cron.

## Configuration

Every knob is a `JGM_*` environment variable, all optional.
`JGM_DIR`, `JGM_SINKS`, `JGM_HEARTBEAT_S`, `JGM_SAMPLE_S`, `JGM_RANK_MODE`, `JGM_LOGGING_LEVEL`,
`JGM_CAPTURE_LOCALS`, `JGM_DISABLED`. The full list is in [docs/CONFIG.md](docs/CONFIG.md).

`pip install "jobgpumonitor[gpu]"` adds `nvidia-ml-py` and `psutil` for cheaper, richer samples.
Without them the same data comes from `nvidia-smi` and `/proc`.

## Ecosystem

This package only **emits**. Storage, API, notifications and dashboards live in a separate
server that consumes the [event protocol](docs/PROTOCOL.md) (JSON Schema in
[`schema/`](schema/)). Write your own consumer in an afternoon, or wait for ours.

## Contributing

```bash
pip install -e ".[dev]" && pytest
```

Design notes in [docs/DESIGN.md](docs/DESIGN.md). Issues and pull requests welcome, especially
reports from clusters and schedulers we have not seen.
