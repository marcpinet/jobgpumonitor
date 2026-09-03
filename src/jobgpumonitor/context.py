"""Where am I running? Scheduler, job identity, rank, container, git, environment.

Everything here reads environment variables and the local filesystem only. No scheduler
command is ever executed: inside a container ``squeue`` does not exist, and on a login node
we must not hammer the controller.
"""

from __future__ import annotations

import os
import platform
import re
import socket
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from ._log import dbg

SECRET_RE = re.compile(
    r"(KEY|TOKEN|SECRET|PASS(WORD|WD)?|CREDENTIAL|AUTH|PRIVATE|COOKIE|SESSION|SIGNATURE|DSN)", re.I
)

ENV_PREFIXES: Tuple[str, ...] = (
    "SLURM_", "SLURMD_", "OAR_", "PBS_", "LSB_", "LSF_", "SGE_", "JOB_ID",
    "CUDA_", "NVIDIA_", "NCCL_", "ROCR_", "HIP_", "OMP_", "MKL_", "OPENBLAS_", "NUMEXPR_",
    "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "NODE_RANK",
    "MASTER_ADDR", "MASTER_PORT", "TORCHELASTIC_", "TORCH_", "PYTORCH_",
    "PYTHON", "VIRTUAL_ENV", "CONDA_", "UV_", "PIP_",
    "HF_", "TRANSFORMERS_", "WANDB_", "MLFLOW_", "NEPTUNE_",
    "JGM_", "ENROOT_", "PYXIS_", "SINGULARITY_", "APPTAINER_", "CONTAINER",
    "HOSTNAME", "USER", "LOGNAME", "HOME", "PWD", "SHELL", "TMPDIR", "SCRATCH",
)

_PACKAGE_IMPORT_NAMES = {
    "scikit-learn": "sklearn",
    "pytorch-lightning": "pytorch_lightning",
}


def _int(v: Optional[str]) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except ValueError:
        return None


def _float(v: Optional[str]) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _hms_to_seconds(v: Optional[str]) -> Optional[int]:
    """Parse ``D-HH:MM:SS``, ``HH:MM:SS``, ``MM:SS`` or ``MM`` (Slurm/OAR style)."""
    if not v:
        return None
    try:
        days = 0
        if "-" in v:
            d, v = v.split("-", 1)
            days = int(d)
        parts = [int(p) for p in v.split(":")]
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h, (m, s) = 0, parts
        elif len(parts) == 1:
            h, m, s = 0, parts[0], 0
        else:
            return None
        return days * 86400 + h * 3600 + m * 60 + s
    except ValueError:
        return None


# --------------------------------------------------------------------------- scheduler


def detect_scheduler(env: Mapping[str, str]) -> Dict[str, Any]:
    """Return a dict describing the batch job we run in (``name == "local"`` if none)."""
    job: Dict[str, Any] = {
        "name": "local",
        "job_id": None,
        "job_name": None,
        "array_job_id": None,
        "array_task_id": None,
        "restart_count": 0,
        "cluster": None,
        "partition": None,
        "nodes": None,
        "node_count": None,
        "ntasks": None,
        "cpus_per_task": None,
        "gpus": None,
        "mem": None,
        "start_ts": None,
        "end_ts": None,
        "time_limit_s": None,
        "step_id": None,
        "interactive_step": False,
    }
    if env.get("SLURM_JOB_ID") or env.get("SLURM_JOBID"):
        job.update(
            name="slurm",
            job_id=env.get("SLURM_JOB_ID") or env.get("SLURM_JOBID"),
            job_name=env.get("SLURM_JOB_NAME"),
            array_job_id=env.get("SLURM_ARRAY_JOB_ID"),
            array_task_id=env.get("SLURM_ARRAY_TASK_ID"),
            restart_count=_int(env.get("SLURM_RESTART_COUNT")) or 0,
            cluster=env.get("SLURM_CLUSTER_NAME"),
            partition=env.get("SLURM_JOB_PARTITION"),
            nodes=env.get("SLURM_JOB_NODELIST") or env.get("SLURM_NODELIST"),
            node_count=_int(env.get("SLURM_JOB_NUM_NODES") or env.get("SLURM_NNODES")),
            ntasks=_int(env.get("SLURM_NTASKS")),
            cpus_per_task=_int(env.get("SLURM_CPUS_PER_TASK")),
            gpus=env.get("SLURM_JOB_GPUS") or env.get("SLURM_GPUS_ON_NODE") or env.get("SLURM_GPUS"),
            mem=env.get("SLURM_MEM_PER_NODE") or env.get("SLURM_MEM_PER_CPU") or env.get("SLURM_MEM_PER_GPU"),
            start_ts=_float(env.get("SLURM_JOB_START_TIME")),
            end_ts=_float(env.get("SLURM_JOB_END_TIME")),
            step_id=env.get("SLURM_STEP_ID"),
            interactive_step=bool(env.get("SLURM_PTY_PORT")),
        )
        if job["start_ts"] and job["end_ts"]:
            job["time_limit_s"] = int(job["end_ts"] - job["start_ts"])
    elif env.get("OAR_JOB_ID"):
        wall = _int(env.get("OAR_JOB_WALLTIME_SECONDS")) or _hms_to_seconds(env.get("OAR_JOB_WALLTIME"))
        job.update(
            name="oar",
            job_id=env.get("OAR_JOB_ID"),
            job_name=env.get("OAR_JOB_NAME"),
            array_job_id=env.get("OAR_ARRAY_ID") if env.get("OAR_ARRAY_INDEX") else None,
            array_task_id=env.get("OAR_ARRAY_INDEX"),
            partition=env.get("OAR_QUEUE"),
            nodes=env.get("OAR_NODEFILE"),
            time_limit_s=wall,
            interactive_step=env.get("OAR_JOB_TYPE") == "interactive" or "OAR_INTERACTIVE" in env,
        )
    elif env.get("PBS_JOBID"):
        job.update(
            name="pbs",
            job_id=env.get("PBS_JOBID"),
            job_name=env.get("PBS_JOBNAME"),
            array_task_id=env.get("PBS_ARRAY_INDEX") or env.get("PBS_ARRAYID"),
            partition=env.get("PBS_QUEUE"),
            nodes=env.get("PBS_NODEFILE"),
            interactive_step=env.get("PBS_ENVIRONMENT") == "PBS_INTERACTIVE",
        )
        if job["array_task_id"]:
            job["array_job_id"] = re.sub(r"\[\d*\]", "[]", job["job_id"] or "")
    elif env.get("LSB_JOBID"):
        job.update(
            name="lsf",
            job_id=env.get("LSB_JOBID"),
            job_name=env.get("LSB_JOBNAME"),
            array_job_id=env.get("LSB_JOBID") if _int(env.get("LSB_JOBINDEX")) else None,
            array_task_id=env.get("LSB_JOBINDEX") if _int(env.get("LSB_JOBINDEX")) else None,
            partition=env.get("LSB_QUEUE"),
            nodes=env.get("LSB_HOSTS"),
        )
    elif env.get("JOB_ID") and env.get("SGE_O_WORKDIR"):
        job.update(
            name="sge",
            job_id=env.get("JOB_ID"),
            job_name=env.get("JOB_NAME"),
            array_job_id=env.get("JOB_ID") if env.get("SGE_TASK_ID") not in (None, "undefined") else None,
            array_task_id=env.get("SGE_TASK_ID") if env.get("SGE_TASK_ID") != "undefined" else None,
            partition=env.get("QUEUE"),
        )
    return job


def parse_mem_bytes(value: Optional[str]) -> Optional[int]:
    """Slurm/PBS memory strings: ``8192`` (MB for Slurm), ``8G``, ``512M``, ``1T``, ``8gb``."""
    if not value:
        return None
    v = str(value).strip().lower().rstrip("b")
    mult = {"k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}
    try:
        if v and v[-1] in mult:
            return int(float(v[:-1]) * mult[v[-1]])
        return int(float(v)) * 1024**2  # Slurm reports plain numbers in MB
    except ValueError:
        return None


def job_key(job: Mapping[str, Any]) -> Optional[str]:
    """Stable per-task identifier: ``<array_job>_<task>`` for array tasks, else the job id."""
    if job.get("array_job_id") and job.get("array_task_id") is not None:
        return f"{job['array_job_id']}_{job['array_task_id']}"
    return job.get("job_id")


# --------------------------------------------------------------------------- rank / container


def detect_rank(env: Mapping[str, str]) -> Optional[Dict[str, Any]]:
    """Distributed rank from torchrun, Slurm, OpenMPI or PMI; ``None`` when not distributed."""
    pairs = (
        ("RANK", "WORLD_SIZE", "LOCAL_RANK", "torchrun"),
        ("SLURM_PROCID", "SLURM_NTASKS", "SLURM_LOCALID", "slurm"),
        ("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_LOCAL_RANK", "openmpi"),
        ("PMI_RANK", "PMI_SIZE", None, "pmi"),
        ("PMIX_RANK", None, None, "pmix"),
    )
    for rk, ws, lr, launcher in pairs:
        r = _int(env.get(rk))
        if r is None:
            continue
        w = _int(env.get(ws)) if ws else None
        if w is None and launcher in ("pmi", "pmix"):
            # Slurm's PMIx plugin exports PMIX_RANK for every step, even single-task ones,
            # without a size: fall back to the step's task count.
            w = _int(env.get("PMIX_SIZE") or env.get("SLURM_STEP_NUM_TASKS") or env.get("SLURM_NTASKS"))
        if w is not None and w <= 1:
            # RANK=0/WORLD_SIZE=1 (some images export it) or a plain `srun python x.py`
            # with SLURM_NTASKS=1: a single task is not a distributed run.
            continue
        if launcher == "slurm" and w is None:
            continue
        return {
            "rank": r,
            "world_size": w,
            "local_rank": _int(env.get(lr)) if lr else None,
            "launcher": launcher,
        }
    return None


def detect_container(env: Mapping[str, str]) -> Optional[str]:
    if env.get("APPTAINER_CONTAINER") or env.get("APPTAINER_NAME"):
        return "apptainer"
    if env.get("SINGULARITY_CONTAINER") or env.get("SINGULARITY_NAME"):
        return "singularity"
    if any(k.startswith(("ENROOT_", "PYXIS_")) for k in env):
        return "enroot"
    if os.path.exists("/usr/sbin/ldconfig.host") or os.path.exists("/etc/enroot"):
        return "enroot"  # enroot bind-mounts the host ldconfig into every container
    if os.path.exists("/run/.containerenv"):
        return "podman"
    if os.path.exists("/.dockerenv"):
        return "docker"
    try:
        with open("/proc/1/cgroup", encoding="utf-8", errors="replace") as f:
            txt = f.read()
        if "docker" in txt or "containerd" in txt:
            return "docker"
        if "kubepods" in txt:
            return "kubernetes"
    except OSError:
        pass
    try:
        with open("/proc/self/mountinfo", encoding="utf-8", errors="replace") as f:
            head = f.read(4096)
        if " / / " in head and ("overlay" in head or "fuse.squashfuse" in head):
            return "unknown"
    except OSError:
        pass
    return None


def mount_points(limit: int = 64) -> List[Dict[str, str]]:
    """Mount table (mount point + fs type), so a consumer can translate container paths."""
    out: List[Dict[str, str]] = []
    try:
        with open("/proc/self/mounts", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mp, fstype = parts[1], parts[2]
                if mp.startswith(("/proc", "/sys", "/dev", "/run")) or fstype in ("cgroup", "cgroup2", "tmpfs", "devpts", "mqueue"):
                    continue
                out.append({"src": parts[0][:200], "path": mp, "type": fstype})
                if len(out) >= limit:
                    break
    except OSError:
        pass
    return out


# --------------------------------------------------------------------------- git / packages / env


def git_info(cwd: str, timeout: float = 2.0) -> Optional[Dict[str, Any]]:
    root = None
    d = os.path.abspath(cwd)
    for _ in range(12):
        if os.path.exists(os.path.join(d, ".git")):
            root = d
            break
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    if root is None:
        return None
    info: Dict[str, Any] = {"root": root}

    def run(*args: str) -> Optional[str]:
        try:
            p = subprocess.run(
                ["git", "-C", root, *args], capture_output=True, text=True, timeout=timeout,
                env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
            )
            return p.stdout.strip() if p.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    info["commit"] = run("rev-parse", "--short=12", "HEAD")
    if info["commit"] is None:
        return info
    info["branch"] = run("rev-parse", "--abbrev-ref", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    info["dirty"] = bool(status) if status is not None else None
    remote = run("remote", "get-url", "origin")
    if remote:
        info["remote"] = re.sub(r"//[^@/]+@", "//", remote)  # strip embedded credentials
    return info


def package_versions(names: Iterable[str]) -> Dict[str, str]:
    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover
        return {}
    out: Dict[str, str] = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except Exception:
            continue
    return out


def filtered_env(
    env: Mapping[str, str],
    extra_prefixes: Iterable[str] = (),
    extra_secret: Iterable[str] = (),
    max_value: int = 512,
) -> Dict[str, str]:
    prefixes = ENV_PREFIXES + tuple(extra_prefixes)
    secret_re = SECRET_RE
    extra = [s for s in extra_secret if s]
    if extra:
        secret_re = re.compile(SECRET_RE.pattern + "|" + "|".join(extra), re.I)
    out: Dict[str, str] = {}
    for k in sorted(env):
        if not (k.startswith(prefixes) or k in prefixes):
            continue
        v = env[k]
        if secret_re.search(k):
            v = "***"
        elif len(v) > max_value:
            v = v[:max_value] + "…"
        out[k] = v
    return out


def stdio_paths() -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for name, fd in (("stdin", 0), ("stdout", 1), ("stderr", 2)):
        try:
            out[name] = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            out[name] = None
    return out


def is_interactive(env: Mapping[str, str], job: Mapping[str, Any]) -> bool:
    if hasattr(sys, "ps1") or sys.flags.interactive:
        return True
    if "ipykernel" in sys.modules:
        return True
    if job.get("interactive_step"):
        return True
    return False


def cpu_count_visible() -> Optional[int]:
    try:
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except Exception:
        return os.cpu_count()


def mem_total_bytes() -> Optional[int]:
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None


# --------------------------------------------------------------------------- assembly


def _safe_hostname() -> str:
    try:
        return socket.gethostname().split(".")[0] or "unknown"
    except Exception:
        return "unknown"


def _safe_user(env: Mapping[str, str]) -> Optional[str]:
    u = env.get("USER") or env.get("LOGNAME")
    if u:
        return u
    try:
        import getpass

        return getpass.getuser()
    except Exception:
        return None


def build_context(config: Any, source: str = "process", env: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """Compute everything that identifies this run. Cheap except one ``git`` call (2 s cap)."""
    env = os.environ if env is None else env
    now = time.time()
    job = detect_scheduler(env)
    rank = detect_rank(env)
    host = _safe_hostname()
    pid = os.getpid()
    cluster = getattr(config, "cluster", None) or job.get("cluster") or job["name"]
    cluster = re.sub(r"[^A-Za-z0-9_.-]", "_", str(cluster))
    key = job_key(job)
    if key is None:
        key = f"{host}-{pid}-{int(now)}"
        restart = 0
    else:
        key = re.sub(r"[^A-Za-z0-9_.-]", "_", str(key))
        restart = int(job.get("restart_count") or 0)
    run_id = f"{cluster}/{key}/{restart}"
    r = rank["rank"] if rank else None
    emitter = f"{source}-r{r if r is not None else 0}-{host}-{pid}"

    deadline: Optional[Dict[str, Any]] = None
    if job.get("end_ts"):
        deadline = {"end_ts": float(job["end_ts"]), "source": "scheduler_env"}
    elif job.get("time_limit_s"):
        deadline = {"end_ts": now + float(job["time_limit_s"]), "source": "walltime_from_now", "approx": True}

    cwd = os.getcwd() if _cwd_ok() else None
    main_file = None
    try:
        main_file = getattr(sys.modules.get("__main__"), "__file__", None)
    except Exception:
        pass

    ctx: Dict[str, Any] = {
        "run_id": run_id,
        "emitter": emitter,
        "cluster": cluster,
        "job": job,
        "rank": rank,
        "host": host,
        "pid": pid,
        "ppid": os.getppid(),
        "user": _safe_user(env),
        "cwd": cwd,
        "argv": list(sys.argv) if source == "process" else None,
        "main_file": main_file if source == "process" else None,
        "executable": sys.executable,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "container": detect_container(env),
        "git": git_info(cwd) if cwd else None,
        "packages": package_versions(getattr(config, "packages", ())),
        "env": filtered_env(env, getattr(config, "env_include", ()), getattr(config, "env_secret", ())),
        "stdio": stdio_paths(),
        "interactive": is_interactive(env, job),
        "deadline": deadline,
        "start_ts": now,
        "cpu_count": cpu_count_visible(),
        "mem_total": mem_total_bytes(),
        "wrapped": bool(env.get("JGM_WRAPPED")),
    }
    if ctx["container"]:
        ctx["mounts"] = mount_points()
    dbg(f"context run_id={run_id} emitter={emitter} scheduler={job['name']} rank={rank}")
    return ctx


def _cwd_ok() -> bool:
    try:
        os.getcwd()
        return True
    except OSError:
        return False


def run_dir_parts(run_id: str) -> Tuple[str, ...]:
    """``cluster/key/restart`` -> path components under ``<base>/runs/``."""
    return tuple(run_id.split("/"))
