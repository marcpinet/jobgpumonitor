from __future__ import annotations

from jobgpumonitor import context as C
from jobgpumonitor.config import Config


def test_slurm_detection_and_array_key():
    env = {
        "SLURM_JOB_ID": "1001", "SLURM_JOB_NAME": "tsad", "SLURM_ARRAY_JOB_ID": "1000",
        "SLURM_ARRAY_TASK_ID": "7", "SLURM_RESTART_COUNT": "2", "SLURM_JOB_PARTITION": "All",
        "SLURM_JOB_START_TIME": "1000", "SLURM_JOB_END_TIME": "4600", "SLURM_CLUSTER_NAME": "marcel",
    }
    job = C.detect_scheduler(env)
    assert job["name"] == "slurm"
    assert job["restart_count"] == 2
    assert job["time_limit_s"] == 3600
    assert C.job_key(job) == "1000_7"
    ctx = C.build_context(Config(), env=env)
    assert ctx["run_id"] == "marcel/1000_7/2"
    assert ctx["deadline"]["source"] == "scheduler_env"
    assert ctx["deadline"]["end_ts"] == 4600.0


def test_oar_detection():
    env = {"OAR_JOB_ID": "555", "OAR_JOB_NAME": "x", "OAR_JOB_WALLTIME": "1:30:00"}
    job = C.detect_scheduler(env)
    assert job["name"] == "oar"
    assert job["time_limit_s"] == 5400
    ctx = C.build_context(Config(), env=env)
    assert ctx["run_id"].startswith("oar/555/0")
    assert ctx["deadline"]["approx"] is True


def test_local_run_id_is_unique_per_process():
    ctx = C.build_context(Config(), env={})
    cluster, key, restart = ctx["run_id"].split("/")
    assert cluster == "local"
    assert restart == "0"
    assert str(ctx["pid"]) in key


def test_cluster_override_sanitized():
    ctx = C.build_context(Config(cluster="my cluster/1"), env={"SLURM_JOB_ID": "1"})
    assert ctx["run_id"].split("/")[0] == "my_cluster_1"


def test_rank_detection_prefers_torchrun_and_ignores_single_task_slurm():
    assert C.detect_rank({"SLURM_PROCID": "0", "SLURM_NTASKS": "1"}) is None
    r = C.detect_rank({"SLURM_PROCID": "3", "SLURM_NTASKS": "8", "SLURM_LOCALID": "3"})
    assert r == {"rank": 3, "world_size": 8, "local_rank": 3, "launcher": "slurm"}
    r = C.detect_rank({"RANK": "1", "WORLD_SIZE": "4", "LOCAL_RANK": "1", "SLURM_PROCID": "0", "SLURM_NTASKS": "4"})
    assert r["launcher"] == "torchrun" and r["rank"] == 1
    assert C.detect_rank({"RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0"}) is None  # some images export this


def test_parse_mem_bytes():
    assert C.parse_mem_bytes("8192") == 8192 * 1024**2
    assert C.parse_mem_bytes("8G") == 8 * 1024**3
    assert C.parse_mem_bytes("512M") == 512 * 1024**2
    assert C.parse_mem_bytes("2gb") == 2 * 1024**3
    assert C.parse_mem_bytes("") is None and C.parse_mem_bytes("x") is None


def test_env_filter_masks_secrets_and_allowlists():
    env = {
        "SLURM_JOB_ID": "1", "WANDB_API_KEY": "abc", "HF_TOKEN": "t", "MY_RANDOM_VAR": "x",
        "CUDA_VISIBLE_DEVICES": "0,1", "AWS_SECRET_ACCESS_KEY": "s", "CUSTOM_PREFIX_A": "1",
    }
    out = C.filtered_env(env, extra_prefixes=("CUSTOM_",))
    assert out["SLURM_JOB_ID"] == "1"
    assert out["WANDB_API_KEY"] == "***"
    assert out["HF_TOKEN"] == "***"
    assert out["CUSTOM_PREFIX_A"] == "1"
    assert "MY_RANDOM_VAR" not in out
    assert "AWS_SECRET_ACCESS_KEY" not in out  # not allow-listed at all


def test_hms_parsing():
    assert C._hms_to_seconds("1-02:00:00") == 93600
    assert C._hms_to_seconds("00:30:00") == 1800
    assert C._hms_to_seconds("45:00") == 2700
    assert C._hms_to_seconds("garbage") is None


def test_config_from_env_types():
    cfg = Config.from_env(env={"JGM_HEARTBEAT_S": "3", "JGM_SINKS": "file, stderr", "JGM_TQDM": "no", "JGM_DISABLED": "1", "JGM_RANK_MODE": "weird"})
    assert cfg.heartbeat_s == 3.0
    assert cfg.sinks == ("file", "stderr")
    assert cfg.tqdm is False
    assert cfg.enabled is False
    assert cfg.rank_mode == "rank0"
    cfg2 = Config.from_env({"heartbeat_s": 7, "unknown_thing": 1}, env={})
    assert cfg2.heartbeat_s == 7 and cfg2.extra == {"unknown_thing": 1}
