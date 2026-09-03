# Event protocol v1

jobgpumonitor emits **one JSON object per line** (JSONL, UTF-8, append-only). This document
is the contract between the emitter and any consumer (a server, a dashboard, a script).
The machine-readable version is [`schema/event.schema.json`](../schema/event.schema.json).

## Envelope

```json
{"v":1,"id":"01J8Z3H2K5R9E2Y7PB3X4Q6MNA","seq":42,"ts":"2026-09-03T10:14:05.123Z","mono":1234.567,
 "run_id":"marcel-c3/1234567/0","emitter":"process-r0-node12-4242","pid":4242,"source":"process",
 "rank":null,"type":"run.heartbeat","data":{"uptime_s":1234.5}}
```

| Field | Meaning |
|---|---|
| `v` | protocol major version, `1` |
| `id` | ULID, unique and sortable by time |
| `seq` | sequence number per `(emitter, pid)`, from 0; a gap means events were lost |
| `ts` | UTC wall-clock time of emission |
| `mono` | seconds since the emitter started, monotonic clock, use it for durations |
| `run_id` | `<cluster>/<job key>/<restart count>`; job key is `<array_job>_<task>` for array tasks, `<host>-<pid>-<start>` without a scheduler |
| `emitter` | `<source>-r<rank>-<host>-<pid>`, or `shell-<host>` for `jgm emit` |
| `pid` | emitting process id |
| `source` | `process` (in-process library), `wrapper` (`jgm run`), `shell` (`jgm emit`), `scheduler` (login-node probe, phase 2) |
| `rank` | global distributed rank, `null` when not distributed |
| `type` | dotted lowercase type; custom types are `custom.<name>` |
| `data` | type-specific object |

## Compatibility rules

- Consumers ignore unknown fields and unknown types.
- Adding a field or a type does not change `v`. Removing or renaming one increments `v`.
- The last line of a file being written may be truncated; consumers keep it until a newline arrives.

## File layout (file sink)

```
$JGM_DIR/runs/<cluster>/<job key>/<restart>/<emitter>.jsonl
```

One file per emitting process, so concurrent appends from different nodes never interleave
(NFS `O_APPEND` is not atomic across clients). A run directory therefore contains one file
per rank, plus `wrapper-*.jsonl` and `shell-<host>.jsonl` when used. Every write is flushed
and fsynced. `$JGM_DIR` defaults to `~/.jobgpumonitor`, then `./.jgm` if the home is not
writable (typical inside a container).

## Event types

### `run.start`
Emitted once per emitter. Everything needed to identify and locate the run:
`scheduler` (name, job_id, job_name, array ids, restart_count, partition, nodes, ntasks,
cpus_per_task, gpus, mem, start_ts, end_ts, time_limit_s), `rank`, `host`, `user`, `cwd`,
`argv`, `main_file`, `python`, `platform`, `container`, `mounts` (containers only), `git`
(commit, branch, dirty, remote without credentials), `packages` (versions), `env`
(allow-listed, secrets masked), `stdio` (real paths of fd 0/1/2, i.e. the `.out`/`.err`
files), `interactive`, `wrapped`, `deadline` (`end_ts`, `source`, `approx`), `resources`
(cpu_count, mem_total, gpu devices, cgroup limits, disk), `emitter_config`.
From `jgm run`: also `command` and `child_pid`.

### `run.heartbeat`
Every `JGM_HEARTBEAT_S` seconds (default 15): `uptime_s`, `deadline_remaining_s`,
`progress` (latest state of each live tqdm bar), `metrics` (latest logged values),
`samples`, `dropped`, `threads`. A consumer that only reads heartbeats can already show
"alive, 63 %, ETA 40 min, loss 0.12".

### `resource.sample`
Every `JGM_SAMPLE_S` seconds (default 10, then 60 after one hour): `gpus[]` (util, mem_util,
mem_used, mem_total, temp_c, power_w, sm_clock_mhz), `proc` (cpu_pct, rss, rss_peak,
threads, children, children_rss, open_fds), `cgroup` (mem_used, mem_limit, mem_peak,
mem_pct, events_oom_kill), `disk` (cwd total/free), `load`.

### `progress.update`
One per tqdm refresh, throttled to `JGM_PROGRESS_S` (default 1 s) per bar: `bar_id`, `desc`,
`n`, `total`, `unit`, `rate`, `elapsed_s`, `eta_s`, `done`, and when a deadline is known
`deadline_remaining_s` and `eta_vs_deadline_s` (negative: will not finish in time).

### `metric.log`
`jobgpumonitor.log(loss=0.1, step=10)`: `metrics`, `step`, `epoch`. Values are sanitised
(numpy/torch scalars converted, NaN/inf become `null`).

### `log.line`
`logging` records at or above `JGM_LOGGING_LEVEL` (default WARNING): `logger`, `level`,
`levelno`, `message`, `file`, `line`, `func`, `exception`.

### `signal.received`
`signal`, `signum`, `previous_handler`, `will_terminate`, `deadline_remaining_s`; from the
wrapper also `forwarded`.

### `run.exception`
`type`, `module`, `message`, `traceback` (text, capped at 32 kB), `frames` (last 12:
file, line, func, code, optional masked `locals`), `is_oom`, `fatal`, `thread`, `kind`
(`error`, `interrupted`, `thread`).

### `run.end`
`status` (`ok`, `error`, `interrupted`, `killed`, `interactive`), `exit_code` (best effort
from inside the process, authoritative from `jgm run`), `duration_s`, `end_ts`,
`exception` (short form), `metrics`, `progress`, `summary` (gpu util mean and idle
fraction, gpu mem max, rss max, cgroup mem max, cpu mean), `dropped`; from `jgm run` also
`signal`, `stderr_tail`, `command`, `forwarded_signals`.

### `custom.*`
`jobgpumonitor.emit("stage", name="eval")` or `jgm emit stage name=eval`.

### Reserved for phase 2
`checkpoint.saved`, `stack.dump`, `scheduler.state` (from the login-node probe: state,
reason, node, time limit, exit code, MaxRSS as reported by `sacct` / `oarstat`).

## What a consumer should derive (not the emitter's job)

- **Alive / hung**: no heartbeat for 3 intervals while the scheduler still says RUNNING.
- **Killed**: file stops without `run.end`; the scheduler state says why (OOM, TIMEOUT, CANCELLED, PREEMPTED, NODE_FAIL).
- **GPU idle**: `resource.sample.gpus[].util < 5` for N minutes.
- **Memory pressure**: `cgroup.mem_pct > 90`.
- **Will time out**: `progress.update.eta_vs_deadline_s < 0`.
- **Array digest**: group runs by `scheduler.array_job_id`.
- **Requeue**: same job key, different restart count.
