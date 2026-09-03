# Configuration

Every option is a `JGM_*` environment variable, or a keyword argument to
`jobgpumonitor.watch(...)` (keyword wins). All are optional.

| Variable | Default | Meaning |
|---|---|---|
| `JGM_DISABLED` | `0` | switch everything off (`JGM_ENABLED=0` works too) |
| `JGM_DIR` | `~/.jobgpumonitor` | base directory of the file sink; falls back to `./.jgm` when the home is not writable |
| `JGM_SINKS` | `file` | comma list of sinks: `file`, `stderr` |
| `JGM_CLUSTER` | scheduler's cluster name, else scheduler name | first component of `run_id` |
| `JGM_HEARTBEAT_S` | `15` | seconds between `run.heartbeat` events |
| `JGM_SAMPLE_S` | `10` | seconds between `resource.sample` events during the first hour |
| `JGM_SAMPLE_SLOW_S` | `60` | sampling interval after `JGM_SAMPLE_SLOW_AFTER_S` |
| `JGM_SAMPLE_SLOW_AFTER_S` | `3600` | when to switch to the slow interval |
| `JGM_PROGRESS_S` | `1` | minimum interval between two `progress.update` for the same tqdm bar |
| `JGM_RANK_MODE` | `rank0` | `rank0`: ranks other than 0 emit only start/heartbeat/exception/end; `all`: every rank emits everything; `off`: other ranks emit nothing |
| `JGM_TQDM` | `1` | hook tqdm |
| `JGM_LOGGING_LEVEL` | `WARNING` | forward `logging` records at or above this level as `log.line`; empty disables |
| `JGM_SIGNALS` | `1` | observe SIGTERM, SIGINT, SIGHUP, SIGUSR1, SIGUSR2, SIGXCPU, SIGQUIT (chained to existing handlers) |
| `JGM_FAULTHANDLER` | `1` | enable `faulthandler` so segfaults leave a trace in stderr |
| `JGM_CAPTURE_LOCALS` | `0` | include local variables (secrets masked, 200 chars each) in crash frames |
| `JGM_ENV_INCLUDE` | | extra environment variable prefixes to record in `run.start` |
| `JGM_ENV_SECRET` | | extra regex fragments marking a variable name as secret |
| `JGM_PACKAGES` | torch, numpy, transformers, … | packages whose versions are recorded |
| `JGM_FSYNC` | `1` | fsync after every event (needed for visibility across NFS clients) |
| `JGM_DEBUG` | `0` | internal diagnostics on stderr |

Variables set by the tooling itself, not meant to be set by hand: `JGM_WRAPPED` (set by
`jgm run` for the child process).
