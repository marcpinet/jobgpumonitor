"""Short GPU smoke test for jobgpumonitor (about 45 s, one GPU, tiny memory).

Exercises: run.start, heartbeats, resource samples with GPU, tqdm progress with ETA vs
deadline, metric.log, log.line, and (with --crash) run.exception + run.end status=error.
"""

import argparse
import logging
import sys
import time

import jobgpumonitor.auto  # noqa: F401  <- the only line a real project needs

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=45)
parser.add_argument("--crash", action="store_true", help="raise at the end to test exception capture")
args = parser.parse_args()

try:
    import torch
except ImportError:
    torch = None

from tqdm import tqdm  # noqa: E402

import jobgpumonitor  # noqa: E402

device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
print(f"device={device} torch={getattr(torch, '__version__', None)}", flush=True)
if torch is not None and device == "cuda":
    print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    a = torch.randn(2048, 2048, device=device)
    b = torch.randn(2048, 2048, device=device)

logging.basicConfig(level=logging.INFO)
logging.getLogger("smoke").warning("this warning should appear as a log.line event")

steps = 30
t_end = time.time() + args.seconds
for step in tqdm(range(steps), desc="train"):
    t_step_end = time.time() + args.seconds / steps
    n = 0
    while time.time() < t_step_end:
        if torch is not None and device == "cuda":
            c = a @ b
            n += 1
        else:
            time.sleep(0.05)
    if torch is not None and device == "cuda":
        torch.cuda.synchronize()
    jobgpumonitor.log(loss=1.0 / (step + 1), matmuls=n, step=step)

jobgpumonitor.emit("stage", name="done", device=device)
if args.crash:
    raise RuntimeError("intentional crash to test run.exception")
print("finished cleanly", flush=True)
sys.exit(0)
