"""Script to monitor the RAM usage by TemporalSentinel2Dataset."""
import multiprocessing as mp
import sys
import time

from biomasstry.datasets import TemporalSentinel2Dataset
from common import MemoryMonitor
import torch

def worker(_, dataset: torch.utils.data.Dataset):
    while True:
        for i, sample in enumerate(dataset):
            input, target, chip_id = sample
            print(f"{i+1}. Chip: {chip_id}. Input: {input.size()}. Target: {target.size()}.")


if __name__ == "__main__":
  start_method = sys.argv[1]
  monitor = MemoryMonitor()
  ds = TemporalSentinel2Dataset()
  print(monitor.table())
  if start_method == "forkserver":
    # Reduce 150M-per-process USS due to "import torch".
    mp.set_forkserver_preload(["torch"])

  ctx = torch.multiprocessing.start_processes(
      worker, (ds, ), nprocs=4, join=False,
      daemon=True, start_method=start_method)
  [monitor.add_pid(pid) for pid in ctx.pids()]

  try:
    for k in range(100):
      print(monitor.table())
      time.sleep(5)
  finally:
    ctx.join()
