import functools
import json
import logging
import math
import multiprocessing
import os
import time
from collections import defaultdict
from contextlib import contextmanager

import tqdm

ctqdm = functools.partial(
    tqdm.tqdm, ncols=0, mininterval=10.0, maxinterval=30.0
)


def enabled_only(method):
    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        if self.enabled:
            method(self, *args, **kwargs)

    return wrapped


class Seri:
    def __init__(
        self,
        name: str = None,
        accelerator=None,
        main_process_only: bool = False,
    ):
        if len(logging.root.handlers) == 0:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(name)
        self.accelerator = accelerator

        self.enabled = (not main_process_only) or (
            (accelerator is None or accelerator.is_main_process)
            and multiprocessing.parent_process() is None
        )

        self.data = defaultdict(list)
        self._buf = defaultdict(_Stats)

        self._tick_curr_name: str | None = None
        self._tick_time = time.perf_counter()
        self._step_time = time.perf_counter()
        self._time_buf = defaultdict(_Stats)
        self._prefix = ""

    @contextmanager
    def context(self, name: str):
        prev_prefix = self._prefix
        try:
            self._prefix = name
            yield
        finally:
            self._prefix = prev_prefix

    @enabled_only
    def push(self, metrics=None, **kwargs):
        metrics = {} if metrics is None else metrics
        for k, v in {**metrics, **kwargs}.items():
            if hasattr(v, "item"):
                v = v.item()
            self._buf[os.path.join(self._prefix, k)].push(v)

    @enabled_only
    def extend(self, metrics=None, **kwargs):
        metrics = {} if metrics is None else metrics
        for k, v in {**metrics, **kwargs}.items():
            if hasattr(v, "detach"):
                v = v.detach().cpu().float().numpy()
            assert len(v.shape) == 1
            for entry in v:
                self._buf[os.path.join(self._prefix, k)].push(float(entry))

    @enabled_only
    def step(self):
        for k, stats in self._buf.items():
            self.data[f"{k}.data"].append(stats.mean())
        for k, stats in self._buf.items():
            self.data[f"{k}.std"].append(stats.std())
        self._buf.clear()

        total_time = time.perf_counter() - self._step_time
        for k, stats in self._time_buf.items():
            self.data[f"{k}.time_avg"].append(stats.mean())
            self.data[f"{k}.time_pct"].append(stats.sum() / total_time * 100.0)
        self._time_buf.clear()
        self._step_time = time.perf_counter()

    @enabled_only
    def tick(self, name: str | None = None):
        if self._tick_curr_name is not None:
            self._time_buf[
                os.path.join(self._prefix, self._tick_curr_name)
            ].push(time.perf_counter() - self._tick_time)
        self._tick_curr_name = name
        self._tick_time = time.perf_counter()

    @enabled_only
    def push_vram(self):
        import torch

        if not torch.cuda.is_available() or (
            self.accelerator is not None
            and self.accelerator.device.type == "cpu"
        ):
            return

        device = None if self.accelerator is None else self.accelerator.device
        max_allocated = torch.cuda.max_memory_allocated(device)
        total = torch.cuda.get_device_properties(device).total_memory
        self.push(
            vram_usage=max_allocated / total * 100.0,
        )

    @enabled_only
    def save(self, filepath: str | os.PathLike):
        with open(filepath, "w") as f:
            json.dump(self.data, f)


class _Stats:
    def __init__(self):
        self.n = 0
        self.m = 0
        self.s = 0

    def push(self, x):
        self.n += 1
        if self.n == 1:
            self.m = x
        else:
            prev = self.m
            self.m = prev + (x - prev) / self.n
            self.s = self.s + (x - prev) * (x - self.m)

    def mean(self):
        return self.m

    def sum(self):
        return self.m * self.n

    def std(self):
        if self.n <= 1:
            return 0
        return math.sqrt(self.s / (self.n - 1))
