"""Stage-level CUDA residency; weights/dtypes and model computations are unchanged."""
from contextlib import contextmanager
from functools import wraps
import os
import sys
import time

import torch

from .. import dev
from ..sampling.fast import clear_weights

GIB = 2 ** 30


def move_model(model, device):
    clear_weights(model)
    # Prep runs under inference_mode. Device transfers must still create ordinary
    # parameter tensors: these weights survive the context and later requests.
    with torch.inference_mode(False), torch.no_grad():
        model.to(device)


def managed_request(method):
    """Release managed weights even when preprocessing, sampling or export raises."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self.residency.request():
            return method(self, *args, **kwargs)
    return wrapped


class ModelResidency:
    """An auto decision latches on for this pipeline, avoiding residency oscillation.

    Thresholds reserve GPU headroom for preprocessing and generation.
    Windows defaults to auto; Linux and ZeroGPU retain their existing default behavior.
    Caller must serialize requests (the demo already holds GPU_LOCK).
    """
    def __init__(self, device=None, mode=None, zero_gpu=False):
        self.device = dev.select(device)
        self.mode = mode or os.environ.get('HAIRDECK_OFFLOAD', 'auto' if sys.platform == 'win32' else 'off')
        if self.mode not in ('auto', 'on', 'off'):
            raise ValueError('HAIRDECK_OFFLOAD must be auto, on or off')
        self.supported = self.device.type == 'cuda' and not zero_gpu
        self.enabled = self.supported and self.mode == 'on'
        self.groups = {}
        self.on_gpu = set()
        self.events = []
        self.reason = 'explicit on' if self.enabled else 'disabled'
        self._consider(initial=True)
        if self.supported and self.mode != 'off':
            self._log(f'mode={self.mode}, offload={self.enabled}: {self.reason}')

    @property
    def initial_device(self):
        return 'cpu' if self.enabled else self.device

    def _log(self, message):
        print(time.strftime('%F %T'), 'MEMORY', message, flush=True)

    def _snapshot(self):
        free, total = torch.cuda.mem_get_info(self.device)
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        return dict(free_gib=round(free / GIB, 3), total_gib=round(total / GIB, 3),
                    allocated_gib=round(allocated / GIB, 3), reserved_gib=round(reserved / GIB, 3)), (
                        free + max(0, reserved - allocated), total)

    def _consider(self, initial=False):
        if not self.supported or self.mode != 'auto' or self.enabled:
            return
        try:
            snapshot, (available, total) = self._snapshot()
            budget = (10 if initial else 4) * GIB
            self.enabled = total < 12 * GIB or available < budget
            self.reason = f"available={available / GIB:.2f} GiB, total={total / GIB:.2f} GiB"
            if not initial:
                self.events.append(dict(stage='request', **snapshot))
        except RuntimeError as exc:
            self.enabled = True
            self.reason = f'CUDA memory query failed; conservative offload ({type(exc).__name__})'
        if self.enabled and not initial:
            self._log('enabling offload: ' + self.reason)

    def register(self, name, *models):
        if name in self.groups:
            raise ValueError(f'duplicate residency group: {name}')
        self.groups[name] = tuple(model for model in models if model is not None)
        if self.enabled:
            for model in self.groups[name]:
                move_model(model, 'cpu')
        else:
            self.on_gpu.add(name)

    def activate(self, name):
        if name not in self.groups:
            raise KeyError(name)
        if not self.enabled:
            return
        # Evict BEFORE loading the next group, so weights never overlap at a switch.
        evicted = False
        for other in tuple(self.on_gpu - {name}):
            for model in self.groups[other]:
                move_model(model, 'cpu')
            self.on_gpu.remove(other)
            evicted = True
        if evicted:
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        if name not in self.on_gpu:
            # Track before transfer so a partial transfer/OOM is cleaned by request().
            self.on_gpu.add(name)
            for model in self.groups[name]:
                move_model(model, self.device)
            try:
                snapshot, _ = self._snapshot()
            except RuntimeError as exc:
                snapshot = dict(memory_query_error=type(exc).__name__)
            self.events.append(dict(stage=name, **snapshot))
            self._log(f'{name}: {snapshot}')

    def release(self):
        if not self.enabled:
            return
        for name in tuple(self.on_gpu):
            for model in self.groups[name]:
                move_model(model, 'cpu')
            self.on_gpu.remove(name)
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()

    @contextmanager
    def request(self):
        self.events = []
        self._consider()
        try:
            yield
        finally:
            self.release()

    def report(self):
        return dict(mode=self.mode, enabled=self.enabled, reason=self.reason, stages=list(self.events))
