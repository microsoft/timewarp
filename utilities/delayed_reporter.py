from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple
import torch
from torch.types import Number

DEFAULT_REPORTER_BUFFER_SIZE = 128


@contextmanager
def ensure_flush():
    try:
        yield None
    finally:
        flush_all()


def set_reporter_buffer_size(buffer_size: int):
    """
    Sets the buffer size to a new value. A larger buffer size reduces device-to-host communications
    but increases the delay in reporting.

    Args:
        buffer_size: new buffer size (default: 128)
    """
    global _delayed_reporter_singleton
    flush_all()
    _delayed_reporter_singleton = DelayerReporterSingleton(buffer_size)


def report_async(scalar: torch.Tensor, task: Callable[[Number], None]):
    """
    Asynchronously obtains the value of a scalar tensor and calls a callback function.

    Args:
        scalar: scalar tensor on some non-cpu device.
        task: callback function called when the value of the tensor is available.
    """
    global _delayed_reporter_singleton
    if _delayed_reporter_singleton is None:
        _delayed_reporter_singleton = DelayerReporterSingleton(DEFAULT_REPORTER_BUFFER_SIZE)
    assert len(scalar.shape) == 0

    reporter = _delayed_reporter_singleton.get_reporter(scalar.dtype, scalar.device)
    callback = reporter.record(scalar)
    callback.schedule(task)
    reporter.flush_if_full()


def flush_all():
    """
    Flushes all the pending communication and reporting.
    """
    if _delayed_reporter_singleton is not None:
        # need to clear up the reporters
        _delayed_reporter_singleton.flush_all()


def flush_stats():
    """
    Returns the number of flushes for each dtype and device.
    """
    return (
        _delayed_reporter_singleton.flush_stats() if _delayed_reporter_singleton is not None else {}
    )


class DelayedReporterCallback:
    def __init__(self):
        self.tasks: List[Callable[[Number], None]] = []

    def schedule(self, task: Callable[[Number], None]):
        self.tasks.append(task)

    def run(self, scalar: Number):
        for task in self.tasks:
            task(scalar)


class DelayedReporter:
    def __init__(self, buffer_size: int, dtype: torch.dtype, device: torch.device):
        self.buffer = torch.empty((buffer_size,), dtype=dtype, device=device)
        self.callbacks: List[DelayedReporterCallback] = []
        self.flush_count = 0

    @property
    def current_position(self):
        return len(self.callbacks)

    def record(self, scalar: torch.Tensor) -> DelayedReporterCallback:
        assert scalar.dtype == self.buffer.dtype
        if self.current_position >= len(self.buffer):
            self.flush()
        self.buffer[self.current_position] = scalar.detach()
        callback = DelayedReporterCallback()
        self.callbacks.append(callback)
        return callback

    def flush(self):
        assert len(self.callbacks) <= len(self.buffer)
        buffer_cpu = self.buffer.cpu()
        for callback, scalar in zip(self.callbacks, buffer_cpu):
            callback.run(scalar.item())
        self.callbacks = []
        self.flush_count += 1

    def flush_if_full(self):
        if self.current_position == len(self.buffer):
            self.flush()


@dataclass(frozen=True)
class DelayerReporterSingleton:
    reporter_buffer_size: int
    # Stores one DelayedReporter per (dtype, device).
    _reporters: Dict[Tuple[torch.dtype, torch.device], DelayedReporter] = field(
        default_factory=dict, repr=False
    )

    def get_reporter(self, dtype: torch.dtype, device: torch.device):
        key = (dtype, device)
        if key in self._reporters:
            return self._reporters[key]
        print(
            f"I: Creating a new delayed reporter for device {device} and dtype {dtype} with size {self.reporter_buffer_size}"
        )
        reporter = DelayedReporter(self.reporter_buffer_size, dtype, device)
        self._reporters[key] = reporter
        return reporter

    def flush_all(self):
        """
        Flushes all the pending communication and reporting.
        """
        for reporter in self._reporters.values():
            reporter.flush()
            assert reporter.current_position == 0

    def flush_stats(self):
        return {key: reporter.flush_count for key, reporter in self._reporters.items()}


# Singleton object that holds the current buffer size and reporters per dtype and device.
_delayed_reporter_singleton = None
