"""Copied from realpython, minor adjustments."""
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class."""


@dataclass
class Timer:
    timers: ClassVar[dict[str, float]] = {}
    name: str | None = None
    text: str = "Elapsed time: {:0.4f} seconds"
    logger: Callable[[str], None] | None = print
    _start_time: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Add timer to dict of timers after initialization."""
        if self.name is not None:
            self.timers.setdefault(self.name, 0)

    def start(self) -> None:
        """Start a new timer."""
        if self._start_time is not None:
            msg = "Timer is running. Use .stop() to stop it"
            raise TimerError(msg)

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time."""
        if self._start_time is None:
            msg = "Timer is not running. Use .start() to start it"
            raise TimerError(msg)

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        # Report elapsed time
        if self.logger:
            self.logger(self.text.format(elapsed_time))
        if self.name:
            self.timers[self.name] += elapsed_time

        return elapsed_time
