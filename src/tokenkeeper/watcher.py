"""File watcher for TokenKeeper.

Provides filesystem monitoring with debouncing and git burst detection
for triggering incremental reindexing.

Exports:
    BurstDetector        -- Sliding-window burst detector for git ops
    DebounceAccumulator  -- Thread-safe time-windowed path batcher
    FileWatcher          -- Watchdog-based file watcher with debounce/burst
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEventHandler, FileSystemEvent
from watchdog.observers import Observer

logger = logging.getLogger("tokenkeeper.watcher")


class DebounceAccumulator:
    """Thread-safe accumulator that batches file paths within a time window.

    When a file path is added, its timer is set/reset to the current time.
    flush() returns only paths whose timer has expired (been quiet for
    at least window_seconds).

    Args:
        window_seconds: How long a path must be quiet before flushing.
    """

    def __init__(self, window_seconds: float = 3.0) -> None:
        self._window = window_seconds
        self._pending: dict[str, float] = {}  # path -> last_event_timestamp
        self._lock = threading.Lock()

    def add(self, path: str) -> None:
        """Add or reset a file path's debounce timer.

        Thread-safe. If the path already exists, its timer is reset.
        """
        with self._lock:
            self._pending[path] = time.monotonic()

    def flush(self) -> list[str]:
        """Return paths that have been quiet for at least window_seconds.

        Thread-safe. Returned paths are removed from pending.
        Returns empty list if no paths are ready.
        """
        now = time.monotonic()
        with self._lock:
            ready = [
                path for path, ts in self._pending.items()
                if now - ts >= self._window
            ]
            for path in ready:
                del self._pending[path]
        return sorted(ready)

    def pending_count(self) -> int:
        """Number of paths currently waiting."""
        with self._lock:
            return len(self._pending)

    def clear(self) -> None:
        """Discard all pending paths."""
        with self._lock:
            self._pending.clear()


class BurstDetector:
    """Detects rapid bursts of file system events (e.g., git operations).

    Maintains a sliding window of event timestamps. When the event count
    within the window exceeds the threshold, is_bursting() returns True.

    The debounce accumulator should suppress flushing while bursting
    to avoid triggering reindex during git checkout/merge/rebase.

    Args:
        threshold: Number of events that triggers burst mode.
        window_seconds: Time window for counting events.
    """

    def __init__(self, threshold: int = 20, window_seconds: float = 5.0) -> None:
        self._threshold = threshold
        self._window = window_seconds
        self._timestamps: list[float] = []
        self._lock = threading.Lock()

    def record_event(self) -> None:
        """Record a file system event."""
        now = time.monotonic()
        with self._lock:
            self._timestamps.append(now)
            cutoff = now - self._window
            self._timestamps = [t for t in self._timestamps if t > cutoff]

    def is_bursting(self) -> bool:
        """Check if we're currently in a burst of events.

        Returns True if the number of events in the last window_seconds
        exceeds threshold.
        """
        now = time.monotonic()
        with self._lock:
            cutoff = now - self._window
            self._timestamps = [t for t in self._timestamps if t > cutoff]
            return len(self._timestamps) >= self._threshold

    def event_count(self) -> int:
        """Number of events in the current window."""
        now = time.monotonic()
        with self._lock:
            cutoff = now - self._window
            self._timestamps = [t for t in self._timestamps if t > cutoff]
            return len(self._timestamps)

    def reset(self) -> None:
        """Clear all recorded events."""
        with self._lock:
            self._timestamps.clear()


class _RagEventHandler(FileSystemEventHandler):
    """Handles filesystem events, filters, and feeds to accumulator.

    Filters events by:
    - File extension (only watches configured extensions)
    - Excluded directory names (e.g. .git, node_modules, __pycache__)
    - Directory events (always ignored)

    Valid events are recorded in the burst detector and added to the
    debounce accumulator for batched processing.
    """

    def __init__(
        self,
        accumulator: DebounceAccumulator,
        burst_detector: BurstDetector,
        extensions: frozenset[str],
        excluded_dirs: frozenset[str],
    ) -> None:
        self._accumulator = accumulator
        self._burst = burst_detector
        self._extensions = extensions
        self._excluded = excluded_dirs

    def _should_handle(self, path: str) -> bool:
        """Check if this path should be watched."""
        p = Path(path)
        # Check exclusion: any path component matching excluded dir names
        for part in p.parts:
            if part in self._excluded:
                return False
        # Check extension
        return p.suffix.lower() in self._extensions

    def on_any_event(self, event: FileSystemEvent) -> None:
        """Handle any filesystem event after filtering."""
        if event.is_directory:
            return
        if not self._should_handle(event.src_path):
            return
        self._burst.record_event()
        self._accumulator.add(event.src_path)


class FileWatcher:
    """Filesystem watcher with debouncing and burst detection.

    Monitors project root for file changes and triggers incremental
    reindexing via a callback when debounced paths are ready.

    Uses watchdog Observer for native filesystem event monitoring,
    DebounceAccumulator for batching rapid changes, and BurstDetector
    for suppressing flushes during git operations.

    Args:
        watch_root: Directory to monitor recursively.
        extensions: File extensions to watch (e.g. {".md", ".py"}).
        excluded_dirs: Directory names to ignore (e.g. {".git"}).
        debounce_seconds: Debounce window for accumulator.
        burst_threshold: Events triggering burst mode.
        burst_window_seconds: Window for burst counting.
        on_files_changed: Callback receiving list of changed file paths.
        poll_interval: How often to check for flushed paths (seconds).
    """

    def __init__(
        self,
        watch_root: Path,
        extensions: frozenset[str],
        excluded_dirs: frozenset[str],
        debounce_seconds: float = 3.0,
        burst_threshold: int = 20,
        burst_window_seconds: float = 5.0,
        on_files_changed: Callable[[list[str]], None] | None = None,
        poll_interval: float = 1.0,
    ) -> None:
        self._watch_root = watch_root
        self._observer = Observer()
        self._accumulator = DebounceAccumulator(debounce_seconds)
        self._burst = BurstDetector(burst_threshold, burst_window_seconds)
        self._handler = _RagEventHandler(
            self._accumulator, self._burst, extensions, excluded_dirs,
        )
        self._callback = on_files_changed or (lambda _paths: None)
        self._poll_interval = poll_interval
        self._running = False
        self._flush_thread: threading.Thread | None = None

    def start(self) -> None:
        """Start watching the root directory recursively.

        Schedules the watchdog observer and spawns a daemon flush thread
        that periodically checks the accumulator.
        """
        self._observer.schedule(
            self._handler, str(self._watch_root), recursive=True,
        )
        self._observer.start()
        self._running = True
        self._flush_thread = threading.Thread(
            target=self._flush_loop, daemon=True,
        )
        self._flush_thread.start()
        logger.info("FileWatcher started on %s", self._watch_root)

    def stop(self) -> None:
        """Stop watching and clean up threads.

        Stops the observer and waits for both the observer and flush
        threads to terminate (with a 5-second timeout each).
        """
        self._running = False
        self._observer.stop()
        self._observer.join(timeout=5)
        if self._flush_thread:
            self._flush_thread.join(timeout=5)
        logger.info("FileWatcher stopped")

    def _flush_loop(self) -> None:
        """Periodically check accumulator and trigger callback."""
        while self._running:
            time.sleep(self._poll_interval)
            if self._burst.is_bursting():
                logger.debug("Burst detected, suppressing flush")
                continue
            ready = self._accumulator.flush()
            if ready:
                logger.info(
                    "Flushing %d changed files for reindex", len(ready),
                )
                try:
                    self._callback(ready)
                except Exception:
                    logger.exception("Error in file change callback")
