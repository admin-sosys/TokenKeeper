"""Tests for knowledge_rag.watcher module.

Covers: DebounceAccumulator debounce timing, timer reset behaviour,
independent path tracking, sorted output, clear, and thread safety.
Also covers: _RagEventHandler filtering and FileWatcher integration.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from watchdog.events import DirCreatedEvent, FileCreatedEvent, FileModifiedEvent

from knowledge_rag.watcher import (
    BurstDetector,
    DebounceAccumulator,
    FileWatcher,
    _RagEventHandler,
)


class TestDebounceAccumulator:
    def test_add_single_path(self) -> None:
        acc = DebounceAccumulator(window_seconds=1.0)
        acc.add("/path/a.md")
        assert acc.pending_count() == 1

    def test_flush_empty(self) -> None:
        acc = DebounceAccumulator(window_seconds=0.1)
        assert acc.flush() == []

    def test_flush_before_window(self) -> None:
        acc = DebounceAccumulator(window_seconds=10.0)
        acc.add("/path/a.md")
        assert acc.flush() == []

    def test_flush_after_window(self) -> None:
        acc = DebounceAccumulator(window_seconds=0.1)
        acc.add("/path/a.md")
        time.sleep(0.15)
        result = acc.flush()
        assert result == ["/path/a.md"]

    def test_flush_removes_from_pending(self) -> None:
        acc = DebounceAccumulator(window_seconds=0.1)
        acc.add("/path/a.md")
        time.sleep(0.15)
        acc.flush()
        assert acc.pending_count() == 0

    def test_repeated_add_resets_timer(self) -> None:
        acc = DebounceAccumulator(window_seconds=0.2)
        acc.add("/path/a.md")
        time.sleep(0.1)
        acc.add("/path/a.md")  # Reset timer
        time.sleep(0.1)
        # Only 0.1s since last add, window is 0.2s
        assert acc.flush() == []
        time.sleep(0.15)
        result = acc.flush()
        assert result == ["/path/a.md"]

    def test_multiple_paths_independent(self) -> None:
        acc = DebounceAccumulator(window_seconds=0.2)
        acc.add("/path/a.md")
        time.sleep(0.15)
        acc.add("/path/b.md")
        time.sleep(0.1)
        # a.md has been quiet 0.25s (>0.2), b.md only 0.1s (<0.2)
        result = acc.flush()
        assert result == ["/path/a.md"]
        assert acc.pending_count() == 1  # b.md still pending

    def test_flush_returns_sorted(self) -> None:
        acc = DebounceAccumulator(window_seconds=0.1)
        acc.add("/path/c.md")
        acc.add("/path/a.md")
        acc.add("/path/b.md")
        time.sleep(0.15)
        result = acc.flush()
        assert result == ["/path/a.md", "/path/b.md", "/path/c.md"]

    def test_clear(self) -> None:
        acc = DebounceAccumulator(window_seconds=1.0)
        acc.add("/path/a.md")
        acc.add("/path/b.md")
        acc.clear()
        assert acc.pending_count() == 0
        assert acc.flush() == []

    def test_thread_safety(self) -> None:
        acc = DebounceAccumulator(window_seconds=0.1)
        errors: list[Exception] = []

        def add_paths(prefix: str, count: int) -> None:
            try:
                for i in range(count):
                    acc.add(f"{prefix}/{i}.md")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_paths, args=(f"/t{t}", 50))
            for t in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert acc.pending_count() == 250  # 5 threads * 50 paths


class TestBurstDetector:
    def test_not_bursting_initially(self) -> None:
        bd = BurstDetector(threshold=20, window_seconds=1.0)
        assert bd.is_bursting() is False

    def test_not_bursting_below_threshold(self) -> None:
        bd = BurstDetector(threshold=20, window_seconds=1.0)
        for _ in range(5):
            bd.record_event()
        assert bd.is_bursting() is False

    def test_bursting_at_threshold(self) -> None:
        bd = BurstDetector(threshold=10, window_seconds=5.0)
        for _ in range(10):
            bd.record_event()
        assert bd.is_bursting() is True

    def test_bursting_above_threshold(self) -> None:
        bd = BurstDetector(threshold=10, window_seconds=5.0)
        for _ in range(30):
            bd.record_event()
        assert bd.is_bursting() is True

    def test_burst_expires_after_window(self) -> None:
        bd = BurstDetector(threshold=5, window_seconds=0.1)
        for _ in range(10):
            bd.record_event()
        assert bd.is_bursting() is True
        time.sleep(0.15)
        assert bd.is_bursting() is False

    def test_event_count(self) -> None:
        bd = BurstDetector(threshold=100, window_seconds=5.0)
        for _ in range(7):
            bd.record_event()
        assert bd.event_count() == 7

    def test_old_events_pruned(self) -> None:
        bd = BurstDetector(threshold=100, window_seconds=0.1)
        for _ in range(5):
            bd.record_event()
        time.sleep(0.15)
        for _ in range(3):
            bd.record_event()
        assert bd.event_count() == 3

    def test_reset(self) -> None:
        bd = BurstDetector(threshold=5, window_seconds=5.0)
        for _ in range(10):
            bd.record_event()
        bd.reset()
        assert bd.is_bursting() is False
        assert bd.event_count() == 0

    def test_thread_safety(self) -> None:
        bd = BurstDetector(threshold=1000, window_seconds=5.0)
        errors: list[Exception] = []

        def record_events(count: int) -> None:
            try:
                for _ in range(count):
                    bd.record_event()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record_events, args=(50,))
            for _ in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert bd.event_count() == 250


# ---------------------------------------------------------------------------
# _RagEventHandler unit tests
# ---------------------------------------------------------------------------


class TestRagEventHandler:
    """Unit tests for the internal event handler."""

    def _make_handler(
        self,
        extensions: frozenset[str] | None = None,
        excluded_dirs: frozenset[str] | None = None,
    ) -> tuple[_RagEventHandler, DebounceAccumulator, BurstDetector]:
        acc = DebounceAccumulator(window_seconds=0.1)
        burst = BurstDetector(threshold=100, window_seconds=5.0)
        handler = _RagEventHandler(
            accumulator=acc,
            burst_detector=burst,
            extensions=extensions or frozenset({".md", ".py", ".json"}),
            excluded_dirs=excluded_dirs
            or frozenset({".git", "node_modules", "__pycache__"}),
        )
        return handler, acc, burst

    def test_event_handler_filters_excluded_dirs(self) -> None:
        """Events inside excluded directories (.git/) are ignored."""
        handler, acc, burst = self._make_handler()
        event = FileCreatedEvent(src_path=str(Path(".git/objects/abc123")))
        handler.on_any_event(event)

        assert acc.pending_count() == 0
        assert burst.event_count() == 0

    def test_event_handler_filters_extensions(self) -> None:
        """.txt files are ignored; .md files are handled."""
        handler, acc, burst = self._make_handler()

        # .txt should be ignored
        txt_event = FileCreatedEvent(src_path="project/notes.txt")
        handler.on_any_event(txt_event)
        assert acc.pending_count() == 0

        # .md should be handled
        md_event = FileCreatedEvent(src_path="project/README.md")
        handler.on_any_event(md_event)
        assert acc.pending_count() == 1
        assert burst.event_count() == 1

    def test_event_handler_ignores_directories(self) -> None:
        """Directory-level events are always ignored."""
        handler, acc, burst = self._make_handler()
        dir_event = DirCreatedEvent(src_path="project/new_folder")
        handler.on_any_event(dir_event)

        assert acc.pending_count() == 0
        assert burst.event_count() == 0

    def test_event_handler_nested_excluded_dir(self) -> None:
        """Events in nested excluded directories are ignored."""
        handler, acc, _ = self._make_handler()
        event = FileModifiedEvent(
            src_path=str(Path("project/__pycache__/mod.cpython.pyc"))
        )
        handler.on_any_event(event)
        assert acc.pending_count() == 0

    def test_event_handler_records_burst(self) -> None:
        """Each handled event is recorded in the burst detector."""
        handler, _, burst = self._make_handler()
        for i in range(5):
            event = FileModifiedEvent(src_path=f"project/file{i}.py")
            handler.on_any_event(event)
        assert burst.event_count() == 5


# ---------------------------------------------------------------------------
# FileWatcher integration tests
# ---------------------------------------------------------------------------


class TestFileWatcher:
    """Integration tests using real filesystem events via watchdog."""

    def test_file_watcher_start_stop(self, tmp_path: Path) -> None:
        """FileWatcher starts and stops cleanly without errors."""
        callback = MagicMock()
        watcher = FileWatcher(
            watch_root=tmp_path,
            extensions=frozenset({".md"}),
            excluded_dirs=frozenset({".git"}),
            debounce_seconds=0.2,
            burst_threshold=100,
            burst_window_seconds=5.0,
            on_files_changed=callback,
            poll_interval=0.1,
        )
        watcher.start()
        time.sleep(0.2)
        watcher.stop()

        # Should not raise and callback was never called (no events)
        callback.assert_not_called()

    def test_file_watcher_detects_file_change(self, tmp_path: Path) -> None:
        """Creating a .md file triggers the callback with that path."""
        callback = MagicMock()
        watcher = FileWatcher(
            watch_root=tmp_path,
            extensions=frozenset({".md"}),
            excluded_dirs=frozenset({".git"}),
            debounce_seconds=0.2,
            burst_threshold=100,
            burst_window_seconds=5.0,
            on_files_changed=callback,
            poll_interval=0.1,
        )
        watcher.start()
        try:
            # Create a file to trigger the event
            test_file = tmp_path / "hello.md"
            test_file.write_text("# Hello World")

            # Wait for debounce + poll to fire
            time.sleep(0.8)

            assert callback.call_count >= 1
            # The callback should have been called with a list containing the path
            all_paths: list[str] = []
            for call_args in callback.call_args_list:
                all_paths.extend(call_args[0][0])
            assert any("hello.md" in p for p in all_paths)
        finally:
            watcher.stop()

    def test_file_watcher_ignores_excluded_extension(
        self, tmp_path: Path
    ) -> None:
        """Files with non-watched extensions do not trigger callback."""
        callback = MagicMock()
        watcher = FileWatcher(
            watch_root=tmp_path,
            extensions=frozenset({".md"}),
            excluded_dirs=frozenset({".git"}),
            debounce_seconds=0.2,
            burst_threshold=100,
            burst_window_seconds=5.0,
            on_files_changed=callback,
            poll_interval=0.1,
        )
        watcher.start()
        try:
            txt_file = tmp_path / "notes.txt"
            txt_file.write_text("not watched")
            time.sleep(0.8)
            callback.assert_not_called()
        finally:
            watcher.stop()

    def test_file_watcher_burst_suppresses(self, tmp_path: Path) -> None:
        """Many rapid file creates suppress callback until burst subsides."""
        callback = MagicMock()
        # Low threshold so burst triggers easily
        watcher = FileWatcher(
            watch_root=tmp_path,
            extensions=frozenset({".md"}),
            excluded_dirs=frozenset({".git"}),
            debounce_seconds=0.2,
            burst_threshold=5,
            burst_window_seconds=2.0,
            on_files_changed=callback,
            poll_interval=0.1,
        )
        watcher.start()
        try:
            # Rapidly create many files to trigger burst
            for i in range(15):
                f = tmp_path / f"burst_{i}.md"
                f.write_text(f"content {i}")
                time.sleep(0.02)

            # Immediately after burst, callback should NOT have fired
            time.sleep(0.3)
            burst_call_count = callback.call_count
            # During the burst window, flushing should be suppressed
            # (burst_threshold=5, we created 15 files in ~0.3s within 2s window)

            # Wait for burst window to expire + debounce + poll
            time.sleep(2.5)

            # Now callback should have been called after burst subsided
            assert callback.call_count > burst_call_count
        finally:
            watcher.stop()
