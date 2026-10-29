"""In-process adapters: a bounded TTL/LRU answer cache and a JSONL trace sink.

In-process is enough for a single server, and losing the cache on restart is harmless. With
several replicas, swap in a Redis AnswerCache (same interface and keys) so they share hits.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class MemoryAnswerCache:
    def __init__(self, *, max_entries: int, ttl_s: int) -> None:
        self._data: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._max = max_entries
        self._ttl = ttl_s
        self._lock = threading.Lock()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            stored_at, value = item
            if time.monotonic() - stored_at > self._ttl:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._data[key] = (time.monotonic(), value)
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)


class NullAnswerCache:
    def get(self, key: str) -> dict[str, Any] | None:
        return None

    def set(self, key: str, value: dict[str, Any]) -> None:
        return None


class JsonlTraceSink:
    """Appends one JSON record per request. Enough to debug any answer after the fact."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._lock = threading.Lock()

    def emit(self, record: dict[str, Any]) -> None:
        # Observability must never fail a request that already has an answer.
        try:
            line = json.dumps(record, default=str)
            with self._lock, self._path.open("a") as fh:
                fh.write(line + "\n")
        except (OSError, TypeError, ValueError) as exc:
            log.warning("trace write failed (%s); request unaffected", exc)
