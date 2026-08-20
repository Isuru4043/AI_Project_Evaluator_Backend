"""Small thread-safe TTL/LRU caches for process-local RAG acceleration."""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from typing import Callable, Generic, Hashable, Optional, Tuple, TypeVar


Key = TypeVar("Key", bound=Hashable)
Value = TypeVar("Value")


def env_int(name: str, default: int, *, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return max(minimum, default)


def env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return max(minimum, default)


class BoundedTTLCache(Generic[Key, Value]):
    """A bounded least-recently-used cache with monotonic expiry."""

    def __init__(self, *, max_entries: int, ttl_seconds: float):
        self.max_entries = max(0, int(max_entries))
        self.ttl_seconds = max(0.0, float(ttl_seconds))
        self._items: OrderedDict[Key, Tuple[float, Value]] = OrderedDict()
        self._lock = threading.RLock()

    @property
    def enabled(self) -> bool:
        return self.max_entries > 0 and self.ttl_seconds > 0

    def get(self, key: Key) -> Tuple[bool, Optional[Value]]:
        if not self.enabled:
            return False, None
        now = time.monotonic()
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return False, None
            expires_at, value = item
            if expires_at <= now:
                self._items.pop(key, None)
                return False, None
            self._items.move_to_end(key)
            return True, value

    def set(self, key: Key, value: Value) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._items[key] = (time.monotonic() + self.ttl_seconds, value)
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def invalidate(self, predicate: Optional[Callable[[Key], bool]] = None) -> None:
        with self._lock:
            if predicate is None:
                self._items.clear()
                return
            for key in list(self._items):
                if predicate(key):
                    self._items.pop(key, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
