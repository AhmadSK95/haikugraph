"""Semantic profile cache keyed by dataset signature for v2 runtime."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Callable

from haikugraph.v2.types import SemanticCatalogV2


def _fingerprint(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return int(stat.st_size), int(stat.st_mtime_ns)


@dataclass
class _SemanticCacheEntry:
    dataset_signature: str
    schema_signature: str
    path_key: str
    file_size: int
    file_mtime_ns: int
    cached_at_epoch_s: float
    profile: SemanticCatalogV2


class SemanticProfileCache:
    """Thread-safe cache for semantic catalogs keyed by dataset signature."""

    def __init__(self, *, max_entries: int = 24, ttl_seconds: int = 900):
        self.max_entries = max(4, int(max_entries))
        self.ttl_seconds = max(30, int(ttl_seconds))
        self._entries: OrderedDict[str, _SemanticCacheEntry] = OrderedDict()
        self._path_index: dict[str, str] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _path_key(db_path: str | Path) -> str:
        return str(Path(db_path).expanduser().resolve())

    def _evict_if_needed(self) -> None:
        while len(self._entries) > self.max_entries:
            sig, removed = self._entries.popitem(last=False)
            if self._path_index.get(removed.path_key) == sig:
                self._path_index.pop(removed.path_key, None)

    def invalidate_path(self, db_path: str | Path) -> bool:
        path_key = self._path_key(db_path)
        with self._lock:
            signature = self._path_index.pop(path_key, None)
            if not signature:
                return False
            self._entries.pop(signature, None)
            return True

    def get_or_build(
        self,
        db_path: str | Path,
        builder: Callable[[str], SemanticCatalogV2],
    ) -> tuple[SemanticCatalogV2, dict[str, object]]:
        path = Path(db_path).expanduser().resolve()
        path_key = str(path)
        size, mtime_ns = _fingerprint(path)
        now = time.time()

        with self._lock:
            existing_sig = self._path_index.get(path_key)
            if existing_sig:
                entry = self._entries.get(existing_sig)
                if entry is not None:
                    age = now - float(entry.cached_at_epoch_s)
                    same_file = entry.file_size == size and entry.file_mtime_ns == mtime_ns
                    if same_file and age <= self.ttl_seconds:
                        self._entries.move_to_end(existing_sig)
                        return entry.profile, {
                            "cache_hit": True,
                            "cache_key": entry.dataset_signature,
                            "cache_age_s": round(age, 2),
                            "dataset_signature": entry.dataset_signature,
                            "schema_signature": entry.schema_signature,
                        }
                    self._entries.pop(existing_sig, None)
            self._path_index.pop(path_key, None)

        profile = builder(str(path))
        dataset_signature = str(profile.dataset_signature or "")
        schema_signature = str(profile.schema_signature or "")
        if not dataset_signature:
            # Defensive fallback; keeps cache key stable even if profile does not set a signature.
            dataset_signature = f"nosig:{path_key}:{size}:{mtime_ns}"

        entry = _SemanticCacheEntry(
            dataset_signature=dataset_signature,
            schema_signature=schema_signature,
            path_key=path_key,
            file_size=size,
            file_mtime_ns=mtime_ns,
            cached_at_epoch_s=now,
            profile=profile,
        )
        with self._lock:
            self._entries[dataset_signature] = entry
            self._entries.move_to_end(dataset_signature)
            self._path_index[path_key] = dataset_signature
            self._evict_if_needed()
        return profile, {
            "cache_hit": False,
            "cache_key": dataset_signature,
            "cache_age_s": 0.0,
            "dataset_signature": dataset_signature,
            "schema_signature": schema_signature,
        }

