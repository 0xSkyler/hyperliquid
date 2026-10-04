"""Persistence off the hot path: the engine enqueues, a background thread writes.

The queue is bounded; if the writer falls behind, records are dropped and counted rather
than ever blocking a trading decision.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import IO, Any

log = logging.getLogger(__name__)


class JsonlSink:
    """One append-only JSON-lines file per stream under `data_dir`."""

    def __init__(self, data_dir: str, maxsize: int = 20000) -> None:
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dropped = 0
        self._q: queue.Queue[tuple[str, Any] | None] = queue.Queue(maxsize)
        self._t = threading.Thread(target=self._run, name="sink", daemon=True)
        self._t.start()

    def put(self, stream: str, obj: Any) -> None:
        try:
            self._q.put_nowait((stream, obj))
        except queue.Full:
            self.dropped += 1

    def close(self) -> None:
        self._q.put(None)
        self._t.join(timeout=5)

    def _write(self, files: dict[str, IO[str]], stream: str, obj: Any) -> None:
        f = files.get(stream)
        if f is None:
            f = files[stream] = open(self.dir / f"{stream}.jsonl", "a", encoding="utf-8")  # noqa: SIM115
        f.write(json.dumps(obj, default=float, separators=(",", ":")) + "\n")

    def _run(self) -> None:
        files: dict[str, IO[str]] = {}
        last_flush = time.monotonic()
        while True:
            try:
                item = self._q.get(timeout=1.0)
            except queue.Empty:
                item = ("", None)
            if item is None:
                break
            if item[0]:
                self._write(files, *item)
            if time.monotonic() - last_flush > 1.0:
                for f in files.values():
                    f.flush()
                last_flush = time.monotonic()
        for f in files.values():
            f.close()


class PostgresSink(JsonlSink):
    """Writes every record to Postgres `events` (see migrations/001_init.sql) instead of files.

    NOTE: not exercised in this repository's test suite (needs a running database).
    """

    def __init__(self, dsn: str, maxsize: int = 20000) -> None:
        self._dsn = dsn
        super().__init__(".", maxsize)

    def _run(self) -> None:
        import asyncio

        import asyncpg

        async def main() -> None:
            conn = await asyncpg.connect(self._dsn)
            try:
                while True:
                    item = await asyncio.to_thread(self._q.get)
                    if item is None:
                        return
                    batch = [item]
                    while len(batch) < 500:
                        try:
                            nxt = self._q.get_nowait()
                        except queue.Empty:
                            break
                        if nxt is None:
                            self._q.put(None)
                            break
                        batch.append(nxt)
                    rows = [(time.time(), s, json.dumps(o, default=float)) for s, o in batch]
                    await conn.executemany(
                        "INSERT INTO events (ts, stream, payload) VALUES (to_timestamp($1), $2, $3::jsonb)", rows
                    )
            finally:
                await conn.close()

        try:
            asyncio.run(main())
        except Exception:  # noqa: BLE001
            log.exception("postgres sink stopped")
