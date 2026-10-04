"""News ingestion: pluggable sources, credibility tags, de-duplication into event clusters.

SECURITY: everything fetched here is untrusted *data*. Text is sanitised, length-limited,
stored and displayed. Nothing in it is ever executed, evaluated, or interpreted as an
instruction, and in this version it does not feed the trading model at all (context only).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from email.utils import parsedate_to_datetime
from typing import Any, Protocol
from xml.etree import ElementTree as ET  # noqa: S405 - expat >= 2.4 mitigates entity expansion; size capped

import aiohttp

log = logging.getLogger(__name__)
_MAX_BYTES = 2_000_000
_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset("the a an of to in on for and or is are as at by with from after says say".split())


@dataclass(slots=True)
class NewsItem:
    source: str
    credibility: float  # 0..1, assigned to the source by the operator
    published_ts: float
    ingested_ts: float
    title: str
    url: str


@dataclass(slots=True)
class Cluster:
    first: NewsItem  # earliest credible report
    sources: set[str] = field(default_factory=set)
    tokens: frozenset[str] = frozenset()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self.first) | {"confirmations": len(self.sources), "sources": sorted(self.sources)}


class NewsSource(Protocol):
    name: str

    async def fetch(self, session: aiohttp.ClientSession) -> list[NewsItem]: ...


def sanitize(text: str, limit: int = 300) -> str:
    return re.sub(r"\s+", " ", "".join(c for c in text if c.isprintable())).strip()[:limit]


def tokens(title: str) -> frozenset[str]:
    return frozenset(w for w in _WORD.findall(title.lower()) if w not in _STOP)


def parse_feed(xml: bytes, name: str, credibility: float, now: float) -> list[NewsItem]:
    root = ET.fromstring(xml)  # noqa: S314
    out = []
    for el in root.iter():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag not in ("item", "entry"):
            continue
        kids = {c.tag.rsplit("}", 1)[-1]: c for c in el}
        title = sanitize(kids["title"].text or "") if "title" in kids else ""
        if not title:
            continue
        link = kids.get("link")
        url = sanitize((link.text or link.get("href") or "") if link is not None else "", 500)
        ts = now
        for k in ("pubDate", "published", "updated"):
            if k in kids and kids[k].text:
                try:
                    raw = kids[k].text or ""
                    ts = parsedate_to_datetime(raw).timestamp() if k == "pubDate" else _iso(raw)
                    break
                except (ValueError, TypeError):
                    pass
        out.append(NewsItem(name, credibility, ts, now, title, url))
    return out


def _iso(s: str) -> float:
    from datetime import datetime

    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


class RssSource:
    def __init__(self, url: str, credibility: float = 0.5) -> None:
        self.url = url
        self.name = re.sub(r"^www\.", "", url.split("/")[2])
        self.credibility = credibility

    async def fetch(self, session: aiohttp.ClientSession) -> list[NewsItem]:
        async with session.get(self.url, headers={"User-Agent": "hl-trader-news/0.1"}) as r:
            r.raise_for_status()
            body = b""
            async for chunk in r.content.iter_chunked(65536):
                body += chunk
                if len(body) > _MAX_BYTES:
                    raise ValueError("feed too large")
        return parse_feed(body, self.name, self.credibility, time.time())


class NewsMonitor:
    def __init__(self, sources: list[NewsSource], poll_s: float = 60.0, similarity: float = 0.6) -> None:
        self.sources = sources
        self.poll_s = poll_s
        self.similarity = similarity
        self.clusters: deque[Cluster] = deque(maxlen=300)
        self._seen: set[tuple[str, str]] = set()
        self.last_ok: dict[str, float] = {}
        self.errors: dict[str, int] = {}

    def add(self, item: NewsItem) -> None:
        key = (item.source, item.title)
        if key in self._seen:
            return
        self._seen.add(key)
        tk = tokens(item.title)
        for c in self.clusters:
            union = len(tk | c.tokens)
            if union and len(tk & c.tokens) / union >= self.similarity:
                c.sources.add(item.source)
                if item.published_ts < c.first.published_ts:
                    c.first = item
                return
        self.clusters.append(Cluster(item, {item.source}, tk))

    async def poll_once(self) -> None:
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for src in self.sources:
                try:
                    for it in await src.fetch(session):
                        self.add(it)
                    self.last_ok[src.name] = time.time()
                except Exception as e:  # noqa: BLE001 - one bad feed must not affect the others
                    self.errors[src.name] = self.errors.get(src.name, 0) + 1
                    log.warning("news source %s failed: %s", src.name, e)

    async def run(self) -> None:
        while True:
            await self.poll_once()
            await asyncio.sleep(self.poll_s)

    def snapshot(self, limit: int = 15) -> dict[str, Any]:
        latest = sorted(self.clusters, key=lambda c: c.first.published_ts, reverse=True)[:limit]
        return {
            "clusters": len(self.clusters),
            "sources_ok": self.last_ok,
            "source_errors": self.errors,
            "latest": [c.to_dict() for c in latest],
        }
