from __future__ import annotations

import json
import os
import re
import time
import uuid
from datetime import UTC, datetime
from typing import Protocol

try:
    import psycopg
except ImportError:  # pragma: no cover - optional dependency
    psycopg = None

try:
    from redis import Redis
    from redis.exceptions import RedisError
except ImportError:  # pragma: no cover - optional dependency
    Redis = None

    class RedisError(Exception):
        pass

from app.models.context import ContextMemoryRecord, HotContextSnapshot
from app.services.context_embeddings import cosine_similarity, embed_text, embedding_dimensions


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_./:-]+")


def _token_set(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


class DurableContextBackend(Protocol):
    def upsert_memories(self, records: list[ContextMemoryRecord]) -> None: ...

    def search(
        self,
        thread_id: str,
        query: str,
        kinds: list[str] | None,
        file_paths: list[str] | None,
        symbols: list[str] | None,
        top_k: int,
    ) -> list[dict]: ...

    def get_by_ids(self, thread_id: str, ids: list[str]) -> list[ContextMemoryRecord]: ...

    def recent(self, thread_id: str, limit: int) -> list[ContextMemoryRecord]: ...


class HotContextBackend(Protocol):
    def get_snapshot(self, thread_id: str) -> HotContextSnapshot | None: ...

    def set_snapshot(self, snapshot: HotContextSnapshot, ttl_seconds: int) -> None: ...


class InMemoryDurableContextStore:
    def __init__(self) -> None:
        self._records: dict[str, list[ContextMemoryRecord]] = {}

    def upsert_memories(self, records: list[ContextMemoryRecord]) -> None:
        for record in records:
            bucket = self._records.setdefault(record.thread_id, [])
            for idx, existing in enumerate(bucket):
                if existing.memory_id == record.memory_id:
                    bucket[idx] = record
                    break
            else:
                bucket.append(record)

    def search(
        self,
        thread_id: str,
        query: str,
        kinds: list[str] | None,
        file_paths: list[str] | None,
        symbols: list[str] | None,
        top_k: int,
    ) -> list[dict]:
        records = self._records.get(thread_id, [])
        query_embedding = embed_text(query)
        results = []
        for record in records:
            if kinds and record.kind not in kinds:
                continue
            if file_paths and not any(path in record.file_paths for path in file_paths):
                continue
            if symbols and not any(symbol in record.symbol_names for symbol in symbols):
                continue

            score = cosine_similarity(query_embedding, record.embedding)
            lowered_query = query.lower()
            query_tokens = _token_set(query)
            record_tokens = _token_set(f"{record.title}\n{record.summary}\n{record.content}")
            if query_tokens and record_tokens:
                overlap = len(query_tokens & record_tokens) / max(1, len(query_tokens))
                score += overlap * 0.45
            if lowered_query and lowered_query in record.summary.lower():
                score += 0.35
            if lowered_query and lowered_query in record.title.lower():
                score += 0.2
            score += record.importance * 0.1 + record.recency_score * 0.05
            results.append({"record": record, "score": score})

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]

    def get_by_ids(self, thread_id: str, ids: list[str]) -> list[ContextMemoryRecord]:
        wanted = set(ids)
        return [record for record in self._records.get(thread_id, []) if record.memory_id in wanted]

    def recent(self, thread_id: str, limit: int) -> list[ContextMemoryRecord]:
        records = list(self._records.get(thread_id, []))
        records.sort(key=lambda item: item.updated_at, reverse=True)
        return records[:limit]


class InMemoryHotContextStore:
    def __init__(self) -> None:
        self._snapshots: dict[str, tuple[HotContextSnapshot, float]] = {}

    def get_snapshot(self, thread_id: str) -> HotContextSnapshot | None:
        cached = self._snapshots.get(thread_id)
        if not cached:
            return None
        snapshot, expires_at = cached
        if time.time() > expires_at:
            self._snapshots.pop(thread_id, None)
            return None
        return snapshot

    def set_snapshot(self, snapshot: HotContextSnapshot, ttl_seconds: int) -> None:
        self._snapshots[snapshot.thread_id] = (snapshot, time.time() + ttl_seconds)


class RedisHotContextStore:
    def __init__(self, redis_url: str, key_prefix: str) -> None:
        if Redis is None:
            raise RedisError("redis package is not installed")
        self._client = Redis.from_url(redis_url, decode_responses=True)
        self._key_prefix = key_prefix

    def _key(self, thread_id: str) -> str:
        return f"{self._key_prefix}{thread_id}"

    def get_snapshot(self, thread_id: str) -> HotContextSnapshot | None:
        payload = self._client.get(self._key(thread_id))
        if not payload:
            return None
        return HotContextSnapshot.model_validate_json(payload)

    def set_snapshot(self, snapshot: HotContextSnapshot, ttl_seconds: int) -> None:
        self._client.set(self._key(snapshot.thread_id), snapshot.model_dump_json(), ex=ttl_seconds)


class PostgresDurableContextStore:
    def __init__(self, dsn: str) -> None:
        if psycopg is None:
            raise RuntimeError("psycopg is not installed")
        self._dsn = dsn
        self._ensure_schema()

    def _connect(self):
        return psycopg.connect(self._dsn)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS context_memories (
                        thread_id TEXT NOT NULL,
                        memory_id TEXT PRIMARY KEY,
                        kind TEXT NOT NULL,
                        title TEXT NOT NULL,
                        content TEXT NOT NULL,
                        summary TEXT NOT NULL,
                        source_turn_range TEXT NULL,
                        source_tool TEXT NULL,
                        file_paths JSONB NOT NULL,
                        symbol_names JSONB NOT NULL,
                        importance DOUBLE PRECISION NOT NULL,
                        recency_score DOUBLE PRECISION NOT NULL,
                        embedding vector(%s) NOT NULL,
                        metadata JSONB NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL
                    )
                    """,
                    (embedding_dimensions(),),
                )
            conn.commit()

    def upsert_memories(self, records: list[ContextMemoryRecord]) -> None:
        if not records:
            return
        with self._connect() as conn:
            with conn.cursor() as cur:
                for record in records:
                    cur.execute(
                        """
                        INSERT INTO context_memories (
                            thread_id, memory_id, kind, title, content, summary, source_turn_range, source_tool,
                            file_paths, symbol_names, importance, recency_score, embedding, metadata, created_at, updated_at
                        ) VALUES (
                            %(thread_id)s, %(memory_id)s, %(kind)s, %(title)s, %(content)s, %(summary)s,
                            %(source_turn_range)s, %(source_tool)s, %(file_paths)s, %(symbol_names)s,
                            %(importance)s, %(recency_score)s, %(embedding)s, %(metadata)s, %(created_at)s, %(updated_at)s
                        )
                        ON CONFLICT (memory_id) DO UPDATE SET
                            kind = EXCLUDED.kind,
                            title = EXCLUDED.title,
                            content = EXCLUDED.content,
                            summary = EXCLUDED.summary,
                            source_turn_range = EXCLUDED.source_turn_range,
                            source_tool = EXCLUDED.source_tool,
                            file_paths = EXCLUDED.file_paths,
                            symbol_names = EXCLUDED.symbol_names,
                            importance = EXCLUDED.importance,
                            recency_score = EXCLUDED.recency_score,
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata,
                            updated_at = EXCLUDED.updated_at
                        """,
                        {
                            **record.model_dump(),
                            "file_paths": json.dumps(record.file_paths),
                            "symbol_names": json.dumps(record.symbol_names),
                            "metadata": json.dumps(record.metadata),
                            "embedding": record.embedding,
                        },
                    )
            conn.commit()

    def _load_records(self, query: str, params: dict) -> list[ContextMemoryRecord]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
        records = []
        for row in rows:
            records.append(
                ContextMemoryRecord(
                    thread_id=row[0],
                    memory_id=row[1],
                    kind=row[2],
                    title=row[3],
                    content=row[4],
                    summary=row[5],
                    source_turn_range=row[6],
                    source_tool=row[7],
                    file_paths=row[8],
                    symbol_names=row[9],
                    importance=row[10],
                    recency_score=row[11],
                    embedding=list(row[12]),
                    metadata=row[13],
                    created_at=row[14],
                    updated_at=row[15],
                )
            )
        return records

    def search(
        self,
        thread_id: str,
        query: str,
        kinds: list[str] | None,
        file_paths: list[str] | None,
        symbols: list[str] | None,
        top_k: int,
    ) -> list[dict]:
        records = self.recent(thread_id, 200)
        query_embedding = embed_text(query)
        query_tokens = _token_set(query)
        scored = []
        for record in records:
            if kinds and record.kind not in kinds:
                continue
            if file_paths and not any(path in record.file_paths for path in file_paths):
                continue
            if symbols and not any(symbol in record.symbol_names for symbol in symbols):
                continue
            score = cosine_similarity(query_embedding, record.embedding)
            record_tokens = _token_set(f"{record.title}\n{record.summary}\n{record.content}")
            if query_tokens and record_tokens:
                score += (len(query_tokens & record_tokens) / max(1, len(query_tokens))) * 0.45
            scored.append({"record": record, "score": score})
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def get_by_ids(self, thread_id: str, ids: list[str]) -> list[ContextMemoryRecord]:
        records = self.recent(thread_id, 500)
        wanted = set(ids)
        return [record for record in records if record.memory_id in wanted]

    def recent(self, thread_id: str, limit: int) -> list[ContextMemoryRecord]:
        return self._load_records(
            """
            SELECT thread_id, memory_id, kind, title, content, summary, source_turn_range, source_tool,
                   file_paths, symbol_names, importance, recency_score, embedding, metadata, created_at, updated_at
            FROM context_memories
            WHERE thread_id = %(thread_id)s
            ORDER BY updated_at DESC
            LIMIT %(limit)s
            """,
            {"thread_id": thread_id, "limit": limit},
        )


class ContextStore:
    def __init__(self) -> None:
        self._durable = self._build_durable_backend()
        self._hot = self._build_hot_backend()

    def _build_durable_backend(self) -> DurableContextBackend:
        dsn = os.getenv("TUBS_CONTEXT_DATABASE_URL", "").strip()
        if dsn and psycopg is not None:
            try:
                return PostgresDurableContextStore(dsn)
            except Exception:
                pass
        return InMemoryDurableContextStore()

    def _build_hot_backend(self) -> HotContextBackend:
        backend_name = os.getenv("TUBS_CONTEXT_HOT_BACKEND", "redis").strip().lower()
        redis_url = os.getenv("REDIS_URL", "").strip()
        key_prefix = os.getenv("TUBS_CONTEXT_HOT_PREFIX", "tubs:context:hot:").strip() or "tubs:context:hot:"
        if backend_name == "redis" and redis_url and Redis is not None:
            try:
                backend = RedisHotContextStore(redis_url, key_prefix)
                backend._client.ping()
                return backend
            except RedisError:
                pass
        return InMemoryHotContextStore()

    def new_memory(
        self,
        *,
        thread_id: str,
        kind: str,
        title: str,
        content: str,
        summary: str,
        source_turn_range: str | None = None,
        source_tool: str | None = None,
        file_paths: list[str] | None = None,
        symbol_names: list[str] | None = None,
        importance: float = 0.5,
        recency_score: float = 0.5,
        metadata: dict | None = None,
    ) -> ContextMemoryRecord:
        now = datetime.now(UTC)
        return ContextMemoryRecord(
            thread_id=thread_id,
            memory_id=f"mem_{uuid.uuid4().hex}",
            kind=kind,  # type: ignore[arg-type]
            title=title,
            content=content,
            summary=summary,
            source_turn_range=source_turn_range,
            source_tool=source_tool,
            file_paths=file_paths or [],
            symbol_names=symbol_names or [],
            importance=importance,
            recency_score=recency_score,
            embedding=embed_text(f"{title}\n{summary}\n{content}"),
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
        )

    def upsert_memories(self, records: list[ContextMemoryRecord]) -> None:
        self._durable.upsert_memories(records)

    def recent(self, thread_id: str, limit: int = 50) -> list[ContextMemoryRecord]:
        return self._durable.recent(thread_id, limit)

    def search(
        self,
        thread_id: str,
        query: str,
        kinds: list[str] | None = None,
        file_paths: list[str] | None = None,
        symbols: list[str] | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        return self._durable.search(thread_id, query, kinds, file_paths, symbols, top_k)

    def get_by_ids(self, thread_id: str, ids: list[str]) -> list[ContextMemoryRecord]:
        return self._durable.get_by_ids(thread_id, ids)

    def get_hot_snapshot(self, thread_id: str) -> HotContextSnapshot | None:
        return self._hot.get_snapshot(thread_id)

    def set_hot_snapshot(self, snapshot: HotContextSnapshot) -> None:
        ttl = max(60, int(os.getenv("TUBS_CONTEXT_HOT_TTL_SECONDS", "21600")))
        self._hot.set_snapshot(snapshot, ttl)


_STORE: ContextStore | None = None


def context_store() -> ContextStore:
    global _STORE
    if _STORE is None:
        _STORE = ContextStore()
    return _STORE


def reset_context_store() -> None:
    global _STORE
    _STORE = None
