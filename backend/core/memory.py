"""
Memory System — Vector database backed long/short-term memory with RAG.

Supports ChromaDB (default) with Elasticsearch as a future option.
Provides embedding-based storage and retrieval for:
  - Conversation history summarization
  - User facts & preferences
  - Knowledge base entries
  - Skill documentation

Retrieval is not pure top-k cosine similarity: candidates are re-ranked with
semantic relevance + recency (exponential decay) + importance (salience stored
per entry) + lexical overlap. This mirrors Generative-Agents-style scoring and
fixes classic top-k failure modes (stale facts outranking fresh ones,
shared-entity facts missed by embeddings alone) without a graph database.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from enum import Enum
from pathlib import Path
from typing import Any

from backend.config import MemoryConfig

logger = logging.getLogger("tamagi.memory")


class MemoryType(str, Enum):
    CONVERSATION = "conversation"
    FACT = "fact"
    KNOWLEDGE = "knowledge"
    SKILL = "skill"
    PREFERENCE = "preference"


# ── Retrieval scoring (memory v2) ─────────────────────────────
# Effective recall score = weighted sum of four 0..1 components.
# Relevance dominates; the rest break ties the way a human remembers:
# recent and important things surface first, and exact names/entities
# rescue items embeddings alone would miss.
_W_RELEVANCE = 0.55
_W_RECENCY = 0.25
_W_IMPORTANCE = 0.10
_W_LEXICAL = 0.10

# Memory half-life: a 30-day-old memory keeps half its recency score.
_RECENCY_HALF_LIFE_DAYS = 30.0

# Per-type salience defaults for entries stored before v2 (no explicit
# importance). Fed knowledge matters more than the average chat triple.
DEFAULT_IMPORTANCE: dict[MemoryType, float] = {
    MemoryType.KNOWLEDGE: 0.80,
    MemoryType.PREFERENCE: 0.75,
    MemoryType.FACT: 0.70,
    MemoryType.SKILL: 0.50,
    MemoryType.CONVERSATION: 0.45,
}

# Small EN+ES stopword set for the lexical component only.
_STOPWORDS = frozenset((
    "the a an and or of to in is are was were it this that for on with as at "
    "be by from what when how do does did you your yours i me my we our they "
    "their he she his her not can will just so if then about have has had "
    "que de la el los las un una y o en es son era como cuando donde quien "
    "por para con se me mi te tu nos su sus esta estan muy mas pero al del "
    "lo qué cómo cuándo dónde quién más también están sería haría "
    "también".split()))

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúüñ]+")


def _default_importance(memory_type: MemoryType) -> float:
    return DEFAULT_IMPORTANCE.get(memory_type, 0.5)


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokens (accents kept), short words and stopwords dropped."""
    return {t for t in _TOKEN_RE.findall(text.lower())
            if len(t) >= 3 and t not in _STOPWORDS}


def _recency_factor(timestamp: float, now: float) -> float:
    """Exponential recency decay in [0, 1]; unknown timestamps score 0."""
    if timestamp <= 0:
        return 0.0
    age_days = max(0.0, (now - timestamp) / 86400.0)
    return math.pow(0.5, age_days / _RECENCY_HALF_LIFE_DAYS)


def _lexical_overlap(query_tokens: set[str], doc_tokens: set[str]) -> float:
    """Fraction of query tokens literally present in the memory."""
    if not query_tokens:
        return 0.0
    return len(query_tokens & doc_tokens) / len(query_tokens)


def score_memories(
    entries: list["MemoryEntry"],
    query: str,
    now: float | None = None,
) -> list["MemoryEntry"]:
    """Re-rank entries by relevance/recency/importance/lexical blend. Pure."""
    now = now if now is not None else time.time()
    q_tokens = _tokenize(query)
    scored = []
    for e in entries:
        recency = _recency_factor(e.timestamp, now)
        importance = (e.importance if e.importance is not None
                      else _default_importance(e.memory_type))
        lexical = _lexical_overlap(q_tokens, _tokenize(e.content))
        score = (_W_RELEVANCE * max(0.0, min(1.0, e.relevance))
                 + _W_RECENCY * recency
                 + _W_IMPORTANCE * max(0.0, min(1.0, importance))
                 + _W_LEXICAL * lexical)
        scored.append((score, e))
    scored.sort(key=lambda p: p[0], reverse=True)
    return [e for _, e in scored]


class MemoryEntry:
    """A single memory entry."""

    def __init__(
        self,
        content: str,
        memory_type: MemoryType,
        metadata: dict[str, Any] | None = None,
        id: str | None = None,
        timestamp: float | None = None,
        relevance: float = 0.0,
        importance: float | None = None,
    ):
        self.content = content
        self.memory_type = memory_type
        self.metadata = metadata or {}
        self.id = id or self._generate_id(content)
        # `or time.time()` used to turn an explicit 0 (pre-v2 entry with no
        # stored timestamp) into "now", making stale memories score max
        # recency. Only a missing timestamp means "now".
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.relevance = relevance
        # Salience 0..1 used by recall re-ranking. None → per-type default
        # from DEFAULT_IMPORTANCE (resolved at store/score time, not here,
        # so a late-set type still picks the right default).
        self.importance = importance

    def _generate_id(self, content: str) -> str:
        h = hashlib.sha256(f"{content}{time.time()}".encode()).hexdigest()[:16]
        return f"mem_{h}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "relevance": self.relevance,
            "importance": (self.importance if self.importance is not None
                           else _default_importance(self.memory_type)),
        }


class MemoryStore:
    """
    ChromaDB-backed vector memory store.

    Provides:
    - store(): Add a memory with automatic embedding
    - recall(): Retrieve relevant memories via semantic search
    - forget(): Remove specific memories
    - summarize_recent(): Get a digest of recent memories
    """

    def __init__(self, config: MemoryConfig):
        self.config = config
        self._collection = None
        self._client = None

    async def initialize(self) -> None:
        """Initialize the ChromaDB connection."""
        try:
            import chromadb
            from chromadb.config import Settings

            persist_dir = Path(self.config.chromadb.persist_directory)
            persist_dir.mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=Settings(anonymized_telemetry=False),
            )

            self._collection = self._client.get_or_create_collection(
                name=self.config.chromadb.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            count = self._collection.count()
            logger.info(
                f"Memory store initialized: {count} memories in "
                f"'{self.config.chromadb.collection_name}'"
            )

        except ImportError:
            logger.error(
                "❌ ChromaDB not installed. Memory will be in-memory only (no persistence). "
                "Install with: pip install chromadb. "
                "This severely limits memory recall effectiveness."
            )
            self._init_fallback()
        except Exception:
            # Corrupt index, locked persist dir, disk errors, ... — without this
            # the exception escaped lifespan and the whole server refused to
            # boot. Degrade to in-memory instead; the error stays in the log.
            logger.exception(
                "❌ ChromaDB initialization failed. Memory will be in-memory only "
                "(no persistence). Check data/chromadb/ for a corrupt index."
            )
            self._init_fallback()

    def _init_fallback(self) -> None:
        """In-memory fallback when ChromaDB isn't available."""
        self._memories: list[MemoryEntry] = []
        self._fallback = True

    @property
    def _using_fallback(self) -> bool:
        return hasattr(self, "_fallback") and self._fallback

    async def store(self, entry: MemoryEntry) -> str:
        """Store a memory entry."""
        if self._using_fallback:
            self._memories.append(entry)
            return entry.id

        # Reserved keys are written as native numbers/strings AFTER the
        # stringified user metadata so a stray user-supplied "timestamp" or
        # "importance" can never masquerade as the real one.
        metadata = {k: str(v) for k, v in entry.metadata.items()
                    if k not in ("memory_type", "timestamp", "importance")}
        metadata["memory_type"] = entry.memory_type.value
        metadata["timestamp"] = entry.timestamp
        metadata["importance"] = float(
            entry.importance if entry.importance is not None
            else _default_importance(entry.memory_type)
        )

        self._collection.upsert(
            ids=[entry.id],
            documents=[entry.content],
            metadatas=[metadata],
        )

        logger.debug(f"Stored memory {entry.id}: {entry.content[:80]}...")
        return entry.id

    async def recall(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        limit: int | None = None,
    ) -> list[MemoryEntry]:
        """Retrieve relevant memories via semantic search + re-ranking."""
        if limit is None:
            limit = self.config.retrieval_limit
        limit = max(1, limit)

        if self._using_fallback:
            # Basic keyword matching fallback — still re-ranked by recency,
            # importance and lexical overlap (relevance stays 0 here since
            # there are no embeddings without ChromaDB).
            results = []
            query_lower = query.lower()
            for mem in self._memories:
                if query_lower in mem.content.lower():
                    results.append(mem)
            logger.debug(f"Memory recall (fallback): query='{query}' found {len(results)} memories")
            return score_memories(results, query)[:limit]

        where_filter = None
        if memory_type:
            where_filter = {"memory_type": memory_type.value}

        # Over-fetch so re-ranking has a pool large enough to promote a
        # recent/important memory past a slightly-better-matching stale one.
        # count() also guards against n_results > index size, which older
        # chromadb versions reject outright.
        try:
            count = self._collection.count()
            if count == 0:
                return []
            results = self._collection.query(
                query_texts=[query],
                n_results=min(limit * 4, count),
                where=where_filter,
            )
        except Exception as e:
            logger.error(f"Memory recall error: {e}")
            return []

        entries = []
        filtered_count = 0
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 1.0
                relevance = 1.0 - distance  # Convert distance to similarity

                if relevance < self.config.relevance_threshold:
                    filtered_count += 1
                    continue

                try:
                    mem_type = MemoryType(meta.get("memory_type", "fact"))
                except ValueError:
                    mem_type = MemoryType.FACT
                try:
                    # Pre-v2 entries carry no importance → None → default at
                    # scoring time, so re-ranking them works unchanged.
                    importance: float | None = float(meta["importance"])
                except (KeyError, TypeError, ValueError):
                    importance = None

                entries.append(MemoryEntry(
                    content=doc,
                    memory_type=mem_type,
                    metadata={k: v for k, v in meta.items()
                              if k not in ("memory_type", "timestamp", "importance")},
                    id=results["ids"][0][i],
                    timestamp=float(meta.get("timestamp", 0)),
                    relevance=relevance,
                    importance=importance,
                ))

        ranked = score_memories(entries, query)[:limit]
        logger.debug(
            f"Memory recall: query='{query}' found {len(ranked)} of {len(entries)} "
            f"candidates (filtered {filtered_count} below threshold "
            f"{self.config.relevance_threshold})"
        )
        return ranked

    async def forget(self, memory_id: str) -> bool:
        """Remove a specific memory."""
        if self._using_fallback:
            self._memories = [m for m in self._memories if m.id != memory_id]
            return True

        try:
            self._collection.delete(ids=[memory_id])
            logger.debug(f"Forgot memory: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"Error forgetting memory {memory_id}: {e}")
            return False

    async def get_stats(self) -> dict[str, Any]:
        """Get memory store statistics."""
        if self._using_fallback:
            return {
                "backend": "in-memory (fallback)",
                "total_memories": len(self._memories),
            }

        return {
            "backend": "chromadb",
            "total_memories": self._collection.count() if self._collection else 0,
            "collection": self.config.chromadb.collection_name,
            "persist_directory": self.config.chromadb.persist_directory,
        }

    async def get_all_memories(
        self, memory_type: MemoryType | None = None, limit: int | None = 50
    ) -> list[MemoryEntry]:
        """Get all memories, optionally filtered by type. limit=None → all."""
        if self._using_fallback:
            mems = self._memories
            if memory_type:
                mems = [m for m in mems if m.memory_type == memory_type]
            return mems[:limit] if limit is not None else list(mems)

        where_filter = None
        if memory_type:
            where_filter = {"memory_type": memory_type.value}

        try:
            count = self._collection.count()
            if count == 0:
                return []

            get_kwargs: dict[str, Any] = {"where": where_filter}
            if limit is not None:
                get_kwargs["limit"] = max(1, min(limit, count))
            results = self._collection.get(**get_kwargs)
        except Exception as e:
            logger.error(f"Error getting all memories: {e}")
            return []

        entries = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"]):
                meta = results["metadatas"][i] if results["metadatas"] else {}
                try:
                    mem_type = MemoryType(meta.get("memory_type", "fact"))
                except ValueError:
                    mem_type = MemoryType.FACT
                try:
                    importance: float | None = float(meta["importance"])
                except (KeyError, TypeError, ValueError):
                    importance = None
                entries.append(MemoryEntry(
                    content=doc,
                    memory_type=mem_type,
                    metadata={k: v for k, v in meta.items()
                              if k not in ("memory_type", "timestamp", "importance")},
                    id=results["ids"][i],
                    timestamp=float(meta.get("timestamp", 0)),
                    importance=importance,
                ))

        return entries


def create_memory_store(config: MemoryConfig) -> "MemoryStore":
    """Factory: return the configured memory backend.

    Returns an ElasticsearchMemoryStore when config.backend == "elasticsearch",
    otherwise returns the default ChromaDB-backed MemoryStore.
    """
    if config.backend == "elasticsearch":
        from backend.core.memory_elasticsearch import ElasticsearchMemoryStore
        return ElasticsearchMemoryStore(config.elasticsearch)  # type: ignore[return-value]
    return MemoryStore(config)
