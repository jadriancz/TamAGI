"""Memory v2: temporal + salience-aware re-ranking (pure fns + end-to-end recall)."""

import shutil
import tempfile
import time
import unittest

from backend.config import ChromaDBConfig, MemoryConfig
from backend.core.memory import (
    DEFAULT_IMPORTANCE,
    MemoryEntry,
    MemoryStore,
    MemoryType,
    _lexical_overlap,
    _recency_factor,
    _tokenize,
    score_memories,
)

NOW = 1_750_000_000.0  # fixed clock so ordering assertions are deterministic


def make_entry(content, mtype=MemoryType.FACT, age_days=0.0, relevance=0.5,
               importance=None, base=NOW):
    return MemoryEntry(
        content=content,
        memory_type=mtype,
        timestamp=base - age_days * 86400.0,
        relevance=relevance,
        importance=importance,
    )


class RecencyTests(unittest.TestCase):
    def test_now_scores_one_unknown_timestamp_scores_zero(self):
        self.assertAlmostEqual(_recency_factor(NOW, NOW), 1.0)
        self.assertEqual(_recency_factor(0, NOW), 0.0)

    def test_half_life_halves_then_quarters(self):
        self.assertAlmostEqual(_recency_factor(NOW - 30 * 86400, NOW), 0.5, places=6)
        self.assertAlmostEqual(_recency_factor(NOW - 60 * 86400, NOW), 0.25, places=6)

    def test_future_timestamp_never_exceeds_one(self):
        self.assertLessEqual(_recency_factor(NOW + 10 * 86400, NOW), 1.0)


class TokenizeTests(unittest.TestCase):
    def test_stopwords_short_words_dropped_accents_kept(self):
        self.assertEqual(_tokenize("La ciudad de Ñuñoa"), {"ciudad", "ñuñoa"})

    def test_es_and_en_stopwords_both_filtered(self):
        self.assertEqual(_tokenize("what is the qué"), set())

    def test_lexical_overlap_fraction(self):
        q = {"santiago", "metro"}
        self.assertAlmostEqual(_lexical_overlap(q, {"santiago", "metro"}), 1.0)
        self.assertAlmostEqual(_lexical_overlap(q, {"santiago"}), 0.5)
        self.assertAlmostEqual(_lexical_overlap(q, {"bus"}), 0.0)
        self.assertAlmostEqual(_lexical_overlap(set(), {"x"}), 0.0)


class ScoreMemoriesTests(unittest.TestCase):
    def test_same_relevance_newer_wins(self):
        fresh = make_entry("charla de trabajo", age_days=2)
        stale = make_entry("charla de trabajo", age_days=45)
        ranked = score_memories([stale, fresh], "charla trabajo", now=NOW)
        self.assertIs(ranked[0], fresh)

    def test_same_relevance_and_age_more_important_type_wins(self):
        talk = make_entry("un día cualquiera", mtype=MemoryType.CONVERSATION, age_days=10)
        fact = make_entry("un día cualquiera", mtype=MemoryType.KNOWLEDGE, age_days=10)
        ranked = score_memories([talk, fact], "día cualquiera", now=NOW)
        self.assertIs(ranked[0], fact)

    def test_explicit_importance_overrides_type_default(self):
        loud = make_entry("nota normal", mtype=MemoryType.CONVERSATION,
                          age_days=10, importance=1.0)
        calm = make_entry("nota normal", mtype=MemoryType.KNOWLEDGE, age_days=10)
        ranked = score_memories([calm, loud], "nota normal", now=NOW)
        self.assertIs(ranked[0], loud)

    def test_lexical_shared_entity_rescues_lower_relevance(self):
        # Same recency/importance; the shared name must outweigh a small
        # semantic-relevance gap (relevance weight 0.55 × Δ0.07 < lexical 0.10).
        query = "¿Qué sabes de Valparaíso?"
        hit = make_entry("Valparaíso tiene cerros y ascensores", relevance=0.60)
        miss = make_entry("Las ciudades puerto tienen cerros", relevance=0.67)
        ranked = score_memories([miss, hit], query, now=NOW)
        self.assertIs(ranked[0], hit)

    def test_pre_v2_entry_without_importance_still_ranks(self):
        legacy = make_entry("recuerdo antiguo", age_days=5)
        legacy.importance = None
        ranked = score_memories([legacy], "recuerdo antiguo", now=NOW)
        self.assertEqual(len(ranked), 1)  # no crash, default resolved

    def test_relevance_is_clamped_to_unit_range(self):
        weird = make_entry("raro", relevance=5.0)
        normal = make_entry("raro", relevance=1.0)
        ranked = score_memories([weird, normal], "raro", now=NOW)
        self.assertAlmostEqual(ranked[0].relevance, 5.0)  # attribute untouched


class DefaultImportanceTests(unittest.TestCase):
    def test_knowledge_outranks_conversation(self):
        self.assertGreater(DEFAULT_IMPORTANCE[MemoryType.KNOWLEDGE],
                           DEFAULT_IMPORTANCE[MemoryType.CONVERSATION])

    def test_to_dict_resolves_importance(self):
        e = make_entry("x", mtype=MemoryType.PREFERENCE)
        self.assertEqual(e.to_dict()["importance"],
                         DEFAULT_IMPORTANCE[MemoryType.PREFERENCE])


class FallbackRecallRankingTests(unittest.IsolatedAsyncioTestCase):
    """In-memory backend (no Chroma) must rank by recency+importance too."""

    async def asyncSetUp(self):
        self.store = MemoryStore(MemoryConfig())
        self.store._init_fallback()
        # The fallback recall scores against the REAL clock (time.time()),
        # so entries must be aged relative to it, not the fixed NOW.
        self.real = time.time()

    async def test_recent_beats_stale_beats_old(self):
        await self.store.store(make_entry("Hablamos de Santiago ayer",
                                          age_days=0, base=self.real))
        await self.store.store(make_entry("Santiago quedó anotado hace mucho",
                                          mtype=MemoryType.KNOWLEDGE,
                                          age_days=60, base=self.real))
        await self.store.store(make_entry("Santiago, charla vieja",
                                          age_days=90, base=self.real))
        ranked = await self.store.recall("santiago")
        self.assertEqual([e.content for e in ranked], [
            "Hablamos de Santiago ayer",
            "Santiago quedó anotado hace mucho",
            "Santiago, charla vieja",
        ])

    async def test_limit_applied_after_ranking(self):
        for days in (90, 45, 0):
            await self.store.store(make_entry("santiago otra vez",
                                              age_days=days, base=self.real))
        ranked = await self.store.recall("santiago", limit=1)
        self.assertEqual(len(ranked), 1)
        self.assertAlmostEqual(ranked[0].timestamp, self.real, places=1)

    async def test_zero_limit_clamps_to_one(self):
        await self.store.store(make_entry("santiago"))
        self.assertEqual(len(await self.store.recall("santiago", limit=0)), 1)


class ChromaRoundTripTests(unittest.IsolatedAsyncioTestCase):
    """Real ChromaDB: numeric importance metadata, legacy entries, ranking."""

    async def asyncSetUp(self):
        # Not TemporaryDirectory: on Windows chromadb keeps its sqlite/HNSW
        # files open until process exit and the implicit cleanup would raise
        # WinError 32. Best-effort rmtree instead; a leaked temp dir is fine.
        self._tmp = tempfile.mkdtemp(prefix="tamagi_mem_test_")
        config = MemoryConfig(chromadb=ChromaDBConfig(
            persist_directory=self._tmp, collection_name="test_memory_v2"))
        self.store = MemoryStore(config)
        await self.store.initialize()

    async def asyncTearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    async def test_store_writes_importance_and_shields_reserved_keys(self):
        now = time.time()
        await self.store.store(MemoryEntry(
            content="El usuario prefiere respuestas cortas",
            memory_type=MemoryType.PREFERENCE,
            metadata={"source": "chat", "timestamp": "FAKE"},  # must be ignored
            timestamp=now,
        ))
        raw = self.store._collection.get(ids=None)["metadatas"][0]
        self.assertEqual(raw["importance"], DEFAULT_IMPORTANCE[MemoryType.PREFERENCE])
        self.assertAlmostEqual(float(raw["timestamp"]), now)
        self.assertEqual(raw["source"], "chat")

        got = await self.store.recall("El usuario prefiere respuestas cortas")
        self.assertTrue(got)
        self.assertAlmostEqual(got[0].importance,
                               DEFAULT_IMPORTANCE[MemoryType.PREFERENCE])
        self.assertNotIn("importance", got[0].metadata)
        self.assertNotIn("timestamp", got[0].metadata)  # fake one dropped

    async def test_legacy_entry_without_importance_or_timestamp(self):
        self.store._collection.upsert(
            ids=["legacy1"],
            documents=["dato guardado antes de la v2"],
            metadatas=[{"memory_type": "fact"}],
        )
        got = await self.store.recall("dato guardado antes de la v2")
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0].importance, None)  # default applied at score time
        self.assertEqual(got[0].timestamp, 0.0)

    async def test_fresh_and_important_beats_stale_equal_text(self):
        now = time.time()
        text = "Anotación sobre el proyecto TamAGI y sus pruebas"
        await self.store.store(MemoryEntry(
            content=text, memory_type=MemoryType.CONVERSATION,
            timestamp=now - 400 * 86400))
        await self.store.store(MemoryEntry(
            content=text, memory_type=MemoryType.KNOWLEDGE,
            timestamp=now, id=None))
        got = await self.store.recall(text, limit=2)
        self.assertEqual(len(got), 2)
        self.assertEqual(got[0].memory_type, MemoryType.KNOWLEDGE)

    async def test_get_all_memories_none_limit_returns_everything(self):
        for i in range(3):
            await self.store.store(MemoryEntry(
                content=f"entrada numerada {i}", memory_type=MemoryType.FACT))
        self.assertEqual(len(await self.store.get_all_memories(limit=None)), 3)


if __name__ == "__main__":
    unittest.main()
