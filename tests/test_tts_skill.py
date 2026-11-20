"""Unit tests for the TTS skill's pure logic (no network)."""

from __future__ import annotations

import re

from backend.skills.tts_skill import TTSSkill, _chunk_text, _clean_for_speech, _TTS_DIR


def test_chunk_short_text_untouched():
    assert _chunk_text("Hola mundo") == ["Hola mundo"]


def test_chunk_respects_server_limit():
    text = "Esta es una frase de prueba. " * 40  # ~1120 chars
    for c in _chunk_text(text):
        assert len(c) <= 480
    assert "".join(_chunk_text(text)).replace(" ", "") == text.replace(" ", "")


def test_chunk_hard_splits_long_sentence():
    text = "palabra " * 200  # una sola oración, ~1600 chars
    chunks = _chunk_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= 480 for c in chunks)


def test_clean_strips_markdown_but_keeps_code_content():
    raw = "## Título\n**Hola** `mundo` ```\nbloque\n``` fin"
    out = _clean_for_speech(raw)
    assert "#" not in out and "**" not in out
    assert "Hola" in out and "mundo" in out


def test_clean_collapses_whitespace():
    assert _clean_for_speech("a\n\n  b\tc") == "a b c"


def test_audio_name_pattern_matches_generated_clips():
    # Same regex the API route uses to gate serving; keep them in sync.
    pattern = re.compile(r"^tts_\d+_[0-9a-f]{10}\.wav$")
    assert pattern.match("tts_1732000000_abcdef1234.wav")
    assert not pattern.match("../config.yaml")
    assert not pattern.match("tts_x.wav")


def test_tts_dir_is_relative_to_data():
    assert str(_TTS_DIR).replace("\\", "/").startswith("data/tts")


def test_skill_metadata():
    s = TTSSkill()
    assert s.name == "speak"
    assert "text" in s.parameters
    tool = s.to_openai_tool()
    assert tool["function"]["name"] == "speak"
