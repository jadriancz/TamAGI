"""Speak skill — text-to-speech so TamAGI can literally talk.

Uses NVIDIA NIM's hosted Chatterbox Multilingual TTS (Resemble AI) over gRPC.
The synthesized WAV is written to data/tts/ and served by the /api/tts router,
so the frontend (or the model's own reply markdown) can play it.

Dependency: nvidia-riva-client (see pyproject.toml). The API key is reused
from config.llm (same NIM account) unless tts.api_key is set explicitly.
"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import Any

from backend.config import get_config
from backend.skills.base import Skill, SkillResult

# Default voice: validated live against the hosted function (2026-11-19).
# Chatterbox voice naming: "Chatterbox-Multilingual.<locale>.<Gender>".
_DEFAULT_VOICE = "Chatterbox-Multilingual.es-ES.Male"
_DEFAULT_LANGUAGE = "es-ES"
_SAMPLE_RATE = 24000

# Strip markdown so headings/code-fences don't get read aloud as symbols.
_MD_NOISE = re.compile(r"```.*?```|`([^`]*)`|\*\*([^*]*)\*\*|[#*_>|]|\[|\]\([^)]*\)", re.S)

# Hard limits on the hosted Chatterbox function (verified 2026-11-19):
#   input  — "Input text length exceeds maximum allowed length (500)"
#   output — "output reached the maximum allowed length of 500 speech
#             tokens (~20s)" — a ~460-char Spanish chunk overruns 20s.
# 300 chars ≈ 13-15s of speech: safely under both caps.
_MAX_TTS_CHARS = 300

_TTS_DIR = Path("data/tts")


def _clean_for_speech(text: str) -> str:
    """Strip markdown noise; collapse whitespace. Keeps code spans' content."""
    text = _MD_NOISE.sub(lambda m: m.group(1) or m.group(2) or " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _chunk_text(text: str, limit: int = _MAX_TTS_CHARS) -> list[str]:
    """Split text into speakable chunks under the server's 500-char cap.

    Prefers sentence boundaries; hard-splits by word when a single sentence
    exceeds the limit on its own.
    """
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    cur = ""
    for sentence in re.split(r"(?<=[.!?…])\s+", text):
        while len(sentence) > limit:
            cut = sentence.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            chunks.append(sentence[:cut])
            sentence = sentence[cut:].lstrip()
        if cur and len(cur) + len(sentence) + 1 <= limit:
            cur = f"{cur} {sentence}"
        elif cur:
            chunks.append(cur)
            cur = sentence
        else:
            cur = sentence
    if cur:
        chunks.append(cur)
    return chunks


class TTSSkill(Skill):
    name = "speak"
    description = (
        "Convierte texto en audio hablado (tu propia voz) y devuelve la URL del WAV. "
        "Úsalo para leer en voz alta tu respuesta — saludos, avisos, o cuando quieras "
        "hablar de verdad. Textos largos se sintetizan por tramos y se unen."
    )
    parameters = {
        "text": {
            "type": "string",
            "description": "Lo que vas a decir en voz alta (texto plano, sin markdown)",
            "required": True,
        },
        "voice": {
            "type": "string",
            "description": f"Voz Chatterbox (por defecto: {_DEFAULT_VOICE})",
            "required": False,
        },
    }

    def __init__(self) -> None:
        self._tts: Any | None = None
        self._auth_voice: str | None = None

    def _service(self) -> tuple[Any, str]:
        """Lazy-init the gRPC client (import + connect on first use)."""
        cfg = get_config().tts
        voice = cfg.voice or _DEFAULT_VOICE
        if self._tts is None or self._auth_voice != voice:
            import riva.client  # heavy: import only when needed

            api_key = cfg.api_key or get_config().llm.api_key
            auth = riva.client.Auth(
                uri="grpc.nvcf.nvidia.com:443",
                use_ssl=True,
                metadata_args=[
                    ["function-id", cfg.function_id],
                    ["authorization", f"Bearer {api_key}"],
                ],
            )
            self._tts = riva.client.SpeechSynthesisService(auth)
            self._auth_voice = voice
        return self._tts, voice

    async def execute(self, **kwargs: Any) -> SkillResult:
        raw = kwargs.get("text", "")
        text = _clean_for_speech(str(raw))
        if not text:
            return SkillResult(success=False, error="No hay texto para hablar")

        cfg = get_config().tts
        try:
            svc, default_voice = self._service()
            voice = str(kwargs.get("voice") or default_voice)
            loop = __import__("asyncio").get_running_loop()

            def synth_one(chunk: str) -> bytes:
                resp = svc.synthesize(
                    text=chunk,
                    voice_name=voice,
                    language_code=cfg.language,
                    sample_rate_hz=_SAMPLE_RATE,
                )
                return resp.audio

            chunks = _chunk_text(text)
            audio_parts: list[bytes] = []
            for chunk in chunks:
                audio_parts.append(await loop.run_in_executor(None, synth_one, chunk))
            # riva returns RAW PCM (no RIFF header) — join parts and envelope
            # once as a proper 24kHz mono 16-bit WAV.
            audio = _pcm_to_wav(b"".join(audio_parts), _SAMPLE_RATE)
        except Exception as exc:
            return SkillResult(success=False, error=f"TTS falló: {type(exc).__name__}: {exc}")

        _TTS_DIR.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(text.encode()).hexdigest()[:10]
        fname = f"tts_{int(time.time())}_{digest}.wav"
        path = _TTS_DIR / fname
        path.write_bytes(audio)

        # Housekeeping: keep only the newest 30 clips.
        clips = sorted(_TTS_DIR.glob("tts_*.wav"))
        for old in clips[:-30]:
            old.unlink(missing_ok=True)

        url = f"/api/tts/audio/{fname}"
        return SkillResult(
            success=True,
            output=f"🔊 Dicho en voz alta. Audio: [{fname}]({url})",
            data={"url": url, "file": fname, "bytes": len(audio), "chars": len(text)},
        )


def _pcm_to_wav(pcm: bytes, sample_rate: int, channels: int = 1, sampwidth: int = 2) -> bytes:
    """Wrap raw little-endian PCM in a proper RIFF/WAVE header (stdlib wave)."""
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()
