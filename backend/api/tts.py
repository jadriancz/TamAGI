"""TTS API — synthesize speech on demand and serve generated audio clips.

POST /api/tts           {"text": "..."} -> {"url": "/api/tts/audio/<file>.wav"}
GET  /api/tts/audio/{name}              -> WAV file

Both paths run the same Chatterbox NIM synthesis as the `speak` skill
(backend/skills/tts_skill.py); the POST exists so the frontend can voice
any assistant message with a 🔊 button without an LLM round-trip.
"""

from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend.skills.tts_skill import TTSSkill, _chunk_text, _clean_for_speech, _TTS_DIR

router = APIRouter(prefix="/api/tts", tags=["tts"])

_skill = TTSSkill()

# Only generated clips (tts_<ts>_<sha>.wav) may be served — blocks traversal
# by construction since the name is validated against this pattern.
_SAFE_NAME = re.compile(r"^tts_\d+_[0-9a-f]{10}\.wav$")


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


class StreamTTSRequest(BaseModel):
    # The stream endpoint chunks to ≤300 chars before synthesis, so its input
    # cap can sit far above the single-shot 2000 (8000 ≈ 27 chunks ≈ 13 min).
    text: str = Field(..., min_length=1, max_length=8000)


@router.post("")
async def synthesize(request: TTSRequest):
    text = _clean_for_speech(request.text)
    if not text:
        raise HTTPException(status_code=400, detail="No queda texto sintetizable")
    result = await _skill.execute(text=text)
    if not result.success:
        raise HTTPException(status_code=502, detail=result.error)
    return {"url": result.data["url"], "bytes": result.data["bytes"]}


@router.post("/stream")
async def synthesize_stream(request: StreamTTSRequest):
    """Streaming variant: one NDJSON line per synthesized chunk.

    The hosted Chatterbox takes ~30s per chunk, so instead of waiting for the
    whole message the frontend plays chunk 0 while chunk 1+ still synthesize.
    Each line: {"i": n, "url": "..."} ... final line {"done": true, "total": N}.
    """
    text = _clean_for_speech(request.text)
    if not text:
        raise HTTPException(status_code=400, detail="No queda texto sintetizable")
    chunks = _chunk_text(text)

    async def gen():
        import asyncio
        import json as _json

        for i, chunk in enumerate(chunks):
            result = await _skill.execute(text=chunk)  # chunk ≤ cap → single synth
            if not result.success:
                yield _json.dumps({"error": result.error}) + "\n"
                return
            yield _json.dumps({"i": i, "url": result.data["url"]}) + "\n"
            await asyncio.sleep(0)  # flush the line to the client immediately
        yield _json.dumps({"done": True, "total": len(chunks)}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@router.get("/audio/{name}")
async def serve_audio(name: str):
    if not _SAFE_NAME.match(name):
        raise HTTPException(status_code=400, detail="Nombre de clip inválido")
    path = _TTS_DIR / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Clip no encontrado")
    return FileResponse(path, media_type="audio/wav", filename=name)
