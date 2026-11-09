"""Briefing — the autonomous work Tama did while you were away, made visible.

The world thread ticks every ~15 minutes. Most ticks are idle or rest; some do
real work (a quest advanced, a task finished, something written). The Briefing
store captures one compact, human-readable record per NON-idle tick so that
when the user returns and starts a new conversation, Tama can honestly say
"while you were out I did X, got blocked on Y, and I'm now working on Z" — with
every claim traceable to a real tick, not invented.

Pure functions are kept free of I/O so they're trivial to test; BriefingStore
is the only thing that touches disk, and like the self-model journal it is
best-effort (a write failure never disturbs the world loop).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("tamagi.briefing")

DEFAULT_BRIEFING_PATH = "data/briefings.jsonl"
_MAX_BRIEFINGS = 300  # rolling cap; oldest dropped on append


def _first_sentence(text: str, max_len: int = 220) -> str:
    """First sentence/line of the tick narration, trimmed and single-lined."""
    cleaned = (text or "").replace("\r", " ")
    # Drop any leftover structured blocks the model sometimes echoes.
    for marker in ("[New State]", "[Verified tool results", "[World pressure]"):
        idx = cleaned.find(marker)
        if idx != -1:
            cleaned = cleaned[:idx]
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return ""
    for stop in (". ", "! ", "? "):
        cut = cleaned.find(stop)
        if 0 < cut < max_len:
            return cleaned[: cut + 1]
    return cleaned[:max_len] + ("…" if len(cleaned) > max_len else "")


def _evidence_lines(tool_results: list[dict[str, Any]], mutations: list[dict[str, Any]]) -> list[str]:
    """Short, verifiable bullets: what each tool actually did + graph changes."""
    lines: list[str] = []
    for res in tool_results:
        name = res.get("name", "?")
        if name in {"express", "recall_memory", "recall_autonomy", "query_world_graph"}:
            continue  # noise: reading/expressing isn't "work done"
        ok = res.get("success", False)
        lines.append(f"{'did' if ok else 'failed'}: {name}")
    for mut in mutations:
        nid = mut.get("id", "?")
        ntype = mut.get("node_type", "node")
        lines.append(f"changed: {ntype} {nid}")
    # De-dup, preserve order, cap.
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            out.append(line)
    return out[:8]


def build_briefing(
    response_text: str,
    location: str,
    mood: str,
    focus: str,
    skills_used: list[str],
    tool_results: list[dict[str, Any]],
    mutations: list[dict[str, Any]],
    tasks_changed: bool,
    kind: str,
    now: float | None = None,
) -> dict[str, Any] | None:
    """One briefing record for a meaningful tick, or None if this tick said
    nothing worth reporting (idle with no evidence)."""
    if kind == "idle":
        return None
    evidence = _evidence_lines(tool_results, mutations)
    summary = _first_sentence(response_text)
    if not summary and not evidence:
        return None  # nothing verifiable — don't invent a briefing
    return {
        "ts": now if now is not None else time.time(),
        "kind": kind,
        "summary": summary,
        "location": location,
        "mood": mood,
        "focus": focus,
        "skills": list(skills_used),
        "evidence": evidence,
        "tasks_changed": bool(tasks_changed),
    }


def render_for_prompt(briefings: list[dict[str, Any]]) -> str:
    """Render recent briefings as a bounded system-prompt block for a new
    conversation. Empty string when there's nothing to report."""
    if not briefings:
        return ""
    lines = []
    for b in briefings:
        when = time.strftime("%b %d %H:%M", time.localtime(b["ts"]))
        head = f"- [{when}] {b['kind']}: {b['summary'] or '(no narration)'}"
        if b.get("evidence"):
            head += " — " + "; ".join(b["evidence"])
        lines.append(head)
    where = briefings[-1].get("focus") or briefings[-1].get("location")
    tail = f"\n(You are now focused on: {where})" if where else ""
    return (
        "\n\n[Autonomous briefings — what you actually did since the user was "
        "last here. Reference naturally ('while you were away I…'), never recite "
        "the list, and don't invent work that isn't here:]\n"
        + "\n".join(lines)
        + tail
    )


class BriefingStore:
    """Append-only JSONL of briefings, newest-last. Best-effort disk store."""

    def __init__(self, path: str | Path = DEFAULT_BRIEFING_PATH) -> None:
        self._path = Path(path)

    def append(self, record: dict[str, Any]) -> bool:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            existing = self._read_lines()
            existing.append(record)
            if len(existing) > _MAX_BRIEFINGS:
                existing = existing[-_MAX_BRIEFINGS:]
            tmp = self._path.with_suffix(".jsonl.tmp")
            tmp.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in existing) + "\n",
                encoding="utf-8",
            )
            tmp.replace(self._path)
            return True
        except Exception:
            logger.debug("Briefing append failed", exc_info=True)
            return False

    def _read_lines(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a torn final line
        return out

    def since(self, cutoff_ts: float, limit: int = 8) -> list[dict[str, Any]]:
        """Briefings newer than cutoff_ts, oldest→newest, capped to the most
        recent `limit`."""
        fresh = [b for b in self._read_lines() if b.get("ts", 0) > cutoff_ts]
        return fresh[-limit:]
