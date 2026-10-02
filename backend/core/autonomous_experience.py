"""Select distinct, evidenced autonomous experiences for identity consolidation."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from backend.core.world_state import parse_new_state


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def experience_metadata(
    text: str,
    tool_results: list[dict[str, Any]],
    work_changed: bool,
    had_events: bool,
) -> dict[str, Any]:
    meaningful_results = [
        result for result in tool_results
        if result.get("name") not in {"express", "recall_memory", "recall_autonomy", "query_world_graph"}
        and not (result.get("name") == "task" and result.get("arguments", {}).get("action") == "list")
    ]
    evidence = bool(meaningful_results or work_changed or had_events)
    payload = {
        "text": _normalize(text) if work_changed or had_events else "",
        "results": meaningful_results,
        "work_changed": work_changed,
        "events": had_events,
    }
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    kind = "idle"
    if meaningful_results or work_changed:
        kind = "blocked" if meaningful_results and not work_changed and all(
            not result.get("success", False) for result in meaningful_results
        ) else "work"
    elif had_events:
        kind = "event"
    return {
        "experience_kind": kind,
        "consolidation_eligible": evidence,
        "experience_signature": signature,
    }


def select_experiences(events: list[dict], since_ts: float) -> list[dict]:
    """Keep history for audit, but count only distinct non-idle evidence."""
    seen: set[str] = set()
    selected = []
    for event in sorted(events, key=lambda e: float(e.get("timestamp", 0))):
        meta = event.get("metadata") or {}
        content = event.get("content") or ""
        if "consolidation_eligible" in meta:
            eligible = bool(meta["consolidation_eligible"])
            signature = meta.get("experience_signature") or _normalize(content)
        else:
            # Legacy records have no tool results; state-only ticks cannot prove work.
            state = parse_new_state(content)
            has_narrative = bool(re.search(r"\[(?:Action|Outcome)\]", content, re.I))
            skills = set(meta.get("skills_used") or [])
            eligible = has_narrative and bool(skills - {"express", "recall", "recall_memory", "recall_autonomy", "task", "query_world_graph"})
            signature = _normalize(content if state is None else content.split("[New State]", 1)[0])
        if not eligible or signature in seen:
            continue
        seen.add(signature)
        if float(event.get("timestamp", 0)) > since_ts:
            selected.append(event)
    return selected
