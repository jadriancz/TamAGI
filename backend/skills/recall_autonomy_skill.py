"""
Recall Autonomy Skill — lets TamAGI browse its own autonomous activity log
(the World Thread's monologue: ticks, reflections, consolidations).

The World Thread runs whether or not anyone is chatting, but nothing in the
chat pipeline exposed that history to the model — so TamAGI honestly (and
wrongly) told users it does nothing between conversations. This skill gives
it read access to data/monologue.jsonl so it can answer "what have you been
up to?" from its actual lived record.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.skills.base import Skill, SkillResult

_LOG_PATH = Path("data/monologue.jsonl")


def _relative_time(ts: float) -> str:
    diff = datetime.now().timestamp() - ts
    if diff < 3600:
        return f"{int(diff // 60)}m ago"
    if diff < 86400:
        return f"{int(diff // 3600)}h ago"
    return f"{int(diff // 86400)}d ago"


class RecallAutonomySkill(Skill):
    """
    Read your own autonomous activity log — what you actually did between
    conversations (world ticks, reflections, identity consolidation).

    Use this when the user asks what you've been doing, or when you want to
    ground yourself in what happened while they were away.
    """

    name = "recall_autonomy"
    description = (
        "Read your autonomous activity log — the record of what you did between "
        "conversations: world ticks, quiet moments, reflections, and identity "
        "consolidation. Use this when the user asks what you've been up to, or "
        "when you want to recall your own recent autonomous life."
    )
    parameters = {
        "limit": {
            "type": "integer",
            "description": "How many recent entries to retrieve (1-20). Default: 5.",
            "default": 5,
        },
        "source": {
            "type": "string",
            "description": (
                "Optional filter by entry source. 'autonomous' = world thread "
                "activity, 'user' = user-triggered. Leave empty for all."
            ),
            "default": "",
        },
    }

    async def execute(self, **kwargs: Any) -> SkillResult:
        limit = min(max(int(kwargs.get("limit", 5)), 1), 20)
        source = str(kwargs.get("source", "")).strip().lower()

        if not _LOG_PATH.exists():
            return SkillResult(
                success=True,
                output="Your autonomy log is empty — no world ticks recorded yet.",
                data={"entries": [], "count": 0},
            )

        try:
            lines = [
                json.loads(line)
                for line in _LOG_PATH.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError) as exc:
            return SkillResult(success=False, error=f"Could not read autonomy log: {exc}")

        if source:
            lines = [e for e in lines if e.get("source") == source]

        entries = lines[-limit:][::-1]  # newest-first
        if not entries:
            return SkillResult(
                success=True,
                output="No matching entries in your autonomy log.",
                data={"entries": [], "count": 0},
            )

        out = [f"Your autonomous activity — {len(entries)} recent entr"
               f"{'y' if len(entries) == 1 else 'ies'}:", ""]
        for e in entries:
            ts = float(e.get("timestamp", 0))
            when = _relative_time(ts)
            title = (e.get("title") or "(untitled)").strip()
            out.append(f"[{when}] {title}")
            # Include the action/summary, trimmed
            content = (e.get("content") or "").strip()
            if content:
                action_line = ""
                for line in content.splitlines():
                    s = line.strip()
                    if s and not s.startswith(("[", "#", "**")):
                        action_line = s[:200]
                        break
                if action_line:
                    out.append(f"  {action_line}")
            out.append("")

        return SkillResult(
            success=True,
            output="\n".join(out).strip(),
            data={"entries": entries, "count": len(entries), "source": source or "all"},
        )
