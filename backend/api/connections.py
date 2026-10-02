"""
WebSocket connection registry — lets non-API subsystems (World Thread)
push server-initiated events to every connected frontend tab.

The chat endpoint registers each accepted socket here and unregisters it
on disconnect. broadcast() never raises: a dead socket is dropped silently.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("tamagi.api.connections")

_clients: dict[Any, str | None] = {}


def register(ws: Any) -> None:
    _clients[ws] = None


def bind_conversation(ws: Any, conversation_id: str | None) -> None:
    if ws in _clients:
        _clients[ws] = conversation_id


def unregister(ws: Any) -> None:
    _clients.pop(ws, None)


def active_count() -> int:
    return len(_clients)


def connected_conversations() -> set[str]:
    return {conversation_id for conversation_id in _clients.values() if conversation_id}


async def broadcast(event: dict[str, Any]) -> int:
    """Send an event to every connected socket. Returns sockets reached."""
    sent = 0
    for ws, conversation_id in list(_clients.items()):
        if event.get("conversation_id") and conversation_id != event["conversation_id"]:
            continue
        try:
            await ws.send_json(event)
            sent += 1
        except Exception:
            _clients.pop(ws, None)
    if sent:
        logger.debug("Broadcast %s to %d socket(s)", event.get("type"), sent)
    return sent
