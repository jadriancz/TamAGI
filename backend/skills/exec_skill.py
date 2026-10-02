"""
Exec Skill — Execute shell commands with a tiered trust model.

Trust tiers (resolved at base command name level):
  safe    — run immediately, no user notification
  notify  — run + emit interim_text so the user can see it in reasoning
  approve — surface an inline approval banner; await user Allow/Deny (30 s timeout = deny)
  block   — always rejected, no override

Unknown commands default to the `approve` tier.

Destructive argument patterns (rm, --force, prune, stop, kill, delete, …) escalate
any notify-tier command to approve-tier regardless of classification.

Runtime approvals ("Allow always") are persisted to data/trusted_commands.json and
promote the base command name to notify-tier for the lifetime of the process.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import shlex
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from backend.config import get_config
from backend.skills.base import Skill, SkillResult

logger = logging.getLogger("tamagi.skills.exec")

# ── Destructive argument patterns ─────────────────────────────
# If any of these appear anywhere in the full command string, the
# command escalates to approve-tier regardless of its base trust level.
_DESTRUCTIVE_PATTERNS = frozenset([
    "rm", "rmi", "--force", "-rf", "-fr", "prune", "stop", "kill",
    "delete", "drop", "truncate", "format", "wipe", "purge",
])

# Persistent runtime trust file
_TRUSTED_FILE = Path("data/trusted_commands.json")

# In-process cache — loaded once, updated on "Allow always"
_runtime_trusted: set[str] = set()


def _load_runtime_trusted() -> None:
    """Load persisted runtime-approved commands into the in-memory cache."""
    global _runtime_trusted
    if _TRUSTED_FILE.exists():
        try:
            data = json.loads(_TRUSTED_FILE.read_text())
            _runtime_trusted = set(data.get("trusted", []))
        except Exception:
            _runtime_trusted = set()


def _save_runtime_trusted() -> None:
    _TRUSTED_FILE.parent.mkdir(parents=True, exist_ok=True)
    _TRUSTED_FILE.write_text(json.dumps({"trusted": sorted(_runtime_trusted)}, indent=2))


def execution_environment() -> str:
    return (
        f"Execution environment: {platform.system()}. exec launches a native program, "
        "NOT a shell. Use program + args (JSON string array) and working_dir. "
        "Do not put pipes, redirects, ;, &&, shell built-ins or variable expansion in command. "
        "Use one program per call; use read to list directories or read files. "
        f"The Python interpreter available to exec is {sys.executable!r}. "
        + ("Use native absolute Windows paths (C:/Users/...); /c/Users/... is a Git Bash "
           "path, not a native Windows path. " if os.name == "nt" else "")
        + "Never repeat a failed command unchanged; report errors and partial progress. "
        "Shell scripts require an explicitly approved interpreter."
    )


def _command_parts(command: str) -> list[str]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>()")
    lexer.whitespace_split = True
    lexer.commenters = ""
    if any(token and all(c in ";&|<>()" for c in token) for token in lexer):
        raise ValueError("Shell operators are not supported. Use program + args, one program per call.")
    if os.name == "nt":
        return [
            part[1:-1] if len(part) >= 2 and part[0] == part[-1] and part[0] in "\"'" else part
            for part in shlex.split(command, posix=False)
        ]
    return shlex.split(command)


def _get_trust_tier(base_cmd: str, exec_trust) -> str:
    """Return 'safe' | 'notify' | 'approve' | 'block' for a base command name."""
    if base_cmd in exec_trust.block:
        return "block"
    if base_cmd in _runtime_trusted:
        return "notify"  # runtime-approved → run + notify, no banner
    if base_cmd in exec_trust.safe:
        return "safe"
    if base_cmd in exec_trust.notify:
        return "notify"
    if base_cmd in exec_trust.approve:
        return "approve"
    return "approve"  # unknown → require approval


# Load persisted approvals at import time
_load_runtime_trusted()


class ExecSkill(Skill):
    name = "exec"
    description = (
        "Execute ONE native program without a shell, preferably using program + args. "
        "Pipes, redirection, semicolons, && and shell built-ins are NOT interpreted. "
        "Read-only tools run immediately; developer tools run with a notification; "
        "risky or unknown programs require approval. Destructive flags and shell "
        "interpreters always require approval. Defaults to the workspace directory."
    )
    parameters = {
        "program": {
            "type": "string",
            "description": "Native executable name or path, e.g. git. Use with args; do not also supply command.",
        },
        "args": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Literal arguments, e.g. [\"status\", \"--short\"]. Paths remain intact. No shell parsing.",
        },
        "command": {
            "type": "string",
            "description": "Legacy alternative: ONE simple program invocation. No shell operators. Prefer program + args.",
        },
        "working_dir": {
            "type": "string",
            "description": "Working directory (defaults to workspace path)",
            "default": "",
        },
    }

    def to_openai_tool(self) -> dict[str, Any]:
        tool = super().to_openai_tool()
        tool["function"]["parameters"]["properties"]["args"]["items"] = {"type": "string"}
        tool["function"]["description"] += " " + execution_environment()
        return tool

    async def execute(self, **kwargs: Any) -> SkillResult:
        config = get_config()
        command = kwargs.get("command", "")
        working_dir = kwargs.get("working_dir", "") or config.workspace.path
        event_callback = kwargs.get("_event_callback")
        pending_approvals: dict | None = kwargs.get("_pending_approvals")
        is_autonomous: bool = bool(kwargs.get("_is_autonomous", False))

        program = kwargs.get("program")
        args = kwargs.get("args", [])
        try:
            if not isinstance(command, str):
                raise ValueError("command must be a string")
            command = command.strip()
            if program is not None:
                if command:
                    raise ValueError("Supply program + args OR command, not both")
                if not isinstance(program, str) or not program.strip():
                    raise ValueError("program must be a non-empty string")
                if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
                    raise ValueError("args must be an array of strings")
                parts = [program.strip(), *args]
                command = shlex.join(parts)
            else:
                if args:
                    raise ValueError("args requires program")
                parts = _command_parts(command)
            if not parts:
                raise ValueError("No program provided")
            if any("\x00" in part for part in parts):
                raise ValueError("NUL characters are not allowed")
        except ValueError as exc:
            return SkillResult(success=False, error=f"Invalid invocation: {exc}", output=str(exc))

        base_cmd = parts[0].replace("\\", "/").rsplit("/", 1)[-1].lower()
        if base_cmd.endswith(".exe"):
            base_cmd = base_cmd[:-4]
        exec_trust = config.guardrails.exec_trust
        tier = _get_trust_tier(base_cmd, exec_trust)

        # Destructive pattern → escalate to approve
        has_destructive = bool(set(parts) & _DESTRUCTIVE_PATTERNS)
        if has_destructive and tier in ("safe", "notify"):
            tier = "approve"
        if tier != "block" and (base_cmd in {
            "bash", "sh", "zsh", "fish", "cmd", "powershell", "pwsh", "wsl",
        } or base_cmd.endswith((".bat", ".cmd"))):
            tier = "approve"

        # ── Block tier ──────────────────────────────────────────
        if tier == "block":
            return SkillResult(
                success=False,
                error=f"Command '{base_cmd}' is blocked and cannot be executed.",
                output=f"I can't run `{base_cmd}` — it's permanently blocked for safety.",
            )

        # ── Approve tier ────────────────────────────────────────
        if tier == "approve":
            if is_autonomous:
                # No user present — auto-deny and tag as capability gap so
                # the ReflectionEngine can teach AURA to avoid this pattern.
                from backend.core.plan_executor import _CAPABILITY_GAP_TAG
                return SkillResult(
                    success=False,
                    error=_CAPABILITY_GAP_TAG,
                    output=(
                        f"`{base_cmd}` requires user approval and cannot run autonomously. "
                        "This has been flagged as a capability gap."
                    ),
                )
            approved, allow_always = await self._request_approval(
                command, base_cmd, event_callback, pending_approvals
            )
            if not approved:
                return SkillResult(
                    success=False,
                    error="User denied execution",
                    output=f"I won't run `{command}` — you chose to deny this command.",
                )
            if allow_always:
                _runtime_trusted.add(base_cmd)
                _save_runtime_trusted()
                logger.info("Permanently approved command: %s", base_cmd)

        # ── Notify tier ─────────────────────────────────────────
        if tier == "notify" and event_callback:
            await event_callback({
                "type": "interim_text",
                "content": f"Running `{command}`",
            })

        logger.info("Executing: %s", command)
        return await self._run(command, parts, working_dir, config.guardrails.exec_timeout)

    async def _request_approval(
        self,
        command: str,
        base_cmd: str,
        event_callback,
        pending_approvals: dict | None,
    ) -> tuple[bool, bool]:
        """
        Emit tool_approval_required and await the user's response.

        Returns (approved, allow_always). Falls back to deny if event_callback
        or pending_approvals are unavailable (e.g., REST API path).
        """
        if not event_callback or pending_approvals is None:
            # No WS channel — default deny for safety
            logger.warning("Approval required for '%s' but no WS channel; denying.", command)
            return False, False

        approval_id = str(uuid4())
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        pending_approvals[approval_id] = future

        await event_callback({
            "type": "tool_approval_required",
            "approval_id": approval_id,
            "command": command,
            "base_command": base_cmd,
            "reason": (
                "Destructive argument detected — approval required."
                if any(p in command for p in _DESTRUCTIVE_PATTERNS)
                else f"'{base_cmd}' requires explicit approval before running."
            ),
            "timeout_seconds": 30,
        })

        try:
            response = await asyncio.wait_for(future, timeout=30)
            approved = response.get("approved", False)
            allow_always = response.get("allow_always", False)
            return approved, allow_always
        except asyncio.TimeoutError:
            logger.info("Approval timed out for: %s", command)
            return False, False
        finally:
            pending_approvals.pop(approval_id, None)

    async def _run(
        self, command: str, parts: list[str], working_dir: str, timeout: int
    ) -> SkillResult:
        try:
            process = await asyncio.create_subprocess_exec(
                *parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                # On Windows (Proactor loop) cancelling the pipe read can hang
                # past the timeout — cap the reaper so the skill always returns.
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    logger.error(
                        "Exec skill: subprocess %s survived kill after %ss timeout; "
                        "abandoning it.", parts[0], timeout,
                    )
                return SkillResult(
                    success=False,
                    error=f"Command timed out after {timeout}s",
                )

            stdout_text = stdout.decode("utf-8", errors="replace").strip()
            stderr_text = stderr.decode("utf-8", errors="replace").strip()
            success = process.returncode == 0

            output_parts = []
            if stdout_text:
                output_parts.append(stdout_text)
            if stderr_text:
                output_parts.append(f"[stderr] {stderr_text}")
            if not output_parts:
                output_parts.append("(no output)")

            return SkillResult(
                success=success,
                output="\n".join(output_parts),
                error=stderr_text if not success else None,
                data={
                    "command": command,
                    "return_code": process.returncode,
                    "working_dir": working_dir,
                },
            )
        except FileNotFoundError:
            return SkillResult(success=False, error=f"Command not found: {parts[0]}")
        except Exception as e:
            return SkillResult(success=False, error=f"Execution error: {e}")
