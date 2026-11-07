"""
Identity file versioning — every rewrite of Tama's soul leaves a trace.

Tama's identity files (IDENTITY.md / SOUL.md / USER.md / TASKS.md) are what
she IS. Consolidation and self-editing rewrite them, so a bad or surprising
rewrite must be recoverable and diffable. This module keeps a SHADOW git
repo at data/identity_history/ (its own .git — the user's workspace is never
touched): each snapshot copies the current files in and commits with a
reason. Without git (or if git fails), falls back to timestamped copies in
<repo>/.history/, pruned to the newest _MAX_SNAPSHOTS.

Deliberately synchronous and best-effort: call it right after a successful
identity write; failures are logged, never raised into the write path.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("tamagi.file_versioning")

DEFAULT_REPO_DIR = "data/identity_history"
_MAX_SNAPSHOTS = 50
_GIT_TIMEOUT = 20


def _git_available() -> bool:
    return shutil.which("git") is not None


def _run_git(repo_dir: Path, *args: str) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_dir), *args],
            capture_output=True, timeout=_GIT_TIMEOUT, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _ensure_repo(repo_dir: Path) -> bool:
    """Init the shadow repo on first use. Returns False when git is unusable."""
    if (repo_dir / ".git").exists():
        return True
    repo_dir.mkdir(parents=True, exist_ok=True)
    if not _git_available():
        return False
    if _run_git(repo_dir, "init", "-q") is None:
        return False
    # Local identity so commits work on machines with no global git config.
    _run_git(repo_dir, "config", "user.name", "TamAGI")
    _run_git(repo_dir, "config", "user.email", "tamagi@local")
    return True


def _fallback_snapshot(repo_dir: Path, files: list[Path], reason: str) -> bool:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    snap = repo_dir / ".history" / stamp
    snap.mkdir(parents=True, exist_ok=True)
    for f in files:
        if f.exists():
            shutil.copy2(f, snap / f.name)
    (snap / ".reason").write_text(reason, encoding="utf-8")
    # Prune oldest snapshots beyond the cap.
    snaps = sorted((repo_dir / ".history").iterdir())
    for old in snaps[:-_MAX_SNAPSHOTS]:
        shutil.rmtree(old, ignore_errors=True)
    logger.info("Identity snapshot (no-git fallback): %s", snap)
    return True


def version_files(
    files: list[Path],
    reason: str,
    repo_dir: str | Path = DEFAULT_REPO_DIR,
) -> bool:
    """Snapshot `files` into the shadow repo with a reason. True if stored.

    No-op (returns False) when none of the files exist or their content is
    unchanged since the last snapshot — the history stays a record of actual
    changes, not of every write call.
    """
    present = [f for f in files if f.exists() and f.is_file()]
    if not present:
        return False
    repo = Path(repo_dir)
    try:
        if _ensure_repo(repo):
            for f in present:
                shutil.copy2(f, repo / f.name)
            status = _run_git(repo, "status", "--porcelain")
            if status is None:
                return _fallback_snapshot(repo, present, reason)
            if not status.stdout.strip():
                return False  # nothing actually changed
            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            _run_git(repo, "add", "-A")
            done = _run_git(repo, "commit", "-m", f"{reason} — {ts}")
            if done is None or done.returncode != 0:
                return _fallback_snapshot(repo, present, reason)
            logger.info("Identity versioned (%s): %s", repo, reason)
            return True
        return _fallback_snapshot(repo, present, reason)
    except Exception:
        # Versioning must never take down identity persistence.
        logger.warning("Identity versioning failed for %r", reason, exc_info=True)
        return False
