"""Fase 3c: shadow-repo versioning of identity files (git path + no-git fallback)."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.core import file_versioning
from backend.core.file_versioning import _fallback_snapshot, version_files


def git_available() -> bool:
    try:
        return subprocess.run(
            ["git", "--version"], capture_output=True, timeout=10, check=False
        ).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


class TmpDirCase(unittest.TestCase):
    def setUp(self):
        self._dir = Path(tempfile.mkdtemp())
        self.repo = self._dir / "identity_history"

    def tearDown(self):
        shutil.rmtree(self._dir, ignore_errors=True)

    def _identity(self, text="I am Tama.\n"):
        f = self._dir / "IDENTITY.md"
        f.write_text(text, encoding="utf-8")
        return f


@unittest.skipUnless(git_available(), "git not installed")
class GitVersioningTests(TmpDirCase):
    def _log(self):
        out = subprocess.run(
            ["git", "-C", str(self.repo), "log", "--oneline"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout

    def test_snapshot_becomes_commit_with_reason(self):
        f = self._identity()
        self.assertTrue(version_files([f], "consolidation: soul/identity rewrite", self.repo))
        self.assertIn("consolidation: soul/identity rewrite", self._log())

    def test_unchanged_content_is_not_committed(self):
        f = self._identity()
        self.assertTrue(version_files([f], "v1", self.repo))
        self.assertFalse(version_files([f], "v1 again", self.repo))
        self.assertNotIn("v1 again", self._log())
        # only one commit exists
        count = subprocess.run(
            ["git", "-C", str(self.repo), "rev-list", "--count", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        self.assertEqual(count, "1")

    def test_real_change_adds_second_commit(self):
        f = self._identity()
        version_files([f], "v1", self.repo)
        f.write_text("I am Tama, changed.\n", encoding="utf-8")
        self.assertTrue(version_files([f], "v2", self.repo))
        log = self._log()
        self.assertIn("v1", log)
        self.assertIn("v2", log)

    def test_missing_files_returns_false(self):
        self.assertFalse(version_files([self._dir / "nope.md"], "nothing", self.repo))


class FallbackVersioningTests(TmpDirCase):
    def test_no_git_falls_back_to_history_copies(self):
        f = self._identity()
        with patch.object(file_versioning, "_git_available", return_value=False):
            self.assertTrue(version_files([f], "onboarding: identity created", self.repo))
        snaps = list((self.repo / ".history").iterdir())
        self.assertEqual(len(snaps), 1)
        self.assertEqual((snaps[0] / "IDENTITY.md").read_text(encoding="utf-8"), "I am Tama.\n")
        self.assertEqual((snaps[0] / ".reason").read_text(encoding="utf-8"),
                         "onboarding: identity created")

    def test_pruning_keeps_newest_snapshots(self):
        history = self.repo / ".history"
        for i in range(4):
            (history / f"20200101-00000{i}").mkdir(parents=True)
        f = self._identity()
        with patch.object(file_versioning, "_MAX_SNAPSHOTS", 2):
            _fallback_snapshot(self.repo, [f], "prune test")
        kept = sorted(p.name for p in history.iterdir())
        self.assertEqual(len(kept), 2)
        self.assertIn("20200101-000003", kept)  # newest of the old ones survived

    def test_copy_failure_is_swallowed(self):
        f = self._identity()
        with patch.object(shutil, "copy2", side_effect=OSError("disk full")):
            self.assertFalse(version_files([f], "boom", self.repo))


if __name__ == "__main__":
    unittest.main()
