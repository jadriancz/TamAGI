"""Read skill: whole-file size cap + streamed max_lines paging."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backend.skills.read_skill import ReadSkill


class ReadSizeGuardTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._dir = tempfile.mkdtemp()
        self.workspace = Path(self._dir) / "workspace"
        self.workspace.mkdir()
        self.config = SimpleNamespace(
            workspace=SimpleNamespace(path=str(self.workspace)),
            guardrails=SimpleNamespace(
                allowed_read_paths=[str(self.workspace)],
                max_read_size=512,  # tiny for test speed
            ),
        )
        patcher = patch("backend.skills.read_skill.get_config", return_value=self.config)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.skill = ReadSkill()

    async def test_small_file_reads_fully(self):
        (self.workspace / "note.md").write_text("line1\nline2\n", encoding="utf-8")
        result = await self.skill.execute(path="note.md")
        self.assertTrue(result.success)
        self.assertIn("line2", result.output)

    async def test_oversized_file_without_max_lines_is_refused(self):
        big = self.workspace / "big.log"
        big.write_text("x" * 4000, encoding="utf-8")
        result = await self.skill.execute(path="big.log")
        self.assertFalse(result.success)
        self.assertIn("read limit", result.error)
        self.assertIn("max_lines", result.error)

    async def test_oversized_file_with_max_lines_pages_streamed(self):
        big = self.workspace / "big.log"
        big.write_text("".join(f"line {i}\n" for i in range(200)), encoding="utf-8")
        result = await self.skill.execute(path="big.log", max_lines=5)
        self.assertTrue(result.success)
        self.assertIn("line 0", result.output)
        self.assertIn("line 4", result.output)
        self.assertNotIn("line 5\n", result.output)
        self.assertIn("(195 more lines)", result.output)

    async def test_exact_limit_still_reads(self):
        (self.workspace / "edge.txt").write_text("y" * 512, encoding="utf-8")
        result = await self.skill.execute(path="edge.txt")
        self.assertTrue(result.success)

    async def test_binary_file_still_reports_encoding_error(self):
        (self.workspace / "blob.bin").write_bytes(b"\xff\xfe\x00binary")
        result = await self.skill.execute(path="blob.bin")
        self.assertFalse(result.success)
        self.assertIn("Binary file", result.error)


if __name__ == "__main__":
    unittest.main()
