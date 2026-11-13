"""Tests for the search_file skill."""

import unittest

from backend.skills.search_skill import SearchFileSkill


class SearchFileSkillTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_simple_glob(self):
        skill = SearchFileSkill()
        result = await skill.execute(pattern="*.py")
        self.assertTrue(result.success)
        self.assertIn("Encontré", result.output)
        self.assertIn("matches", result.data)

    async def test_search_python_files(self):
        skill = SearchFileSkill()
        result = await skill.execute(pattern="*.py", max_results=5)
        self.assertTrue(result.success, f"Error: {result.error}")
        self.assertGreater(len(result.data.get("matches", [])), 0)

    async def test_empty_pattern(self):
        skill = SearchFileSkill()
        result = await skill.execute(pattern="")
        self.assertFalse(result.success)
        self.assertIn("requerido", result.error)


if __name__ == "__main__":
    unittest.main()