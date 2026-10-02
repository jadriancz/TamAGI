import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from backend.config import TamAGIConfig
from backend.skills.base import SkillResult
from backend.skills.exec_skill import ExecSkill, _command_parts


class ExecSkillTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = TamAGIConfig()
        self.skill = ExecSkill()
        self.skill._run = AsyncMock(return_value=SkillResult(success=True, output="Done"))
        self.config_patch = patch("backend.skills.exec_skill.get_config", return_value=self.config)
        self.config_patch.start()
        self.addCleanup(self.config_patch.stop)
        trusted_patch = patch("backend.skills.exec_skill._runtime_trusted", set())
        trusted_patch.start()
        self.addCleanup(trusted_patch.stop)

    async def test_structured_args_preserve_windows_paths_and_spaces(self):
        path = r"C:\Users\User Name\Documents\Project"
        events = AsyncMock()
        result = await self.skill.execute(
            program="git", args=["-C", path, "status", "--short"], _event_callback=events,
        )
        self.assertTrue(result.success)
        self.assertEqual(self.skill._run.call_args.args[1], ["git", "-C", path, "status", "--short"])
        events.assert_awaited_once()

    async def test_simple_legacy_invocation_remains_supported(self):
        result = await self.skill.execute(command="git status --short")
        self.assertTrue(result.success)
        self.assertEqual(self.skill._run.call_args.args[1], ["git", "status", "--short"])

    async def test_compound_legacy_commands_are_rejected_before_launch(self):
        for command in (
            "git status;git log", "ls data && ls models", "find data | sort",
            "git status > report.txt", "git status 2>/dev/null", "ls & ls", "echo $(pwd)",
        ):
            with self.subTest(command=command):
                result = await self.skill.execute(command=command)
                self.assertFalse(result.success)
                self.assertIn("Shell operators", result.error)
        self.skill._run.assert_not_awaited()

    async def test_structured_args_are_literal_not_shell_syntax(self):
        result = await self.skill.execute(program="git", args=["log", "--format=a;b|c"])
        self.assertTrue(result.success)
        self.assertEqual(self.skill._run.call_args.args[1][-1], "--format=a;b|c")

    async def test_invalid_invocations_are_rejected(self):
        for kwargs in (
            {}, {"program": ""}, {"program": "git", "args": "status"},
            {"program": "git", "args": [1]}, {"command": None},
            {"program": "git", "command": "ls"}, {"command": "git log", "args": ["x"]},
            {"program": "git", "args": ["\x00"]},
        ):
            with self.subTest(kwargs=kwargs):
                self.assertFalse((await self.skill.execute(**kwargs)).success)
        self.skill._run.assert_not_awaited()

    async def test_native_exe_basename_uses_existing_trust_tier(self):
        result = await self.skill.execute(program=r"C:\Program Files\Git\cmd\git.exe", args=["status"])
        self.assertTrue(result.success)
        self.skill._run.assert_awaited_once()

    async def test_blocked_program_remains_blocked_by_absolute_path(self):
        self.config.guardrails.exec_trust.block = ["danger"]
        result = await self.skill.execute(program=r"C:\bin\danger.exe", args=[])
        self.assertFalse(result.success)
        self.assertIn("blocked", result.error)
        self.skill._run.assert_not_awaited()

    async def test_destructive_args_require_approval(self):
        self.skill._request_approval = AsyncMock(return_value=(False, False))
        result = await self.skill.execute(program="git", args=["push", "--force"])
        self.assertFalse(result.success)
        self.skill._request_approval.assert_awaited_once()
        self.skill._run.assert_not_awaited()

    async def test_shell_interpreter_requires_approval_even_if_trusted(self):
        self.config.guardrails.exec_trust.safe = ["powershell"]
        self.skill._request_approval = AsyncMock(return_value=(False, False))
        result = await self.skill.execute(program="powershell.exe", args=["-Command", "Get-Location"])
        self.assertFalse(result.success)
        self.skill._request_approval.assert_awaited_once()
        self.skill._run.assert_not_awaited()

    async def test_shell_interpreter_is_denied_autonomously(self):
        self.config.guardrails.exec_trust.safe = ["bash"]
        result = await self.skill.execute(program="bash", args=["-c", "pwd"], _is_autonomous=True)
        self.assertFalse(result.success)
        self.skill._run.assert_not_awaited()

    def test_windows_legacy_parsing_preserves_backslashes(self):
        with patch("backend.skills.exec_skill.os.name", "nt"):
            self.assertEqual(
                _command_parts(r'git -C "C:\Users\User Name\Project" status'),
                ["git", "-C", r"C:\Users\User Name\Project", "status"],
            )

    def test_tool_schema_describes_array_items_and_environment(self):
        tool = self.skill.to_openai_tool()["function"]
        self.assertEqual(tool["parameters"]["properties"]["args"]["items"], {"type": "string"})
        self.assertIn("NOT a shell", tool["description"])
        self.assertNotIn("command", tool["parameters"]["required"])


class ExecProcessTests(unittest.IsolatedAsyncioTestCase):
    async def test_process_is_launched_with_exact_argv_and_cwd(self):
        process = SimpleNamespace(
            communicate=AsyncMock(return_value=(b"result", b"")), returncode=0,
        )
        with patch("backend.skills.exec_skill.asyncio.create_subprocess_exec", AsyncMock(return_value=process)) as launch:
            result = await ExecSkill()._run("git status", ["git", "status"], "work-dir", 30)
        self.assertTrue(result.success)
        self.assertEqual(result.output, "result")
        self.assertEqual(launch.call_args.args, ("git", "status"))
        self.assertEqual(launch.call_args.kwargs["cwd"], "work-dir")

    async def test_stderr_and_nonzero_exit_remain_failures(self):
        process = SimpleNamespace(
            communicate=AsyncMock(return_value=(b"", b"bad argument")), returncode=2,
        )
        with patch("backend.skills.exec_skill.asyncio.create_subprocess_exec", AsyncMock(return_value=process)):
            result = await ExecSkill()._run("git status", ["git", "status"], ".", 30)
        self.assertFalse(result.success)
        self.assertIn("bad argument", result.error)
        self.assertEqual(result.data["return_code"], 2)

    async def test_timeout_kills_process_and_returns_clear_error(self):
        process = SimpleNamespace(
            communicate=AsyncMock(side_effect=TimeoutError()), kill=Mock(), wait=AsyncMock(),
        )
        with patch("backend.skills.exec_skill.asyncio.create_subprocess_exec", AsyncMock(return_value=process)):
            result = await ExecSkill()._run("git status", ["git", "status"], ".", 30)
        self.assertFalse(result.success)
        self.assertIn("timed out", result.error)
        process.kill.assert_called_once()


if __name__ == "__main__":
    unittest.main()
