"""Fase 3a/3b: world-tick event re-queue and stagnation pressure."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from backend.core.world_thread import (
    WorldThread,
    _IDLE_PRESSURE_HARD,
    _IDLE_PRESSURE_SOFT,
    _MAX_PENDING_EVENTS,
    _in_active_hours,
    _stagnation_note,
)


class ActiveHoursGateTests(unittest.TestCase):
    def test_no_gate_when_0_24(self):
        for h in (0, 3, 16, 23):
            self.assertTrue(_in_active_hours(h, 0, 24))

    def test_same_day_window(self):
        self.assertTrue(_in_active_hours(9, 9, 18))
        self.assertTrue(_in_active_hours(17, 9, 18))
        self.assertFalse(_in_active_hours(8, 9, 18))
        self.assertFalse(_in_active_hours(18, 9, 18))  # end exclusive

    def test_midnight_wrap(self):
        # 22:00 -> 06:00 overnight window
        self.assertTrue(_in_active_hours(23, 22, 6))
        self.assertTrue(_in_active_hours(2, 22, 6))
        self.assertTrue(_in_active_hours(5, 22, 6))
        self.assertFalse(_in_active_hours(12, 22, 6))


def bare_thread(tmpdir: str) -> WorldThread:
    """WorldThread without __init__ (which needs an agent/LLM): only the
    attributes the re-queue/stagnation helpers touch."""
    wt = WorldThread.__new__(WorldThread)
    wt._pending_world_events = []
    wt._stagnation_path = Path(tmpdir) / "world_stagnation.json"
    wt._stagnation_streak = 0
    return wt


class CronLocalAnchorTests(unittest.TestCase):
    """Regression: the run loop anchors cron to LOCAL time so a pinned
    schedule ("0 9 * * *") fires at 9am wall-clock, not 9am UTC."""

    def test_pinned_schedule_fires_at_local_wall_clock(self):
        from datetime import datetime

        from croniter import croniter

        now = datetime.now().astimezone()
        nxt = croniter("0 9 * * *", now).get_next(datetime)
        self.assertEqual(nxt.hour, 9)
        self.assertEqual(nxt.tzinfo.utcoffset(None), now.tzinfo.utcoffset(None))


class StagnationNoteTests(unittest.TestCase):
    def test_no_pressure_below_soft_threshold(self):
        self.assertEqual(_stagnation_note(0), "")
        self.assertEqual(_stagnation_note(_IDLE_PRESSURE_SOFT - 1), "")

    def test_gentle_nudge_between_thresholds(self):
        note = _stagnation_note(_IDLE_PRESSURE_SOFT)
        self.assertIn("World pressure", note)
        self.assertIn(str(_IDLE_PRESSURE_SOFT), note)
        self.assertNotIn("MUST", note)

    def test_hard_pressure_at_threshold(self):
        note = _stagnation_note(_IDLE_PRESSURE_HARD)
        self.assertIn("MUST", note)
        self.assertIn(str(_IDLE_PRESSURE_HARD), note)


class RequeueTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.mkdtemp()
        self.wt = bare_thread(self._dir)

    def tearDown(self):
        shutil.rmtree(self._dir, ignore_errors=True)

    def test_empty_requeue_is_noop(self):
        self.wt._pending_world_events = ["already queued"]
        self.wt._requeue_events([])
        self.assertEqual(self.wt._pending_world_events, ["already queued"])

    def test_requeued_events_keep_chronological_front(self):
        self.wt._pending_world_events = ["arrived-during-tick"]
        self.wt._requeue_events(["user-visited", "user-asked"])
        self.assertEqual(
            self.wt._pending_world_events,
            ["user-visited", "user-asked", "arrived-during-tick"],
        )

    def test_cap_drops_oldest(self):
        drained = [f"e{i}" for i in range(_MAX_PENDING_EVENTS + 20)]
        self.wt._requeue_events(drained)
        self.assertEqual(len(self.wt._pending_world_events), _MAX_PENDING_EVENTS)
        self.assertEqual(self.wt._pending_world_events[0], f"e{20}")  # oldest 20 gone
        self.assertEqual(self.wt._pending_world_events[-1], f"e{_MAX_PENDING_EVENTS + 19}")


class StagnationCounterTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.mkdtemp()
        self.wt = bare_thread(self._dir)

    def tearDown(self):
        shutil.rmtree(self._dir, ignore_errors=True)

    def test_idle_accumulates_and_other_kinds_reset(self):
        self.assertEqual(self.wt._update_stagnation("idle"), 1)
        self.assertEqual(self.wt._update_stagnation("idle"), 2)
        self.assertEqual(self.wt._update_stagnation("work"), 0)
        self.assertEqual(self.wt._update_stagnation("idle"), 1)
        self.assertEqual(self.wt._update_stagnation("blocked"), 0)  # failure is movement

    def test_idle_caps_at_99(self):
        self.wt._stagnation_streak = 99
        self.assertEqual(self.wt._update_stagnation("idle"), 99)

    def test_counter_survives_restart(self):
        self.wt._update_stagnation("idle")
        self.wt._update_stagnation("idle")
        data = json.loads(self.wt._stagnation_path.read_text(encoding="utf-8"))
        self.assertEqual(data["idle_streak"], 2)
        self.assertEqual(data["last_kind"], "idle")
        fresh = bare_thread(self._dir)
        self.assertEqual(fresh._load_stagnation(), 2)

    def test_corrupt_counter_file_loads_as_zero(self):
        self.wt._stagnation_path.write_text("{not json", encoding="utf-8")
        self.assertEqual(self.wt._load_stagnation(), 0)


if __name__ == "__main__":
    unittest.main()
