import unittest

from backend.core.agent import _TOOL_RESULT_MAX_CHARS, _clamp_tool_payload


class ClampToolPayloadTests(unittest.TestCase):
    def test_small_payloads_pass_through_unchanged(self):
        payload = {"success": True, "output": "pose set to wave"}
        self.assertEqual(_clamp_tool_payload(payload), payload)

    def test_oversized_output_keeps_head_and_tail_with_marker(self):
        value = "H" * 20_000 + "M" * 20_000 + "T" * 20_000
        clamped = _clamp_tool_payload({"output": value})["output"]
        self.assertLess(len(clamped), len(value))
        self.assertIn("truncated: 60,000 chars total", clamped)
        self.assertTrue(clamped.startswith("H" * 10_000))
        self.assertTrue(clamped.endswith("T" * 10_000))

    def test_clamped_fields_stay_near_the_limit(self):
        clamped = _clamp_tool_payload({"error": "E" * 40_000})["error"]
        self.assertLessEqual(len(clamped), _TOOL_RESULT_MAX_CHARS + 500)

    def test_non_string_and_nested_values_are_left_alone(self):
        payload = {"success": True, "data": {"blob": "x" * 40_000}}
        self.assertEqual(_clamp_tool_payload(payload), payload)

    def test_original_payload_is_not_mutated(self):
        payload = {"output": "y" * 40_000}
        _clamp_tool_payload(payload)
        self.assertEqual(len(payload["output"]), 40_000)


if __name__ == "__main__":
    unittest.main()
