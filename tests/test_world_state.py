import unittest

from backend.core.world_state import parse_new_state


class ParseNewStateTests(unittest.TestCase):
    def assert_state_fields(self, labels: str) -> None:
        response = f"""[New State]
{labels}Location/Setting{labels}: El taller junto a la ventana.
{labels}Internal State/Mood{labels}: Tranquila y concentrada.
{labels}Current Focus{labels}: Terminar la revisión.
{labels}Available Actions{labels}:
- Abrir los archivos
- Ejecutar las pruebas
"""
        state = parse_new_state(response)
        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual(state.location, "El taller junto a la ventana")
        self.assertEqual(state.mood, "Tranquila y concentrada")
        self.assertEqual(state.focus, "Terminar la revisión")
        self.assertEqual(state.available_actions, ["Abrir los archivos", "Ejecutar las pruebas"])

    def test_parses_plain_field_labels(self) -> None:
        self.assert_state_fields("")

    def test_parses_bold_labels_with_colon_inside(self) -> None:
        response = """[New State]
**Location/Setting:** El taller junto a la ventana.
**Internal State/Mood:** Tranquila y concentrada.
**Current Focus:** Terminar la revisión.
**Available Actions:**
- Abrir los archivos
- Ejecutar las pruebas
"""
        state = parse_new_state(response)
        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual(state.location, "El taller junto a la ventana")
        self.assertEqual(state.mood, "Tranquila y concentrada")
        self.assertEqual(state.focus, "Terminar la revisión")
        self.assertEqual(state.available_actions, ["Abrir los archivos", "Ejecutar las pruebas"])

    def test_parses_bold_labels_with_colon_outside(self) -> None:
        response = """[New State]
**Location/Setting**: El taller junto a la ventana.
**Internal State/Mood**: Tranquila y concentrada.
**Current Focus**: Terminar la revisión.
**Available Actions**:
- Abrir los archivos
- Ejecutar las pruebas
"""
        state = parse_new_state(response)
        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual(state.location, "El taller junto a la ventana")
        self.assertEqual(state.mood, "Tranquila y concentrada")
        self.assertEqual(state.focus, "Terminar la revisión")
        self.assertEqual(state.available_actions, ["Abrir los archivos", "Ejecutar las pruebas"])

    def test_keeps_original_markdown_block(self) -> None:
        response = """[New State]
**Location/Setting:** Escritorio.
**Internal State/Mood:** Atenta.
**Current Focus:** Parser.
**Available Actions:**
- Validar cambios
"""
        state = parse_new_state(response)
        self.assertIsNotNone(state)
        assert state is not None
        self.assertIn("**Location/Setting:** Escritorio.", state.raw_state_block)


if __name__ == "__main__":
    unittest.main()
