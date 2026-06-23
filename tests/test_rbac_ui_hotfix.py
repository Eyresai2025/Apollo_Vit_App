from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGIN = ROOT / "src" / "Pages" / "login_window.py"
GUI = ROOT / "GUI.py"


class RBACUIHotfixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.login_text = LOGIN.read_text(encoding="utf-8")
        cls.gui_text = GUI.read_text(encoding="utf-8")
        ast.parse(cls.login_text)
        ast.parse(cls.gui_text)

    def test_original_login_media_layout_is_preserved(self):
        self.assertIn("login_ai.gif", self.login_text)
        self.assertIn("showMaximized", self.login_text)

    def test_login_uses_rbac_service(self):
        self.assertIn("self.service.authenticate", self.login_text)
        self.assertIn("must_change_password", self.login_text)

    def test_public_signup_and_password_reset_are_blocked(self):
        self.assertIn("self-sign-up is disabled", self.login_text)
        self.assertIn("Password resets are managed by an Administrator", self.login_text)

    def test_qt_checked_argument_is_not_forwarded(self):
        self.assertIn("lambda _checked=False, callback=slot: callback()", self.gui_text)
        self.assertIn("isinstance(args[0], bool)", self.gui_text)


if __name__ == "__main__":
    unittest.main()
