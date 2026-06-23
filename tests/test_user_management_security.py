from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.COMMON.config import SecurityConfig
from src.COMMON.security import Role, SecurityService


class UserManagementSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="apollo_user_mgmt_")
        config = SecurityConfig(
            enabled=True,
            database_path=Path(self.temp_dir.name) / "security.db",
            session_timeout_minutes=30,
            max_failed_attempts=5,
            lockout_minutes=15,
            password_min_length=8,
            password_max_length=128,
            require_complex_password=True,
            allow_self_signup=False,
        )
        self.service = SecurityService(config)
        ok, message, admin = self.service.bootstrap_admin(
            full_name="Primary Admin",
            username="admin1",
            email="admin1@example.com",
            password="Admin1234",
            must_change_password=False,
        )
        self.assertTrue(ok, message)
        self.admin = admin

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_final_active_admin_cannot_be_disabled(self) -> None:
        ok, message = self.service.set_user_active(
            self.admin,
            self.admin.user_id,
            False,
        )
        self.assertFalse(ok)
        self.assertIn("cannot disable your own", message.lower())

    def test_final_active_admin_cannot_be_demoted(self) -> None:
        ok, message = self.service.set_user_role(
            self.admin,
            self.admin.user_id,
            Role.OPERATOR,
        )
        self.assertFalse(ok)
        self.assertIn("cannot remove your own", message.lower())

    def test_one_admin_can_change_another_after_second_admin_exists(self) -> None:
        ok, message, second_admin = self.service.create_user(
            actor=self.admin,
            full_name="Second Admin",
            username="admin2",
            email="admin2@example.com",
            password="Admin5678",
            role=Role.ADMIN,
            must_change_password=False,
        )
        self.assertTrue(ok, message)

        ok, message = self.service.set_user_role(
            self.admin,
            second_admin.user_id,
            Role.QUALITY_ENGINEER,
        )
        self.assertTrue(ok, message)

    def test_status_role_and_reset_actions_are_audited(self) -> None:
        ok, message, operator = self.service.create_user(
            actor=self.admin,
            full_name="Plant Operator",
            username="operator1",
            email="operator1@example.com",
            password="Operator123",
            role=Role.OPERATOR,
            must_change_password=False,
        )
        self.assertTrue(ok, message)

        self.assertTrue(self.service.set_user_role(
            self.admin, operator.user_id, Role.MAINTENANCE
        )[0])
        self.assertTrue(self.service.admin_reset_password(
            self.admin, operator.user_id, "Changed123", require_change=True
        )[0])
        self.assertTrue(self.service.set_user_active(
            self.admin, operator.user_id, False
        )[0])

        codes = {event["event_code"] for event in self.service.list_audit_events()}
        self.assertIn("USER_ROLE_CHANGED", codes)
        self.assertIn("USER_PASSWORD_RESET", codes)
        self.assertIn("USER_STATUS_CHANGED", codes)


if __name__ == "__main__":
    unittest.main()
