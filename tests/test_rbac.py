from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from src.COMMON.config import SecurityConfig
from src.COMMON.security import (
    PasswordHasher,
    Permission,
    Role,
    SecurityService,
)


class RBACTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="apollo_rbac_test_")
        self.config = SecurityConfig(
            enabled=True,
            database_path=Path(self.temp_dir.name) / "security.db",
            session_timeout_minutes=1,
            max_failed_attempts=2,
            lockout_minutes=1,
            password_min_length=8,
            password_max_length=128,
            require_complex_password=True,
            allow_self_signup=False,
        )
        self.service = SecurityService(self.config)
        ok, message, admin = self.service.bootstrap_admin(
            full_name="Admin User",
            username="admin",
            email="admin@example.com",
            password="Admin1234",
            must_change_password=False,
        )
        self.assertTrue(ok, message)
        self.assertIsNotNone(admin)
        self.admin = admin

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_password_hash_is_salted_and_verifiable(self) -> None:
        first = PasswordHasher.hash_password("Password123")
        second = PasswordHasher.hash_password("Password123")
        self.assertNotEqual(first, second)
        self.assertNotIn("Password123", first)
        self.assertTrue(PasswordHasher.verify_password("Password123", first))
        self.assertFalse(PasswordHasher.verify_password("Wrong123", first))

    def test_bootstrap_admin_has_all_permissions(self) -> None:
        self.assertEqual(self.admin.role, Role.ADMIN)
        self.assertTrue(self.admin.has_permission(Permission.USER_MANAGE))
        self.assertTrue(self.admin.has_permission(Permission.PLC_SERVO_RESET))

    def test_admin_can_create_and_authenticate_operator(self) -> None:
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
        self.assertIsNotNone(operator)
        result = self.service.authenticate("operator1", "Operator123")
        self.assertTrue(result.success)
        self.assertEqual(result.user.role, Role.OPERATOR)

    def test_operator_cannot_create_users(self) -> None:
        ok, _, operator = self.service.create_user(
            actor=self.admin,
            full_name="Plant Operator",
            username="operator2",
            email="operator2@example.com",
            password="Operator123",
            role=Role.OPERATOR,
            must_change_password=False,
        )
        self.assertTrue(ok)
        ok, message, _ = self.service.create_user(
            actor=operator,
            full_name="Forbidden User",
            username="forbidden",
            email="forbidden@example.com",
            password="Forbidden123",
            role=Role.ADMIN,
        )
        self.assertFalse(ok)
        self.assertIn("Administrator", message)

    def test_failed_login_causes_lockout(self) -> None:
        self.assertFalse(self.service.authenticate("admin", "Wrong123").success)
        second = self.service.authenticate("admin", "Wrong123")
        self.assertFalse(second.success)
        self.assertEqual(second.error_code, "AUTH-LOCKED")
        blocked = self.service.authenticate("admin", "Admin1234")
        self.assertFalse(blocked.success)
        self.assertEqual(blocked.error_code, "AUTH-LOCKED")

    def test_disabled_user_cannot_login(self) -> None:
        ok, _, operator = self.service.create_user(
            actor=self.admin,
            full_name="Disabled Operator",
            username="disabled",
            email="disabled@example.com",
            password="Operator123",
            role=Role.OPERATOR,
            must_change_password=False,
        )
        self.assertTrue(ok)
        ok, message = self.service.set_user_active(self.admin, operator.user_id, False)
        self.assertTrue(ok, message)
        result = self.service.authenticate("disabled", "Operator123")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "AUTH-DISABLED")

    def test_session_expiry(self) -> None:
        session = self.service.create_session(self.admin)
        self.assertFalse(session.expired)
        session.last_activity_monotonic = time.monotonic() - 61
        self.assertTrue(session.expired)

    def test_security_audit_is_written(self) -> None:
        self.service.authenticate("admin", "Admin1234")
        events = self.service.list_audit_events()
        codes = {event["event_code"] for event in events}
        self.assertIn("USER_CREATED", codes)
        self.assertIn("LOGIN_SUCCESS", codes)


if __name__ == "__main__":
    unittest.main()
