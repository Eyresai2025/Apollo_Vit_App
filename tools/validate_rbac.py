"""Validate Apollo role-based access control without starting the GUI."""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.COMMON.config import SecurityConfig, get_config_manager  # noqa: E402
from src.COMMON.security import (  # noqa: E402
    ALL_PERMISSIONS,
    Permission,
    Role,
    SecurityService,
)


def main() -> int:
    manager = get_config_manager()
    config = manager.config
    checks = []

    checks.append(("CONFIG", config.security.session_timeout_minutes >= 1))
    checks.append(("ROLE_COUNT", len(Role) == 5))
    checks.append(("ADMIN_PERMISSIONS", SecurityService.permissions_for_role(Role.ADMIN) == frozenset(ALL_PERMISSIONS)))
    checks.append(("OPERATOR_LIVE", Permission.INSPECTION_RUN.value in SecurityService.permissions_for_role(Role.OPERATOR)))
    checks.append(("OPERATOR_NO_ADMIN", Permission.USER_MANAGE.value not in SecurityService.permissions_for_role(Role.OPERATOR)))

    with tempfile.TemporaryDirectory(prefix="apollo_rbac_") as tmp:
        test_config = SecurityConfig(
            enabled=True,
            database_path=Path(tmp) / "security_test.db",
            session_timeout_minutes=1,
            max_failed_attempts=3,
            lockout_minutes=1,
            password_min_length=8,
            password_max_length=128,
            require_complex_password=True,
            allow_self_signup=False,
        )
        service = SecurityService(test_config)
        ok, _, admin = service.bootstrap_admin(
            full_name="Validation Administrator",
            username="validation_admin",
            email="validation@example.com",
            password="Validation123",
            must_change_password=False,
        )
        checks.append(("BOOTSTRAP_ADMIN", bool(ok and admin)))
        auth = service.authenticate("validation_admin", "Validation123")
        checks.append(("AUTHENTICATION", bool(auth.success and auth.user)))
        if auth.user:
            session = service.create_session(auth.user)
            checks.append(("SESSION", not session.expired and session.remaining_seconds > 0))
            ok_user, _, operator = service.create_user(
                actor=auth.user,
                full_name="Validation Operator",
                username="validation_operator",
                email="operator@example.com",
                password="Operator123",
                role=Role.OPERATOR,
            )
            checks.append(("ADMIN_CREATE_USER", bool(ok_user and operator)))
        checks.append(("AUDIT", len(service.list_audit_events()) >= 2))

    configured_service = SecurityService(config.security)
    user_count = configured_service.user_count()

    print("=" * 78)
    print("APOLLO VIT ROLE-BASED ACCESS VALIDATION")
    print("=" * 78)
    print(f"Enabled          : {config.security.enabled}")
    print(f"Security DB      : {config.security.database_path}")
    print(f"Session timeout  : {config.security.session_timeout_minutes} minutes")
    print(f"Maximum failures : {config.security.max_failed_attempts}")
    print(f"Lockout          : {config.security.lockout_minutes} minutes")
    print(f"Configured users : {user_count}")
    print("-" * 78)
    for name, passed in checks:
        print(f"{name:<22}: {'OK' if passed else 'FAILED'}")

    failed = [name for name, passed in checks if not passed]
    print("-" * 78)
    if failed:
        print(f"Status           : FAILED ({', '.join(failed)})")
        return 1
    if config.security.enabled and user_count == 0:
        print("Status           : PASSED_WITH_SETUP_REQUIRED")
        print("Next command     : python tools\\create_admin_user.py")
        return 0
    print("Status           : PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
