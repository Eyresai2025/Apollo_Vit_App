from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Add the Apollo project root to Python's import path when this script is run
# directly as: python tools\\validate_user_management.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.COMMON.config import SecurityConfig
from src.COMMON.security import Role, SecurityService


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    ui_path = project_root / "src" / "Pages" / "user_management_page.py"
    ui_text = ui_path.read_text(encoding="utf-8") if ui_path.exists() else ""

    checks = {
        "UI_FILE": ui_path.exists(),
        "SEARCH": "self.search_edit" in ui_text,
        "ROLE_FILTER": "self.role_filter" in ui_text,
        "STATUS_FILTER": "self.status_filter" in ui_text,
        "DATE_FORMATTING": "_format_datetime" in ui_text,
        "SELF_DISABLE_GUARD": "cannot disable your own" in ui_text.lower(),
        "LAST_ADMIN_UI_GUARD": "final active Administrator" in ui_text,
        "CONFIRMATIONS": "Confirm role change" in ui_text and "Confirm password reset" in ui_text,
    }

    with tempfile.TemporaryDirectory(prefix="apollo_user_management_") as tmp:
        config = SecurityConfig(
            enabled=True,
            database_path=Path(tmp) / "security.db",
            session_timeout_minutes=30,
            max_failed_attempts=5,
            lockout_minutes=15,
            password_min_length=8,
            password_max_length=128,
            require_complex_password=True,
            allow_self_signup=False,
        )
        service = SecurityService(config)
        ok, _, admin1 = service.bootstrap_admin(
            full_name="Admin One",
            username="admin1",
            email="admin1@example.com",
            password="Admin1234",
            must_change_password=False,
        )
        checks["BOOTSTRAP_ADMIN"] = bool(ok and admin1)

        ok_disable, _ = service.set_user_active(admin1, admin1.user_id, False)
        checks["SELF_DISABLE_BACKEND_GUARD"] = not ok_disable

        ok, _, admin2 = service.create_user(
            actor=admin1,
            full_name="Admin Two",
            username="admin2",
            email="admin2@example.com",
            password="Admin5678",
            role=Role.ADMIN,
            must_change_password=False,
        )
        checks["SECOND_ADMIN"] = bool(ok and admin2)

        ok_demote, _ = service.set_user_role(admin1, admin2.user_id, Role.OPERATOR)
        checks["ROLE_CHANGE_WITH_TWO_ADMINS"] = ok_demote

        audit_codes = {event["event_code"] for event in service.list_audit_events()}
        checks["AUDIT"] = "USER_ROLE_CHANGED" in audit_codes

    print("=" * 78)
    print("APOLLO VIT USER MANAGEMENT VALIDATION")
    print("=" * 78)
    for name, result in checks.items():
        print(f"{name:<34}: {'OK' if result else 'FAILED'}")
    print("-" * 78)
    passed = all(checks.values())
    print(f"Status{'':<28}: {'PASSED' if passed else 'FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
