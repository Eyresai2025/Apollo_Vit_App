"""Create the first local Apollo administrator account.

Run from the Apollo project root:
    python tools\\create_admin_user.py
"""

from __future__ import annotations

import getpass
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.COMMON.config import get_config  # noqa: E402
from src.COMMON.security import SecurityService  # noqa: E402


def _required(prompt: str) -> str:
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("This value is required.")


def main() -> int:
    config = get_config()
    service = SecurityService(config.security)

    print("=" * 78)
    print("APOLLO VIT FIRST ADMINISTRATOR SETUP")
    print("=" * 78)
    print(f"Security database: {service.database_path}")

    if service.user_count() > 0:
        print("\nA user account already exists. Bootstrap creation is disabled.")
        print("Sign in as an Administrator and use User Management to add users.")
        return 1

    full_name = _required("Full name : ")
    username = _required("Username  : ")
    email = _required("Email     : ")

    while True:
        password = getpass.getpass("Password  : ")
        confirm = getpass.getpass("Confirm   : ")
        if password != confirm:
            print("Passwords do not match. Try again.\n")
            continue
        valid, message = service.validate_password(password)
        if not valid:
            print(f"{message}\n")
            continue
        break

    ok, message, user = service.bootstrap_admin(
        full_name=full_name,
        username=username,
        email=email,
        password=password,
        must_change_password=False,
    )
    if not ok or user is None:
        print(f"\nFAILED: {message}")
        return 1

    print("\nAdministrator created successfully.")
    print(f"Username : {user.username}")
    print(f"Role     : {user.role.value}")
    print("\nYou can now start the application with: python GUI.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
