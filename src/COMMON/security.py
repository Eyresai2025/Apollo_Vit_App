"""Local role-based access control for Apollo VIT.

The security database is intentionally local and independent of MongoDB so an
operator can still authenticate when the plant network or central database is
unavailable. Passwords are stored with ``hashlib.scrypt`` and a unique random
salt; plaintext passwords are never persisted.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

from src.COMMON.config import SecurityConfig, get_config
from src.COMMON.structured_logging import get_logger

logger = get_logger(__name__, component="SECURITY")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Optional[datetime] = None) -> str:
    return (value or _utc_now()).isoformat(timespec="seconds")


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


class Role(str, Enum):
    ADMIN = "ADMIN"
    OPERATOR = "OPERATOR"
    QUALITY_ENGINEER = "QUALITY_ENGINEER"
    MAINTENANCE = "MAINTENANCE"
    AI_ENGINEER = "AI_ENGINEER"


class Permission(str, Enum):
    INSPECTION_RUN = "inspection.run"
    HARDWARE_TEST = "hardware.test"
    DEVICE_CONFIGURE = "device.configure"
    CAPTURE_CONFIGURE = "capture.configure"
    AXIS_VIEW = "axis.view"
    SKU_MANAGE = "sku.manage"
    RECIPE_MANAGE = "recipe.manage"
    REPEATABILITY_RUN = "repeatability.run"
    OSC_MANAGE = "osc.manage"
    DASHBOARD_VIEW = "dashboard.view"
    ANNOTATION_USE = "annotation.use"
    ROI_MEASURE = "roi.measure"
    PLC_AUTO_START = "plc.auto_start"
    PLC_SERVO_RESET = "plc.servo_reset"
    USER_MANAGE = "user.manage"
    SECURITY_AUDIT_VIEW = "security.audit.view"


ALL_PERMISSIONS: Set[str] = {permission.value for permission in Permission}

ROLE_PERMISSIONS: Mapping[Role, Set[str]] = {
    Role.ADMIN: set(ALL_PERMISSIONS),
    Role.OPERATOR: {
        Permission.INSPECTION_RUN.value,
        Permission.DASHBOARD_VIEW.value,
    },
    Role.QUALITY_ENGINEER: {
        Permission.INSPECTION_RUN.value,
        Permission.REPEATABILITY_RUN.value,
        Permission.OSC_MANAGE.value,
        Permission.DASHBOARD_VIEW.value,
        Permission.ANNOTATION_USE.value,
        Permission.ROI_MEASURE.value,
    },
    Role.MAINTENANCE: {
        Permission.HARDWARE_TEST.value,
        Permission.DEVICE_CONFIGURE.value,
        Permission.CAPTURE_CONFIGURE.value,
        Permission.AXIS_VIEW.value,
        Permission.DASHBOARD_VIEW.value,
        Permission.PLC_AUTO_START.value,
        Permission.PLC_SERVO_RESET.value,
    },
    Role.AI_ENGINEER: {
        Permission.SKU_MANAGE.value,
        Permission.RECIPE_MANAGE.value,
        Permission.DASHBOARD_VIEW.value,
        Permission.ANNOTATION_USE.value,
        Permission.ROI_MEASURE.value,
        Permission.REPEATABILITY_RUN.value,
    },
}


@dataclass(frozen=True)
class UserPrincipal:
    user_id: int
    username: str
    full_name: str
    email: str
    role: Role
    permissions: frozenset[str]
    must_change_password: bool = False

    def has_permission(self, permission: str | Permission) -> bool:
        value = permission.value if isinstance(permission, Permission) else str(permission)
        return value in self.permissions

    def to_safe_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "full_name": self.full_name,
            "email": self.email,
            "role": self.role.value,
            "permissions": sorted(self.permissions),
            "must_change_password": self.must_change_password,
        }


@dataclass(frozen=True)
class AuthResult:
    success: bool
    message: str
    user: Optional[UserPrincipal] = None
    error_code: Optional[str] = None
    locked_until: Optional[datetime] = None


@dataclass
class SessionContext:
    session_id: str
    user: UserPrincipal
    timeout_seconds: int
    created_monotonic: float
    last_activity_monotonic: float
    closed: bool = False

    @classmethod
    def create(cls, user: UserPrincipal, timeout_minutes: int) -> "SessionContext":
        now = time.monotonic()
        return cls(
            session_id=str(uuid.uuid4()),
            user=user,
            timeout_seconds=max(60, int(timeout_minutes) * 60),
            created_monotonic=now,
            last_activity_monotonic=now,
        )

    def touch(self) -> None:
        if not self.closed:
            self.last_activity_monotonic = time.monotonic()

    @property
    def expired(self) -> bool:
        return self.closed or (time.monotonic() - self.last_activity_monotonic) >= self.timeout_seconds

    @property
    def remaining_seconds(self) -> int:
        if self.closed:
            return 0
        return max(0, int(self.timeout_seconds - (time.monotonic() - self.last_activity_monotonic)))

    def close(self) -> None:
        self.closed = True


class PasswordHasher:
    """Versioned scrypt password hashing and verification."""

    ALGORITHM = "scrypt"
    N = 2**14
    R = 8
    P = 1
    DKLEN = 32

    @classmethod
    def hash_password(cls, password: str) -> str:
        salt = secrets.token_bytes(16)
        digest = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=cls.N,
            r=cls.R,
            p=cls.P,
            dklen=cls.DKLEN,
        )
        return "$".join(
            [
                cls.ALGORITHM,
                str(cls.N),
                str(cls.R),
                str(cls.P),
                salt.hex(),
                digest.hex(),
            ]
        )

    @classmethod
    def verify_password(cls, password: str, encoded: str) -> bool:
        try:
            algorithm, n, r, p, salt_hex, digest_hex = encoded.split("$", 5)
            if algorithm != cls.ALGORITHM:
                return False
            actual = hashlib.scrypt(
                password.encode("utf-8"),
                salt=bytes.fromhex(salt_hex),
                n=int(n),
                r=int(r),
                p=int(p),
                dklen=len(bytes.fromhex(digest_hex)),
            )
            return hmac.compare_digest(actual, bytes.fromhex(digest_hex))
        except (ValueError, TypeError):
            return False


class SecurityService:
    """Thread-safe SQLite-backed user, role, session and audit service."""

    def __init__(
        self,
        config: Optional[SecurityConfig] = None,
        database_path: Optional[Path | str] = None,
    ) -> None:
        self.config = config or get_config().security
        self.database_path = Path(database_path or self.config.database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open one SQLite transaction and always release the file handle.

        ``sqlite3.Connection`` commits or rolls back when used as a context
        manager, but it does not close itself. Explicit closing is required on
        Windows so temporary RBAC databases can be removed after validation
        and so long-running GUI sessions do not leak database handles.
        """
        conn = sqlite3.connect(
            str(self.database_path),
            timeout=10.0,
            isolation_level=None,
            check_same_thread=False,
        )
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            with conn:
                yield conn
        finally:
            conn.close()

    def _initialize_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    full_name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    failed_attempts INTEGER NOT NULL DEFAULT 0,
                    locked_until TEXT,
                    must_change_password INTEGER NOT NULL DEFAULT 0,
                    last_login_at TEXT,
                    password_changed_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS security_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    occurred_at TEXT NOT NULL,
                    actor_user_id INTEGER,
                    actor_username TEXT,
                    event_code TEXT NOT NULL,
                    target_user_id INTEGER,
                    target_username TEXT,
                    outcome TEXT NOT NULL,
                    details_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_security_audit_occurred
                    ON security_audit(occurred_at DESC);
                CREATE INDEX IF NOT EXISTS idx_security_audit_actor
                    ON security_audit(actor_username);
                """
            )

    @staticmethod
    def _normalize_username(value: str) -> str:
        return value.strip()

    @staticmethod
    def _normalize_email(value: str) -> str:
        return value.strip().lower()

    def validate_password(self, password: str) -> Tuple[bool, str]:
        if len(password) < self.config.password_min_length:
            return False, f"Password must be at least {self.config.password_min_length} characters."
        if len(password) > self.config.password_max_length:
            return False, f"Password cannot exceed {self.config.password_max_length} characters."
        if self.config.require_complex_password:
            if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
                return False, "Password must contain at least one letter and one number."
        return True, ""

    @staticmethod
    def _validate_identity(username: str, email: str, full_name: str) -> Tuple[bool, str]:
        if not full_name.strip():
            return False, "Full name is required."
        if not re.fullmatch(r"[A-Za-z0-9_.-]{3,64}", username.strip()):
            return False, "Username must be 3–64 characters using letters, numbers, dot, underscore or hyphen."
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email.strip()):
            return False, "A valid email address is required."
        return True, ""

    @staticmethod
    def permissions_for_role(role: Role | str) -> frozenset[str]:
        resolved = role if isinstance(role, Role) else Role(str(role).upper())
        return frozenset(ROLE_PERMISSIONS[resolved])

    def _row_to_principal(self, row: sqlite3.Row) -> UserPrincipal:
        role = Role(str(row["role"]).upper())
        return UserPrincipal(
            user_id=int(row["id"]),
            username=str(row["username"]),
            full_name=str(row["full_name"]),
            email=str(row["email"]),
            role=role,
            permissions=self.permissions_for_role(role),
            must_change_password=bool(row["must_change_password"]),
        )

    def user_count(self) -> int:
        with self._lock, self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])

    def create_user(
        self,
        *,
        full_name: str,
        username: str,
        email: str,
        password: str,
        role: Role | str,
        actor: Optional[UserPrincipal] = None,
        must_change_password: bool = True,
        bootstrap: bool = False,
    ) -> Tuple[bool, str, Optional[UserPrincipal]]:
        username = self._normalize_username(username)
        email = self._normalize_email(email)
        valid, message = self._validate_identity(username, email, full_name)
        if not valid:
            return False, message, None
        valid, message = self.validate_password(password)
        if not valid:
            return False, message, None
        try:
            resolved_role = role if isinstance(role, Role) else Role(str(role).upper())
        except ValueError:
            return False, "Invalid role.", None

        if not bootstrap:
            if actor is None or not actor.has_permission(Permission.USER_MANAGE):
                self.audit(
                    "USER_CREATE_DENIED",
                    actor=actor,
                    target_username=username,
                    outcome="DENIED",
                    details={"requested_role": resolved_role.value},
                )
                return False, "Administrator permission is required.", None
        elif self.user_count() > 0:
            return False, "Bootstrap is allowed only when no users exist.", None

        now = _iso()
        password_hash = PasswordHasher.hash_password(password)
        try:
            with self._lock, self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO users (
                        username, email, full_name, role, password_hash,
                        is_active, failed_attempts, locked_until,
                        must_change_password, password_changed_at,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 1, 0, NULL, ?, ?, ?, ?)
                    """,
                    (
                        username,
                        email,
                        full_name.strip(),
                        resolved_role.value,
                        password_hash,
                        1 if must_change_password else 0,
                        now,
                        now,
                        now,
                    ),
                )
                user_id = int(cursor.lastrowid)
                row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        except sqlite3.IntegrityError:
            return False, "Username or email already exists.", None

        principal = self._row_to_principal(row)
        self.audit(
            "USER_CREATED",
            actor=actor,
            target=principal,
            outcome="SUCCESS",
            details={"role": resolved_role.value, "bootstrap": bootstrap},
        )
        logger.info(
            "Security user created",
            extra={
                "event_code": "SECURITY_USER_CREATED",
                "user_id": actor.user_id if actor else None,
                "status": "SUCCESS",
                "details": {"target_username": username, "role": resolved_role.value},
            },
        )
        return True, "User created successfully.", principal

    def bootstrap_admin(
        self,
        *,
        full_name: str,
        username: str,
        email: str,
        password: str,
        must_change_password: bool = False,
    ) -> Tuple[bool, str, Optional[UserPrincipal]]:
        return self.create_user(
            full_name=full_name,
            username=username,
            email=email,
            password=password,
            role=Role.ADMIN,
            actor=None,
            must_change_password=must_change_password,
            bootstrap=True,
        )

    def authenticate(self, identifier: str, password: str) -> AuthResult:
        identifier = identifier.strip()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ? COLLATE NOCASE OR email = ? COLLATE NOCASE",
                (identifier, identifier),
            ).fetchone()

            if row is None:
                self.audit(
                    "LOGIN_FAILED",
                    actor_username=identifier,
                    outcome="FAILED",
                    details={"reason": "ACCOUNT_NOT_FOUND"},
                )
                logger.warning(
                    "Login failed for unknown account",
                    extra={"event_code": "AUTH_LOGIN_FAILED", "error_code": "AUTH-001", "status": "FAILED"},
                )
                return AuthResult(False, "Invalid username/email or password.", error_code="AUTH-001")

            locked_until = _parse_iso(row["locked_until"])
            now = _utc_now()
            if locked_until and locked_until > now:
                self.audit(
                    "LOGIN_BLOCKED_LOCKOUT",
                    actor_user_id=int(row["id"]),
                    actor_username=str(row["username"]),
                    outcome="DENIED",
                    details={"locked_until": locked_until.isoformat()},
                )
                return AuthResult(
                    False,
                    f"Account is locked until {locked_until.astimezone().strftime('%Y-%m-%d %H:%M:%S')}",
                    error_code="AUTH-LOCKED",
                    locked_until=locked_until,
                )

            if not bool(row["is_active"]):
                self.audit(
                    "LOGIN_BLOCKED_DISABLED",
                    actor_user_id=int(row["id"]),
                    actor_username=str(row["username"]),
                    outcome="DENIED",
                )
                return AuthResult(False, "Account is disabled. Contact an administrator.", error_code="AUTH-DISABLED")

            if not PasswordHasher.verify_password(password, str(row["password_hash"])):
                attempts = int(row["failed_attempts"]) + 1
                new_locked_until: Optional[str] = None
                if attempts >= self.config.max_failed_attempts:
                    new_locked_until = _iso(now + timedelta(minutes=self.config.lockout_minutes))
                    attempts = 0
                conn.execute(
                    "UPDATE users SET failed_attempts = ?, locked_until = ?, updated_at = ? WHERE id = ?",
                    (attempts, new_locked_until, _iso(), int(row["id"])),
                )
                self.audit(
                    "LOGIN_FAILED",
                    actor_user_id=int(row["id"]),
                    actor_username=str(row["username"]),
                    outcome="FAILED",
                    details={"reason": "BAD_PASSWORD", "lockout_started": bool(new_locked_until)},
                )
                return AuthResult(
                    False,
                    "Invalid username/email or password."
                    if not new_locked_until
                    else f"Too many failed attempts. Account locked for {self.config.lockout_minutes} minutes.",
                    error_code="AUTH-002" if not new_locked_until else "AUTH-LOCKED",
                    locked_until=_parse_iso(new_locked_until),
                )

            conn.execute(
                """
                UPDATE users
                SET failed_attempts = 0, locked_until = NULL,
                    last_login_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (_iso(), _iso(), int(row["id"])),
            )
            refreshed = conn.execute("SELECT * FROM users WHERE id = ?", (int(row["id"]),)).fetchone()

        principal = self._row_to_principal(refreshed)
        self.audit("LOGIN_SUCCESS", actor=principal, outcome="SUCCESS")
        logger.info(
            "User authenticated",
            extra={
                "event_code": "AUTH_LOGIN_SUCCESS",
                "user_id": principal.user_id,
                "status": "SUCCESS",
                "details": {"username": principal.username, "role": principal.role.value},
            },
        )
        return AuthResult(True, "Login successful.", user=principal)

    def create_session(self, user: UserPrincipal) -> SessionContext:
        session = SessionContext.create(user, self.config.session_timeout_minutes)
        self.audit(
            "SESSION_CREATED",
            actor=user,
            outcome="SUCCESS",
            details={"session_id": session.session_id, "timeout_minutes": self.config.session_timeout_minutes},
        )
        return session

    def close_session(self, session: SessionContext, reason: str = "LOGOUT") -> None:
        if session.closed:
            return
        session.close()
        self.audit(
            "SESSION_CLOSED",
            actor=session.user,
            outcome="SUCCESS",
            details={"session_id": session.session_id, "reason": reason},
        )
        logger.info(
            "User session closed",
            extra={
                "event_code": "AUTH_SESSION_CLOSED",
                "user_id": session.user.user_id,
                "status": reason,
                "details": {"session_id": session.session_id},
            },
        )

    def has_permission(self, user: UserPrincipal, permission: str | Permission) -> bool:
        return user.has_permission(permission)

    def record_permission_denied(
        self,
        user: UserPrincipal,
        permission: str | Permission,
        action: str,
    ) -> None:
        value = permission.value if isinstance(permission, Permission) else str(permission)
        self.audit(
            "PERMISSION_DENIED",
            actor=user,
            outcome="DENIED",
            details={"permission": value, "action": action},
        )
        logger.warning(
            "Permission denied",
            extra={
                "event_code": "AUTH_PERMISSION_DENIED",
                "error_code": "AUTH-403",
                "user_id": user.user_id,
                "status": "DENIED",
                "details": {"permission": value, "action": action, "role": user.role.value},
            },
        )

    def list_users(self) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, username, email, full_name, role, is_active,
                       failed_attempts, locked_until, must_change_password,
                       last_login_at, created_at, updated_at
                FROM users ORDER BY username COLLATE NOCASE
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_user(self, user_id: int) -> Optional[UserPrincipal]:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (int(user_id),)).fetchone()
        return self._row_to_principal(row) if row else None

    def set_user_active(
        self,
        actor: UserPrincipal,
        user_id: int,
        is_active: bool,
    ) -> Tuple[bool, str]:
        """Enable or disable one account with administrator safety checks."""
        if not actor.has_permission(Permission.USER_MANAGE):
            return False, "Administrator permission is required."
        if actor.user_id == int(user_id) and not is_active:
            return False, "You cannot disable your own active account."

        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (int(user_id),)).fetchone()
            if row is None:
                return False, "User not found."

            # Never allow the final active Administrator to be disabled. This
            # backend rule protects the system even if a different UI or script
            # calls SecurityService directly.
            if (
                not is_active
                and bool(row["is_active"])
                and str(row["role"]).upper() == Role.ADMIN.value
            ):
                active_admin_count = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM users WHERE role = ? AND is_active = 1",
                        (Role.ADMIN.value,),
                    ).fetchone()[0]
                )
                if active_admin_count <= 1:
                    return False, (
                        "The final active Administrator cannot be disabled. "
                        "Create or activate another Administrator first."
                    )

            conn.execute(
                "UPDATE users SET is_active = ?, failed_attempts = 0, "
                "locked_until = NULL, updated_at = ? WHERE id = ?",
                (1 if is_active else 0, _iso(), int(user_id)),
            )

        target = self._row_to_principal(row)
        self.audit(
            "USER_STATUS_CHANGED",
            actor=actor,
            target=target,
            outcome="SUCCESS",
            details={"is_active": is_active},
        )
        logger.info(
            "Security user status changed",
            extra={
                "event_code": "SECURITY_USER_STATUS_CHANGED",
                "user_id": actor.user_id,
                "status": "ENABLED" if is_active else "DISABLED",
                "details": {
                    "actor_username": actor.username,
                    "target_user_id": int(user_id),
                    "target_username": target.username,
                    "is_active": bool(is_active),
                },
            },
        )
        return True, "User enabled successfully." if is_active else "User disabled successfully."

    def set_user_role(
        self,
        actor: UserPrincipal,
        user_id: int,
        role: Role | str,
    ) -> Tuple[bool, str]:
        """Change a user's role while preserving at least one active Admin."""
        if not actor.has_permission(Permission.USER_MANAGE):
            return False, "Administrator permission is required."
        try:
            resolved = role if isinstance(role, Role) else Role(str(role).upper())
        except ValueError:
            return False, "Invalid role."
        if actor.user_id == int(user_id) and resolved != Role.ADMIN:
            return False, "You cannot remove your own administrator role."

        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (int(user_id),)).fetchone()
            if row is None:
                return False, "User not found."

            old_role = Role(str(row["role"]).upper())
            if old_role == resolved:
                return True, "The selected role is already assigned."

            # Demoting an active Admin is forbidden when it would leave the
            # application without an active Administrator account.
            if (
                old_role == Role.ADMIN
                and resolved != Role.ADMIN
                and bool(row["is_active"])
            ):
                active_admin_count = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM users WHERE role = ? AND is_active = 1",
                        (Role.ADMIN.value,),
                    ).fetchone()[0]
                )
                if active_admin_count <= 1:
                    return False, (
                        "The final active Administrator cannot be demoted. "
                        "Create or activate another Administrator first."
                    )

            conn.execute(
                "UPDATE users SET role = ?, updated_at = ? WHERE id = ?",
                (resolved.value, _iso(), int(user_id)),
            )

        target = self._row_to_principal(row)
        self.audit(
            "USER_ROLE_CHANGED",
            actor=actor,
            target=target,
            outcome="SUCCESS",
            details={"old_role": target.role.value, "new_role": resolved.value},
        )
        logger.info(
            "Security user role changed",
            extra={
                "event_code": "SECURITY_USER_ROLE_CHANGED",
                "user_id": actor.user_id,
                "status": "SUCCESS",
                "details": {
                    "actor_username": actor.username,
                    "target_user_id": int(user_id),
                    "target_username": target.username,
                    "old_role": target.role.value,
                    "new_role": resolved.value,
                },
            },
        )
        return True, "User role updated successfully."

    def admin_reset_password(
        self,
        actor: UserPrincipal,
        user_id: int,
        new_password: str,
        *,
        require_change: bool = True,
    ) -> Tuple[bool, str]:
        if not actor.has_permission(Permission.USER_MANAGE):
            return False, "Administrator permission is required."
        valid, message = self.validate_password(new_password)
        if not valid:
            return False, message
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (int(user_id),)).fetchone()
            if row is None:
                return False, "User not found."
            conn.execute(
                """
                UPDATE users
                SET password_hash = ?, must_change_password = ?,
                    failed_attempts = 0, locked_until = NULL,
                    password_changed_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    PasswordHasher.hash_password(new_password),
                    1 if require_change else 0,
                    _iso(),
                    _iso(),
                    int(user_id),
                ),
            )
        target = self._row_to_principal(row)
        self.audit(
            "USER_PASSWORD_RESET",
            actor=actor,
            target=target,
            outcome="SUCCESS",
            details={"require_change": bool(require_change)},
        )
        logger.info(
            "Security user password reset",
            extra={
                "event_code": "SECURITY_USER_PASSWORD_RESET",
                "user_id": actor.user_id,
                "status": "SUCCESS",
                "details": {
                    "actor_username": actor.username,
                    "target_user_id": int(user_id),
                    "target_username": target.username,
                    "must_change_password": bool(require_change),
                },
            },
        )
        return True, "Password reset successfully."

    def change_password(
        self,
        user: UserPrincipal,
        current_password: str,
        new_password: str,
    ) -> Tuple[bool, str]:
        valid, message = self.validate_password(new_password)
        if not valid:
            return False, message
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user.user_id,)).fetchone()
            if row is None or not PasswordHasher.verify_password(current_password, str(row["password_hash"])):
                return False, "Current password is incorrect."
            conn.execute(
                """
                UPDATE users
                SET password_hash = ?, must_change_password = 0,
                    password_changed_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (PasswordHasher.hash_password(new_password), _iso(), _iso(), user.user_id),
            )
        self.audit("PASSWORD_CHANGED", actor=user, outcome="SUCCESS")
        return True, "Password changed successfully."

    def audit(
        self,
        event_code: str,
        *,
        actor: Optional[UserPrincipal] = None,
        actor_user_id: Optional[int] = None,
        actor_username: Optional[str] = None,
        target: Optional[UserPrincipal] = None,
        target_user_id: Optional[int] = None,
        target_username: Optional[str] = None,
        outcome: str,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO security_audit (
                    occurred_at, actor_user_id, actor_username, event_code,
                    target_user_id, target_username, outcome, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _iso(),
                    actor.user_id if actor else actor_user_id,
                    actor.username if actor else actor_username,
                    event_code,
                    target.user_id if target else target_user_id,
                    target.username if target else target_username,
                    outcome,
                    json.dumps(dict(details or {}), default=str, sort_keys=True),
                ),
            )

    def list_audit_events(self, limit: int = 200) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM security_audit ORDER BY id DESC LIMIT ?",
                (max(1, min(int(limit), 5000)),),
            ).fetchall()
        events: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["details"] = json.loads(item.pop("details_json"))
            except (ValueError, TypeError):
                item["details"] = {}
            events.append(item)
        return events


_SECURITY_SERVICE: Optional[SecurityService] = None
_SECURITY_LOCK = threading.Lock()


def get_security_service(*, force_reload: bool = False) -> SecurityService:
    global _SECURITY_SERVICE
    with _SECURITY_LOCK:
        if force_reload or _SECURITY_SERVICE is None:
            _SECURITY_SERVICE = SecurityService()
        return _SECURITY_SERVICE


def reset_security_service_for_tests() -> None:
    global _SECURITY_SERVICE
    with _SECURITY_LOCK:
        _SECURITY_SERVICE = None
