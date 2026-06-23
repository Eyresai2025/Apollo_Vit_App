"""Production User & Role Management page for Apollo VIT RBAC.

This page is intentionally UI-only. All authoritative permission checks,
password rules, account-state changes and audit writes remain inside
``src.COMMON.security.SecurityService``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from PyQt5.QtCore import Qt  # type: ignore
from PyQt5.QtGui import QColor  # type: ignore
from PyQt5.QtWidgets import (  # type: ignore
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.COMMON.security import Permission, Role, SecurityService, SessionContext
from src.COMMON.structured_logging import get_logger

logger = get_logger(__name__, component="USER_MANAGEMENT")


PAGE_STYLE = """
QWidget {
    color:#172033;
    font:12px 'Segoe UI';
}
QWidget#userManagementPage,
QDialog {
    background:#F4F7FB;
}
QLabel {
    background:transparent;
    border:none;
}
QFrame#pageCard,
QFrame#summaryCard,
QFrame#selectedCard {
    background:white;
    border:1px solid #DCE3EC;
    border-radius:12px;
}
QLabel#title {
    font:800 23px 'Segoe UI';
    color:#172033;
}
QLabel#subtitle {
    color:#64748B;
    font:12px 'Segoe UI';
}
QLabel#summaryTitle {
    color:#64748B;
    font:700 11px 'Segoe UI';
}
QLabel#summaryValue {
    color:#571C86;
    font:800 24px 'Segoe UI';
}
QLabel#selectedTitle {
    color:#172033;
    font:800 13px 'Segoe UI';
}
QLabel#selectedText {
    color:#475569;
    font:12px 'Segoe UI';
}
QPushButton {
    background:#571C86;
    color:white;
    border:none;
    border-radius:7px;
    padding:8px 14px;
    font:600 12px 'Segoe UI';
}
QPushButton:hover { background:#6D28A4; }
QPushButton:pressed { background:#46166B; }
QPushButton:disabled {
    background:#CBD5E1;
    color:#64748B;
}
QPushButton#secondaryButton {
    background:white;
    color:#571C86;
    border:1px solid #571C86;
}
QPushButton#secondaryButton:hover { background:#F4EAFB; }
QPushButton#dangerButton { background:#B42318; }
QPushButton#dangerButton:hover { background:#912018; }
QPushButton#successButton { background:#198754; }
QPushButton#successButton:hover { background:#157347; }
QTableWidget {
    background:white;
    alternate-background-color:#F8FAFC;
    border:1px solid #DCE3EC;
    border-radius:8px;
    gridline-color:#E6EBF1;
    selection-background-color:#E9D8F7;
    selection-color:#172033;
}
QTableWidget::item { padding:5px; }
QHeaderView::section {
    background:#EEF2F7;
    color:#334155;
    border:none;
    border-bottom:1px solid #DCE3EC;
    padding:8px;
    font:700 11px 'Segoe UI';
}
QLineEdit, QComboBox {
    background:white;
    color:#172033;
    border:1px solid #CBD5E1;
    border-radius:7px;
    padding:7px 9px;
    min-height:24px;
}
QLineEdit:focus, QComboBox:focus { border:1px solid #571C86; }
"""


ROLE_DISPLAY = {
    Role.ADMIN.value: "Administrator",
    Role.OPERATOR.value: "Operator",
    Role.QUALITY_ENGINEER.value: "Quality Engineer",
    Role.MAINTENANCE.value: "Maintenance",
    Role.AI_ENGINEER.value: "AI Engineer",
}


def _role_display(value: object) -> str:
    raw = str(value or "").upper()
    return ROLE_DISPLAY.get(raw, raw.replace("_", " ").title() or "Unknown")


def _parse_datetime(value: object) -> Optional[datetime]:
    if value in (None, "", "-"):
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except (TypeError, ValueError):
        return None


def _format_datetime(value: object, empty_text: str = "Never") -> str:
    parsed = _parse_datetime(value)
    if parsed is None:
        return empty_text
    try:
        parsed = parsed.astimezone()
    except (ValueError, OSError):
        pass
    return parsed.strftime("%d/%m/%Y %I:%M %p")


def _is_currently_locked(user: Dict[str, object]) -> bool:
    locked_until = _parse_datetime(user.get("locked_until"))
    if locked_until is None:
        return False
    now = datetime.now(timezone.utc)
    if locked_until.tzinfo is None:
        locked_until = locked_until.replace(tzinfo=timezone.utc)
    return locked_until.astimezone(timezone.utc) > now


def _status_text(user: Dict[str, object]) -> str:
    if not bool(user.get("is_active")):
        return "Disabled"
    if _is_currently_locked(user):
        return "Locked"
    if bool(user.get("must_change_password")):
        return "Temporary Password"
    return "Active"


class CreateUserDialog(QDialog):
    def __init__(self, service: SecurityService, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("userManagementPage")
        self.service = service
        self.setWindowTitle("Create Apollo User")
        self.setMinimumWidth(500)
        self.setStyleSheet(PAGE_STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(14)

        title = QLabel("Create New User")
        title.setObjectName("selectedTitle")
        layout.addWidget(title)

        form = QFormLayout()
        form.setSpacing(12)
        self.full_name = QLineEdit()
        self.full_name.setPlaceholderText("Employee full name")
        self.username = QLineEdit()
        self.username.setPlaceholderText("Unique login username")
        self.email = QLineEdit()
        self.email.setPlaceholderText("name@company.com")
        self.role = QComboBox()
        for role in Role:
            self.role.addItem(_role_display(role.value), role.value)
        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.Password)
        self.password.setPlaceholderText("Temporary password")
        self.confirm = QLineEdit()
        self.confirm.setEchoMode(QLineEdit.Password)
        self.confirm.setPlaceholderText("Repeat temporary password")

        form.addRow("Full name", self.full_name)
        form.addRow("Username", self.username)
        form.addRow("Email", self.email)
        form.addRow("Role", self.role)
        form.addRow("Temporary password", self.password)
        form.addRow("Confirm password", self.confirm)
        layout.addLayout(form)

        note = QLabel(
            "The account is created as active. The user must change the temporary "
            "password during the first successful login."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#64748B; background:#F8FAFC; padding:9px; border-radius:7px;")
        layout.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _validate(self) -> None:
        if not self.full_name.text().strip():
            QMessageBox.warning(self, "Create user", "Full name is required.")
            self.full_name.setFocus()
            return
        if not self.username.text().strip():
            QMessageBox.warning(self, "Create user", "Username is required.")
            self.username.setFocus()
            return
        if not self.email.text().strip():
            QMessageBox.warning(self, "Create user", "Email is required.")
            self.email.setFocus()
            return
        if self.password.text() != self.confirm.text():
            QMessageBox.warning(self, "Create user", "The passwords do not match.")
            self.confirm.setFocus()
            return
        valid, message = self.service.validate_password(self.password.text())
        if not valid:
            QMessageBox.warning(self, "Create user", message)
            self.password.setFocus()
            return
        self.accept()

    def values(self) -> Dict[str, object]:
        return {
            "full_name": self.full_name.text().strip(),
            "username": self.username.text().strip(),
            "email": self.email.text().strip(),
            "role": self.role.currentData(),
            "password": self.password.text(),
        }


class ResetPasswordDialog(QDialog):
    def __init__(self, service: SecurityService, username: str, parent=None) -> None:
        super().__init__(parent)
        self.service = service
        self.setWindowTitle(f"Reset Password • {username}")
        self.setMinimumWidth(460)
        self.setStyleSheet(PAGE_STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(14)

        form = QFormLayout()
        form.setSpacing(12)
        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.Password)
        self.password.setPlaceholderText("New temporary password")
        self.confirm = QLineEdit()
        self.confirm.setEchoMode(QLineEdit.Password)
        self.confirm.setPlaceholderText("Repeat temporary password")
        form.addRow("Temporary password", self.password)
        form.addRow("Confirm password", self.confirm)
        layout.addLayout(form)

        note = QLabel("The user must change this temporary password at the next login.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#64748B; background:#F8FAFC; padding:9px; border-radius:7px;")
        layout.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _validate(self) -> None:
        if self.password.text() != self.confirm.text():
            QMessageBox.warning(self, "Reset password", "The passwords do not match.")
            return
        valid, message = self.service.validate_password(self.password.text())
        if not valid:
            QMessageBox.warning(self, "Reset password", message)
            return
        self.accept()


class UserManagementPage(QWidget):
    """Administrator-only user management page."""

    def __init__(
        self,
        service: SecurityService,
        session: SessionContext,
        on_close=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.service = service
        self.session = session
        self.on_close = on_close
        self._all_users: List[Dict[str, object]] = []
        self._users_by_id: Dict[int, Dict[str, object]] = {}
        self._displayed_user_ids: List[int] = []
        self.setStyleSheet(PAGE_STYLE)
        self._build_ui()
        self.refresh_users()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)

        top = QHBoxLayout()
        heading_box = QVBoxLayout()
        title = QLabel("User & Role Management")
        title.setObjectName("title")
        heading_box.addWidget(title)
        subtitle = QLabel(
            f"Signed in as {self.session.user.full_name} • Administrator. "
            "Create accounts, assign roles and control access."
        )
        subtitle.setObjectName("subtitle")
        heading_box.addWidget(subtitle)
        top.addLayout(heading_box)
        top.addStretch()

        self.create_button = QPushButton("+ Create User")
        self.create_button.clicked.connect(self.create_user)
        top.addWidget(self.create_button)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setObjectName("secondaryButton")
        self.refresh_button.clicked.connect(self.refresh_users)
        top.addWidget(self.refresh_button)
        root.addLayout(top)

        # Summary cards
        summary_row = QHBoxLayout()
        summary_row.setSpacing(10)
        self.summary_labels: Dict[str, QLabel] = {}
        for key, caption in (
            ("total", "TOTAL USERS"),
            ("active", "ACTIVE"),
            ("disabled", "DISABLED"),
            ("locked", "LOCKED"),
            ("admins", "ACTIVE ADMINS"),
        ):
            card = QFrame()
            card.setObjectName("summaryCard")
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            card.setMinimumHeight(72)
            layout = QVBoxLayout(card)
            layout.setContentsMargins(12, 9, 12, 9)
            label = QLabel(caption)
            label.setObjectName("summaryTitle")
            value = QLabel("0")
            value.setObjectName("summaryValue")
            layout.addWidget(label)
            layout.addWidget(value)
            self.summary_labels[key] = value
            summary_row.addWidget(card)
        root.addLayout(summary_row)

        card = QFrame()
        card.setObjectName("pageCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 12, 12, 12)
        card_layout.setSpacing(10)

        # Search and filters
        filter_row = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search username, full name, email, role or status...")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.textChanged.connect(self.apply_filters)
        filter_row.addWidget(self.search_edit, 1)

        self.role_filter = QComboBox()
        self.role_filter.addItem("All roles", "ALL")
        for role in Role:
            self.role_filter.addItem(_role_display(role.value), role.value)
        self.role_filter.currentIndexChanged.connect(self.apply_filters)
        filter_row.addWidget(self.role_filter)

        self.status_filter = QComboBox()
        for caption, value in (
            ("All statuses", "ALL"),
            ("Active", "ACTIVE"),
            ("Disabled", "DISABLED"),
            ("Locked", "LOCKED"),
            ("Temporary password", "TEMPORARY_PASSWORD"),
        ):
            self.status_filter.addItem(caption, value)
        self.status_filter.currentIndexChanged.connect(self.apply_filters)
        filter_row.addWidget(self.status_filter)

        self.result_count_label = QLabel("Showing 0 of 0 users")
        self.result_count_label.setObjectName("subtitle")
        filter_row.addWidget(self.result_count_label)
        card_layout.addLayout(filter_row)

        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(
            [
                "ID",
                "Username",
                "Full name",
                "Email",
                "Role",
                "Status",
                "Last login",
                "Locked until",
                "Created",
            ]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(34)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._update_selected_user_panel)
        card_layout.addWidget(self.table, 1)

        # Selected user information and actions
        selected_card = QFrame()
        selected_card.setObjectName("selectedCard")
        selected_layout = QHBoxLayout(selected_card)
        selected_layout.setContentsMargins(12, 10, 12, 10)

        selected_text_box = QVBoxLayout()
        self.selected_title = QLabel("No user selected")
        self.selected_title.setObjectName("selectedTitle")
        self.selected_details = QLabel("Select a user row to enable account actions.")
        self.selected_details.setObjectName("selectedText")
        selected_text_box.addWidget(self.selected_title)
        selected_text_box.addWidget(self.selected_details)
        selected_layout.addLayout(selected_text_box, 1)

        self.role_button = QPushButton("Change Role")
        self.role_button.clicked.connect(self.change_role)
        selected_layout.addWidget(self.role_button)

        self.reset_button = QPushButton("Reset Password")
        self.reset_button.clicked.connect(self.reset_password)
        selected_layout.addWidget(self.reset_button)

        self.status_button = QPushButton("Disable User")
        self.status_button.setObjectName("dangerButton")
        self.status_button.clicked.connect(self.toggle_status)
        selected_layout.addWidget(self.status_button)

        card_layout.addWidget(selected_card)
        root.addWidget(card, 1)

        self._update_selected_user_panel()

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------
    def _selected_user(self) -> Optional[Dict[str, object]]:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        row = rows[0].row()
        id_item = self.table.item(row, 0)
        if id_item is None:
            return None
        user_id = id_item.data(Qt.UserRole)
        if user_id is None:
            try:
                user_id = int(id_item.text())
            except (TypeError, ValueError):
                return None
        return self._users_by_id.get(int(user_id))

    def _active_admin_count(self) -> int:
        return sum(
            1
            for user in self._all_users
            if bool(user.get("is_active"))
            and str(user.get("role", "")).upper() == Role.ADMIN.value
        )

    def _is_last_active_admin(self, user: Dict[str, object]) -> bool:
        return (
            bool(user.get("is_active"))
            and str(user.get("role", "")).upper() == Role.ADMIN.value
            and self._active_admin_count() <= 1
        )

    def _matches_filters(self, user: Dict[str, object]) -> bool:
        query = self.search_edit.text().strip().lower()
        role_filter = str(self.role_filter.currentData() or "ALL")
        status_filter = str(self.status_filter.currentData() or "ALL")
        status = _status_text(user)

        searchable = " ".join(
            [
                str(user.get("username", "")),
                str(user.get("full_name", "")),
                str(user.get("email", "")),
                _role_display(user.get("role")),
                status,
            ]
        ).lower()
        if query and query not in searchable:
            return False
        if role_filter != "ALL" and str(user.get("role", "")).upper() != role_filter:
            return False

        if status_filter == "ACTIVE" and status != "Active":
            return False
        if status_filter == "DISABLED" and status != "Disabled":
            return False
        if status_filter == "LOCKED" and status != "Locked":
            return False
        if status_filter == "TEMPORARY_PASSWORD" and status != "Temporary Password":
            return False
        return True

    def _update_summary(self) -> None:
        total = len(self._all_users)
        active = sum(1 for u in self._all_users if bool(u.get("is_active")))
        disabled = total - active
        locked = sum(1 for u in self._all_users if _is_currently_locked(u))
        admins = self._active_admin_count()
        values = {
            "total": total,
            "active": active,
            "disabled": disabled,
            "locked": locked,
            "admins": admins,
        }
        for key, value in values.items():
            self.summary_labels[key].setText(str(value))

    def _make_item(self, text: object, *, user_id: int, center: bool = False) -> QTableWidgetItem:
        item = QTableWidgetItem(str(text))
        item.setData(Qt.UserRole, int(user_id))
        if center:
            item.setTextAlignment(Qt.AlignCenter)
        return item

    # ------------------------------------------------------------------
    # Refresh and filtering
    # ------------------------------------------------------------------
    def refresh_users(self, _checked=False) -> None:
        if not self.session.user.has_permission(Permission.USER_MANAGE):
            QMessageBox.critical(self, "Access denied", "Administrator permission is required.")
            return
        try:
            users = self.service.list_users()
            self._all_users = [dict(user) for user in users]
            self._users_by_id = {int(user["id"]): user for user in self._all_users}
            self._update_summary()
            self.apply_filters()
            logger.info(
                "User management list refreshed",
                extra={
                    "event_code": "USER_MANAGEMENT_REFRESHED",
                    "user_id": self.session.user.user_id,
                    "status": "SUCCESS",
                    "details": {"user_count": len(self._all_users)},
                },
            )
        except Exception as exc:
            logger.exception(
                "Failed to load user list",
                extra={
                    "event_code": "USER_MANAGEMENT_REFRESH_FAILED",
                    "error_code": "AUTH-UI-201",
                    "user_id": self.session.user.user_id,
                },
            )
            QMessageBox.critical(self, "User Management", f"Failed to load users:\n\n{exc}")

    def apply_filters(self, *_args) -> None:
        displayed = [user for user in self._all_users if self._matches_filters(user)]
        self._displayed_user_ids = [int(user["id"]) for user in displayed]

        self.table.setSortingEnabled(False)
        self.table.clearSelection()
        self.table.setRowCount(len(displayed))

        for row_index, user in enumerate(displayed):
            user_id = int(user["id"])
            status = _status_text(user)
            values = [
                user_id,
                user.get("username", ""),
                user.get("full_name", ""),
                user.get("email", ""),
                _role_display(user.get("role")),
                status,
                _format_datetime(user.get("last_login_at"), "Never"),
                _format_datetime(user.get("locked_until"), "Not locked"),
                _format_datetime(user.get("created_at"), "Unknown"),
            ]
            for col, value in enumerate(values):
                item = self._make_item(value, user_id=user_id, center=col in (0, 4, 5))
                if col == 5:
                    if status == "Active":
                        item.setForeground(QColor("#157347"))
                    elif status == "Disabled":
                        item.setForeground(QColor("#B42318"))
                    elif status == "Locked":
                        item.setForeground(QColor("#B54708"))
                    else:
                        item.setForeground(QColor("#571C86"))
                self.table.setItem(row_index, col, item)

        self.table.setSortingEnabled(True)
        self.result_count_label.setText(
            f"Showing {len(displayed)} of {len(self._all_users)} users"
        )
        self._update_selected_user_panel()

    # ------------------------------------------------------------------
    # Selection/action state
    # ------------------------------------------------------------------
    def _update_selected_user_panel(self) -> None:
        selected = self._selected_user()
        enabled = selected is not None
        self.role_button.setEnabled(enabled)
        self.reset_button.setEnabled(enabled)
        self.status_button.setEnabled(enabled)

        if selected is None:
            self.selected_title.setText("No user selected")
            self.selected_details.setText("Select a user row to enable account actions.")
            self.status_button.setText("Disable User")
            self.status_button.setObjectName("dangerButton")
            self.status_button.style().unpolish(self.status_button)
            self.status_button.style().polish(self.status_button)
            return

        username = str(selected.get("username", ""))
        full_name = str(selected.get("full_name", ""))
        role_text = _role_display(selected.get("role"))
        status = _status_text(selected)
        self.selected_title.setText(f"Selected: {full_name} ({username})")
        self.selected_details.setText(f"{role_text} • {status} • {selected.get('email', '')}")

        is_active = bool(selected.get("is_active"))
        self.status_button.setText("Disable User" if is_active else "Enable User")
        self.status_button.setObjectName("dangerButton" if is_active else "successButton")
        self.status_button.style().unpolish(self.status_button)
        self.status_button.style().polish(self.status_button)

        # Immediate UI protection. SecurityService repeats these checks so the
        # rules cannot be bypassed by calling the backend directly.
        is_self = int(selected["id"]) == int(self.session.user.user_id)
        last_active_admin = self._is_last_active_admin(selected)
        if is_self and is_active:
            self.status_button.setEnabled(False)
            self.status_button.setToolTip("You cannot disable your own active account.")
        elif last_active_admin and is_active:
            self.status_button.setEnabled(False)
            self.status_button.setToolTip("The final active Administrator cannot be disabled.")
        else:
            self.status_button.setEnabled(True)
            self.status_button.setToolTip("")

        if is_self and str(selected.get("role", "")).upper() == Role.ADMIN.value:
            self.role_button.setToolTip("Your own Administrator role cannot be removed.")
        else:
            self.role_button.setToolTip("")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def create_user(self, _checked=False) -> None:
        dialog = CreateUserDialog(self.service, self)
        if dialog.exec_() != QDialog.Accepted:
            return
        ok, message, principal = self.service.create_user(
            actor=self.session.user,
            must_change_password=True,
            **dialog.values(),
        )
        if not ok:
            QMessageBox.warning(self, "Create user", message)
            return
        logger.info(
            "User created from User Management page",
            extra={
                "event_code": "USER_MANAGEMENT_USER_CREATED",
                "user_id": self.session.user.user_id,
                "status": "SUCCESS",
                "details": {
                    "actor": self.session.user.username,
                    "target": principal.username if principal else dialog.values()["username"],
                },
            },
        )
        QMessageBox.information(self, "Create user", message)
        self.refresh_users()

    def change_role(self, _checked=False) -> None:
        user = self._selected_user()
        if not user:
            return

        current_role = str(user.get("role", "")).upper()
        is_self = int(user["id"]) == int(self.session.user.user_id)

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Change Role • {user['username']}")
        dialog.setMinimumWidth(430)
        dialog.setStyleSheet(PAGE_STYLE)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)
        layout.addWidget(QLabel(f"Current role: {_role_display(current_role)}"))
        layout.addWidget(QLabel("Select the new role:"))

        combo = QComboBox()
        for role in Role:
            combo.addItem(_role_display(role.value), role.value)
            if role.value == current_role:
                combo.setCurrentIndex(combo.count() - 1)
        layout.addWidget(combo)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec_() != QDialog.Accepted:
            return
        new_role = str(combo.currentData())
        if new_role == current_role:
            QMessageBox.information(self, "Change role", "The selected role is already assigned.")
            return
        if is_self and new_role != Role.ADMIN.value:
            QMessageBox.warning(self, "Change role", "You cannot remove your own Administrator role.")
            return
        if self._is_last_active_admin(user) and new_role != Role.ADMIN.value:
            QMessageBox.warning(
                self,
                "Change role",
                "This is the final active Administrator. Create or activate another "
                "Administrator before changing this role.",
            )
            return

        response = QMessageBox.question(
            self,
            "Confirm role change",
            f"Change {user['username']} from {_role_display(current_role)} "
            f"to {_role_display(new_role)}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if response != QMessageBox.Yes:
            return

        ok, message = self.service.set_user_role(
            self.session.user,
            int(user["id"]),
            new_role,
        )
        if not ok:
            QMessageBox.warning(self, "Change role", message)
            return
        QMessageBox.information(self, "Change role", message)
        self.refresh_users()

    def reset_password(self, _checked=False) -> None:
        user = self._selected_user()
        if not user:
            return
        dialog = ResetPasswordDialog(self.service, str(user["username"]), self)
        if dialog.exec_() != QDialog.Accepted:
            return

        response = QMessageBox.question(
            self,
            "Confirm password reset",
            f"Reset the password for {user['username']}?\n\n"
            "The current password will stop working and the user must change "
            "the new temporary password at the next login.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if response != QMessageBox.Yes:
            return

        ok, message = self.service.admin_reset_password(
            self.session.user,
            int(user["id"]),
            dialog.password.text(),
            require_change=True,
        )
        if not ok:
            QMessageBox.warning(self, "Reset password", message)
            return
        QMessageBox.information(self, "Reset password", message)
        self.refresh_users()

    def toggle_status(self, _checked=False) -> None:
        user = self._selected_user()
        if not user:
            return

        new_status = not bool(user.get("is_active"))
        is_self = int(user["id"]) == int(self.session.user.user_id)
        if is_self and not new_status:
            QMessageBox.warning(self, "User status", "You cannot disable your own active account.")
            return
        if not new_status and self._is_last_active_admin(user):
            QMessageBox.warning(
                self,
                "User status",
                "This is the final active Administrator. Create or activate another "
                "Administrator before disabling this account.",
            )
            return

        action = "enable" if new_status else "disable"
        consequence = (
            "The user will be able to sign in again. Any current lockout will be cleared."
            if new_status
            else "The user will no longer be able to sign in."
        )
        response = QMessageBox.question(
            self,
            "Confirm user status",
            f"Are you sure you want to {action} {user['username']}?\n\n{consequence}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if response != QMessageBox.Yes:
            return

        ok, message = self.service.set_user_active(
            self.session.user,
            int(user["id"]),
            new_status,
        )
        if not ok:
            QMessageBox.warning(self, "User status", message)
            return
        QMessageBox.information(self, "User status", message)
        self.refresh_users()
