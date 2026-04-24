# login_window.py
import os
import sys
from pathlib import Path

from PyQt5.QtWidgets import (  # type: ignore
    QApplication, QDialog, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit,
    QPushButton, QTabWidget, QFrame, QSizePolicy, QMessageBox,
    QFormLayout, QSpacerItem, QDialogButtonBox
)
from PyQt5.QtGui import QIcon, QPixmap, QMovie  # type: ignore
from PyQt5.QtCore import Qt, QUrl, QTimer  # type: ignore

# ---- Optional multimedia (for MP4) ----
try:
    from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent, QMediaPlaylist  # type: ignore
    from PyQt5.QtMultimediaWidgets import QVideoWidget  # type: ignore
    MULTIMEDIA_AVAILABLE = True
except ImportError:
    MULTIMEDIA_AVAILABLE = False

from src.COMMON.db import create_account, authenticate_user, reset_password


def _app_base_dir() -> Path:
    """
    Locate the base directory for bundled resources.
    - PyInstaller onefile: sys._MEIPASS
    - PyInstaller onedir: folder containing exe
    - Normal run: folder containing this file (login_window.py)
    """
    if getattr(sys, "frozen", False):
        if hasattr(sys, "_MEIPASS"):
            return Path(sys._MEIPASS)
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent.parent


class ClickableLabel(QLabel):
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.linkActivated.emit("clicked")
        super().mousePressEvent(event)


class AnimatedLabel(QLabel):
    """Label that auto-scales a QMovie while keeping aspect ratio."""
    def __init__(self, movie: QMovie, parent=None):
        super().__init__(parent)
        self.movie = movie
        self.setMovie(self.movie)
        self.setAlignment(Qt.AlignCenter)

    def resizeEvent(self, event):
        if self.movie:
            frame_size = self.movie.currentPixmap().size()
            if frame_size.isValid():
                scaled = frame_size.scaled(
                    self.size(),
                    Qt.KeepAspectRatio
                )
                self.movie.setScaledSize(scaled)
        super().resizeEvent(event)


class LoginWindow(QDialog):
    def __init__(self, media_path=None, parent=None):
        super().__init__(parent)
        self.logged_in_user = None

        base_dir = _app_base_dir()
        self.media_path = media_path or str(base_dir / "media")

        self.setWindowTitle("Apollo Tyres • Login")
        self.setWindowIcon(QIcon(os.path.join(self.media_path, "img", "Apollo.ico")))

        flags = (
            Qt.Window
            | Qt.WindowTitleHint
            | Qt.WindowSystemMenuHint
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        self.setWindowFlags(flags)

        screen = QApplication.primaryScreen().availableGeometry()
        self.screen_w = screen.width()
        self.screen_h = screen.height()
        self.ui_scale = min(self.screen_w / 1920.0, self.screen_h / 1080.0)

        self.resize(int(self.screen_w * 0.98), int(self.screen_h * 0.94))
        self.setMinimumSize(int(self.screen_w * 0.80), int(self.screen_h * 0.80))

        # Open maximized, but keep the inner login panel centered and controlled
        QTimer.singleShot(0, self.showMaximized)

        self.setStyleSheet("""
            QDialog {
                background-color: #050B1E;
            }
        """)

        # Keep references so GC doesn't kill media
        self.player = None
        self.playlist = None
        self.video_widget = None
        self.movie = None

        self._build_ui()

    def s(self, value):
        return max(1, int(value * self.ui_scale))

    def _build_ui(self):
        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(self.s(8), self.s(8), self.s(8), self.s(8))
        root_layout.setSpacing(0)

        radius = self.s(24)

        # Left panel
        left_frame = QFrame()
        left_frame.setMinimumWidth(self.s(400))
        left_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_frame.setStyleSheet(f"""
            QFrame {{
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #050B1E,
                    stop:0.4 #071633,
                    stop:1 #091E3F);
                border-top-left-radius: {radius}px;
                border-bottom-left-radius: {radius}px;
                border-top-right-radius: 0px;
                border-bottom-right-radius: 0px;
            }}
        """)
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        # Right panel
        right_frame = QFrame()
        right_frame.setMinimumWidth(self.s(400))
        right_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_frame.setStyleSheet(f"""
            QFrame {{
                background-color: #0F172A;
                border-top-right-radius: {radius}px;
                border-bottom-right-radius: {radius}px;
                border-top-left-radius: 0px;
                border-bottom-left-radius: 0px;
            }}
        """)
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(self.s(28), self.s(24), self.s(28), self.s(24))
        right_layout.setSpacing(self.s(14))

        # -------- FULLCARD MEDIA (GIF / MP4 / IMAGE) --------
        gif_path = os.path.join(self.media_path, "img", "login_ai.gif")
        mp4_path = os.path.join(self.media_path, "img", "login_ai.mp4")
        jpg_path = os.path.join(self.media_path, "img", "login_ai.jpg")

        media_container = QWidget()
        media_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        media_layout = QVBoxLayout(media_container)
        media_layout.setContentsMargins(0, 0, 0, 0)
        media_layout.setSpacing(0)

        if os.path.exists(gif_path):
            self.movie = QMovie(gif_path)
            self.movie.setCacheMode(QMovie.CacheAll)
            gif_label = AnimatedLabel(self.movie)
            gif_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            media_layout.addWidget(gif_label)
            self.movie.start()

        elif os.path.exists(mp4_path) and MULTIMEDIA_AVAILABLE:
            self.video_widget = QVideoWidget()
            self.video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            self.playlist = QMediaPlaylist(self)
            self.playlist.addMedia(QMediaContent(QUrl.fromLocalFile(mp4_path)))
            self.playlist.setPlaybackMode(QMediaPlaylist.Loop)

            self.player = QMediaPlayer(self)
            self.player.setVideoOutput(self.video_widget)
            self.player.setPlaylist(self.playlist)
            self.player.play()

            media_layout.addWidget(self.video_widget)

        elif os.path.exists(jpg_path):
            img_label = QLabel()
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            pix = QPixmap(jpg_path)
            img_label.setPixmap(
                pix.scaled(
                    int(self.screen_w * 0.36),
                    int(self.screen_h * 0.62),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )
            media_layout.addWidget(img_label)

        else:
            img_label = QLabel("AI IMAGE\n(put 'login_ai.gif' in media/img)")
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            img_label.setStyleSheet(
                f"color:#6B7280; font: {self.s(12)}px 'Segoe UI';"
            )
            media_layout.addWidget(img_label)

        left_layout.addWidget(media_container)

        # Header
        header = QLabel("Welcome back")
        header.setStyleSheet(
            f"color: #E5E7EB; font: bold {self.s(22)}px 'Segoe UI';"
        )
        right_layout.addWidget(header)

        sub = QLabel("Login or create a new account to access Apollo Smart QC+.")
        sub.setWordWrap(True)
        sub.setStyleSheet(
            f"color: #9CA3AF; font: {self.s(12)}px 'Segoe UI';"
        )
        right_layout.addWidget(sub)

        # Tabs
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.setTabPosition(QTabWidget.North)
        tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: none;
            }}
            QTabBar {{
                qproperty-drawBase: 0;
                border: none;
            }}
            QTabBar::tab {{
                background: transparent;
                color: #9CA3AF;
                padding: {self.s(6)}px {self.s(18)}px;
                font: {self.s(13)}px 'Segoe UI';
                min-width: {self.s(80)}px;
                min-height: {self.s(28)}px;
                margin-top: {self.s(6)}px;
                border: none;
            }}
            QTabBar::tab:selected {{
                color: #38BDF8;
                border-bottom: 2px solid #38BDF8;
            }}
        """)

        # ----- Login tab -----
        login_tab = QWidget()
        login_layout = QVBoxLayout(login_tab)
        login_layout.setContentsMargins(0, self.s(10), 0, 0)
        login_layout.setSpacing(self.s(10))

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignTop)
        form.setVerticalSpacing(self.s(12))
        form.setHorizontalSpacing(self.s(12))

        self.login_identifier = QLineEdit()
        self.login_identifier.setPlaceholderText("Username or Email")
        self._style_line_edit(self.login_identifier)

        self.login_password = QLineEdit()
        self.login_password.setEchoMode(QLineEdit.Password)
        self.login_password.setPlaceholderText("Password")
        self._style_line_edit(self.login_password)

        login_user_label = QLabel("User / Email")
        login_user_label.setStyleSheet(
            f"color:#E5E7EB; font: {self.s(12)}px 'Segoe UI';"
        )

        login_pwd_label = QLabel("Password")
        login_pwd_label.setStyleSheet(
            f"color:#E5E7EB; font: {self.s(12)}px 'Segoe UI';"
        )

        form.addRow(login_user_label, self.login_identifier)
        form.addRow(login_pwd_label, self.login_password)
        login_layout.addLayout(form)

        forgot = ClickableLabel("<a style='color:#60A5FA; text-decoration:none;'>Forgot password?</a>")
        forgot.setTextInteractionFlags(Qt.TextBrowserInteraction)
        forgot.linkActivated.connect(self._open_forgot_password_dialog)
        forgot.setStyleSheet(f"margin-top: {self.s(4)}px; font: {self.s(12)}px 'Segoe UI';")
        login_layout.addWidget(forgot, alignment=Qt.AlignRight)

        login_layout.addSpacing(self.s(8))

        login_btn = QPushButton("Sign In")
        login_btn.clicked.connect(self._handle_login)
        login_btn.setCursor(Qt.PointingHandCursor)
        login_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        login_btn.setFixedHeight(self.s(48))
        login_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #2563EB;
                color: white;
                font: bold {self.s(14)}px 'Segoe UI';
                border-radius: {self.s(24)}px;
            }}
            QPushButton:hover {{
                background-color: #1D4ED8;
            }}
            QPushButton:pressed {{
                background-color: #1E40AF;
            }}
        """)
        login_layout.addWidget(login_btn)

        login_layout.addItem(QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Expanding))
        tabs.addTab(login_tab, "Login")

        # ----- Sign Up tab -----
        signup_tab = QWidget()
        signup_layout = QVBoxLayout(signup_tab)
        signup_layout.setContentsMargins(0, self.s(10), 0, 0)
        signup_layout.setSpacing(self.s(10))

        s_form = QFormLayout()
        s_form.setLabelAlignment(Qt.AlignLeft)
        s_form.setFormAlignment(Qt.AlignTop)
        s_form.setVerticalSpacing(self.s(12))
        s_form.setHorizontalSpacing(self.s(12))

        self.s_fullname = QLineEdit()
        self.s_fullname.setPlaceholderText("Full Name")
        self._style_line_edit(self.s_fullname)

        self.s_username = QLineEdit()
        self.s_username.setPlaceholderText("Unique username")
        self._style_line_edit(self.s_username)

        self.s_email = QLineEdit()
        self.s_email.setPlaceholderText("Email")
        self._style_line_edit(self.s_email)

        self.s_password = QLineEdit()
        self.s_password.setPlaceholderText("Password (5–10 characters)")
        self.s_password.setEchoMode(QLineEdit.Password)
        self._style_line_edit(self.s_password)

        self.s_confirm = QLineEdit()
        self.s_confirm.setPlaceholderText("Confirm password")
        self.s_confirm.setEchoMode(QLineEdit.Password)
        self._style_line_edit(self.s_confirm)

        s_full_lbl = QLabel("Full Name")
        s_full_lbl.setStyleSheet(f"color:#E5E7EB; font: {self.s(12)}px 'Segoe UI';")

        s_user_lbl = QLabel("Username")
        s_user_lbl.setStyleSheet(f"color:#E5E7EB; font: {self.s(12)}px 'Segoe UI';")

        s_email_lbl = QLabel("Email")
        s_email_lbl.setStyleSheet(f"color:#E5E7EB; font: {self.s(12)}px 'Segoe UI';")

        s_pwd_lbl = QLabel("Password")
        s_pwd_lbl.setStyleSheet(f"color:#E5E7EB; font: {self.s(12)}px 'Segoe UI';")

        s_cpwd_lbl = QLabel("Confirm")
        s_cpwd_lbl.setStyleSheet(f"color:#E5E7EB; font: {self.s(12)}px 'Segoe UI';")

        s_form.addRow(s_full_lbl, self.s_fullname)
        s_form.addRow(s_user_lbl, self.s_username)
        s_form.addRow(s_email_lbl, self.s_email)
        s_form.addRow(s_pwd_lbl, self.s_password)
        s_form.addRow(s_cpwd_lbl, self.s_confirm)

        signup_layout.addLayout(s_form)

        signup_btn = QPushButton("Create Account")
        signup_btn.clicked.connect(self._handle_signup)
        signup_btn.setCursor(Qt.PointingHandCursor)
        signup_btn.setFixedHeight(self.s(48))
        signup_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        signup_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #22C55E;
                color: white;
                font: bold {self.s(14)}px 'Segoe UI';
                border-radius: {self.s(24)}px;
            }}
            QPushButton:hover {{
                background-color: #16A34A;
            }}
            QPushButton:pressed {{
                background-color: #15803D;
            }}
        """)
        signup_layout.addWidget(signup_btn)

        signup_layout.addItem(QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Expanding))
        tabs.addTab(signup_tab, "Sign Up")

        right_layout.addWidget(tabs)
        right_layout.addStretch(1)

        footer = QLabel("© Radome Technologies & Apollo Tyres")
        footer.setStyleSheet(
            f"color:#6B7280; font: {self.s(11)}px 'Segoe UI';"
        )
        right_layout.addWidget(footer, alignment=Qt.AlignRight)

        # Center wrapper: bigger than before, but still controlled
        panel_wrap = QWidget()
        panel_wrap.setMinimumWidth(int(self.screen_w * 0.82))
        panel_wrap.setMaximumWidth(int(self.screen_w * 0.88))
        panel_wrap.setMinimumHeight(int(self.screen_h * 0.74))
        panel_wrap.setMaximumHeight(int(self.screen_h * 0.86))
        panel_wrap.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        panel_layout = QHBoxLayout(panel_wrap)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(0)
        panel_layout.addWidget(left_frame, 11)
        panel_layout.addWidget(right_frame, 9)

        root_layout.addStretch(1)
        root_layout.addWidget(panel_wrap, 0, Qt.AlignCenter)
        root_layout.addStretch(1)

    def _style_line_edit(self, le: QLineEdit):
        le.setMinimumHeight(self.s(44))
        le.setStyleSheet(f"""
            QLineEdit {{
                background-color: #020617;
                border-radius: {self.s(10)}px;
                border: 1px solid #1E293B;
                padding: {self.s(8)}px {self.s(10)}px;
                color: #E5E7EB;
                font: {self.s(13)}px 'Segoe UI';
            }}
            QLineEdit:focus {{
                border: 1px solid #38BDF8;
            }}
        """)

    def _show_message(self, icon, title, text, parent=None):
        if parent is None:
            parent = self

        msg = QMessageBox(parent)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setIcon(icon)
        msg.setStyleSheet(f"""
            QMessageBox {{
                background-color: #020617;
            }}
            QLabel {{
                color: #FFFFFF;
                font: {self.s(13)}px 'Segoe UI';
            }}
            QPushButton {{
                background-color: #2563EB;
                color: #FFFFFF;
                padding: {self.s(4)}px {self.s(12)}px;
                border-radius: {self.s(6)}px;
                min-width: {self.s(90)}px;
                min-height: {self.s(30)}px;
            }}
            QPushButton:hover {{
                background-color: #1D4ED8;
            }}
        """)
        msg.exec_()

    def _check_password_rules(self, pwd: str):
        if len(pwd) < 5:
            return False, "Password must be at least 5 characters long."
        if len(pwd) > 10:
            return False, "Password cannot be more than 10 characters."
        return True, ""

    def _handle_login(self):
        identifier = self.login_identifier.text().strip()
        password = self.login_password.text().strip()

        if not identifier or not password:
            self._show_message(
                QMessageBox.Warning,
                "Missing Fields",
                "Please enter username/email and password."
            )
            return

        ok_pwd, msg_pwd = self._check_password_rules(password)
        if not ok_pwd:
            self._show_message(QMessageBox.Warning, "Password Rule", msg_pwd)
            return

        ok, msg, user = authenticate_user(identifier, password)
        if not ok:
            self._show_message(QMessageBox.Critical, "Login failed", msg)
            return

        self.logged_in_user = user
        self._show_message(QMessageBox.Information, "Success", "Login successful.")
        self.accept()

    def _handle_signup(self):
        full_name = self.s_fullname.text().strip()
        username = self.s_username.text().strip()
        email = self.s_email.text().strip()
        pwd = self.s_password.text().strip()
        cpwd = self.s_confirm.text().strip()

        if not all([full_name, username, email, pwd, cpwd]):
            self._show_message(
                QMessageBox.Warning,
                "Missing fields",
                "Please fill all fields."
            )
            return

        if "@" not in email or email.startswith("@") or email.endswith("@"):
            self._show_message(
                QMessageBox.Warning,
                "Invalid email",
                "Please enter a valid email address (must contain '@')."
            )
            return

        ok_pwd, msg_pwd = self._check_password_rules(pwd)
        if not ok_pwd:
            self._show_message(
                QMessageBox.Warning,
                "Password rule",
                msg_pwd
            )
            return

        if pwd != cpwd:
            self._show_message(
                QMessageBox.Warning,
                "Password mismatch",
                "Passwords do not match."
            )
            return

        ok, msg = create_account(full_name, username, email, pwd)
        if not ok:
            self._show_message(
                QMessageBox.Critical,
                "Sign Up failed",
                msg
            )
            return

        self._show_message(
            QMessageBox.Information,
            "Account created",
            msg
        )

        self.login_identifier.setText(username)
        self.login_password.setText(pwd)

    def _open_forgot_password_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Reset password")
        dialog.setModal(True)
        dialog.resize(self.s(420), self.s(260))
        dialog.setMinimumSize(self.s(360), self.s(240))
        dialog.setStyleSheet("QDialog { background-color: #020617; color: #E5E7EB; }")

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(self.s(20), self.s(20), self.s(20), self.s(20))
        layout.setSpacing(self.s(12))

        info = QLabel("Enter your username or email and your new password.")
        info.setWordWrap(True)
        info.setStyleSheet(f"color:#9CA3AF; font: {self.s(12)}px 'Segoe UI';")
        layout.addWidget(info)

        form = QFormLayout()
        form.setVerticalSpacing(self.s(10))
        form.setHorizontalSpacing(self.s(12))

        identifier_edit = QLineEdit()
        identifier_edit.setPlaceholderText("Username or Email")
        self._style_line_edit(identifier_edit)

        new_pwd = QLineEdit()
        new_pwd.setPlaceholderText("New password (5–10 characters)")
        new_pwd.setEchoMode(QLineEdit.Password)
        self._style_line_edit(new_pwd)

        confirm_pwd = QLineEdit()
        confirm_pwd.setPlaceholderText("Confirm new password")
        confirm_pwd.setEchoMode(QLineEdit.Password)
        self._style_line_edit(confirm_pwd)

        id_lbl = QLabel("User / Email")
        id_lbl.setStyleSheet(f"color:#E5E7EB; font: {self.s(12)}px 'Segoe UI';")

        np_lbl = QLabel("New Password")
        np_lbl.setStyleSheet(f"color:#E5E7EB; font: {self.s(12)}px 'Segoe UI';")

        cp_lbl = QLabel("Confirm")
        cp_lbl.setStyleSheet(f"color:#E5E7EB; font: {self.s(12)}px 'Segoe UI';")

        form.addRow(id_lbl, identifier_edit)
        form.addRow(np_lbl, new_pwd)
        form.addRow(cp_lbl, confirm_pwd)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Update")
        buttons.button(QDialogButtonBox.Cancel).setText("Cancel")
        layout.addWidget(buttons)

        def on_accept():
            identifier = identifier_edit.text().strip()
            p1 = new_pwd.text().strip()
            p2 = confirm_pwd.text().strip()

            if not identifier or not p1 or not p2:
                self._show_message(
                    QMessageBox.Warning,
                    "Missing fields",
                    "Please fill all fields.",
                    parent=dialog,
                )
                return

            ok_pwd, msg_pwd = self._check_password_rules(p1)
            if not ok_pwd:
                self._show_message(QMessageBox.Warning, "Password rule", msg_pwd, parent=dialog)
                return

            if p1 != p2:
                self._show_message(
                    QMessageBox.Warning,
                    "Password mismatch",
                    "Passwords do not match.",
                    parent=dialog
                )
                return

            ok, msg = reset_password(identifier, p1)
            if not ok:
                self._show_message(QMessageBox.Critical, "Error", msg, parent=dialog)
                return

            self._show_message(QMessageBox.Information, "Success", msg, parent=dialog)
            dialog.accept()

        buttons.accepted.connect(on_accept)
        buttons.rejected.connect(dialog.reject)

        dialog.exec_()