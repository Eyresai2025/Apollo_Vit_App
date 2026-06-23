from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGIN = ROOT / "src" / "Pages" / "login_window.py"
GUI = ROOT / "GUI.py"


def has_method_arg(tree: ast.AST, class_name: str, method_name: str, arg_name: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                    return any(arg.arg == arg_name for arg in item.args.args)
    return False


def main() -> int:
    login_text = LOGIN.read_text(encoding="utf-8")
    gui_text = GUI.read_text(encoding="utf-8")
    login_tree = ast.parse(login_text)
    gui_tree = ast.parse(gui_text)

    checks = {
        "ORIGINAL_MEDIA_UI": "login_ai.gif" in login_text and "showMaximized" in login_text,
        "RBAC_SERVICE": "SecurityService" in login_text and "self.service.authenticate" in login_text,
        "TEMP_PASSWORD": "PasswordChangeDialog" in login_text and "must_change_password" in login_text,
        "LOGIN_SERVICE_ARG": has_method_arg(login_tree, "LoginWindow", "__init__", "service"),
        "ADMIN_SIGNUP_POLICY": "self-sign-up is disabled" in login_text,
        "ADMIN_RESET_POLICY": "Password resets are managed by an Administrator" in login_text,
        "USER_PAGE_BOOL_SAFE": has_method_arg(gui_tree, "MainWindow", "open_user_management_page", "_checked"),
        "SIDEBAR_BOOL_SAFE": "lambda _checked=False, callback=slot: callback()" in gui_text,
        "DECORATOR_BOOL_SAFE": "isinstance(args[0], bool)" in gui_text,
    }

    print("=" * 78)
    print("APOLLO RBAC ORIGINAL LOGIN UI HOTFIX VALIDATION")
    print("=" * 78)
    failed = False
    for name, ok in checks.items():
        print(f"{name:<24}: {'OK' if ok else 'FAILED'}")
        failed = failed or not ok
    print("-" * 78)
    print("Status                  :", "PASSED" if not failed else "FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
