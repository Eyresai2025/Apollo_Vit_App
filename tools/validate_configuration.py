"""Validate Apollo VIT configuration without connecting to hardware."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow direct execution: python tools/validate_configuration.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.COMMON.config import ConfigManager, ValidationSeverity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env",
        default=str(PROJECT_ROOT / ".env"),
        help="Path to .env file or project folder",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        help="Optional output path for a masked configuration snapshot",
    )
    parser.add_argument(
        "--show-values",
        action="store_true",
        help="Print typed values with secrets masked",
    )
    args = parser.parse_args()

    manager = ConfigManager(args.env)
    report = manager.validation_report

    print("=" * 78)
    print("APOLLO VIT CONFIGURATION VALIDATION")
    print("=" * 78)
    print(f"Project root : {manager.project_root}")
    print(f"Environment  : {manager.env_path}")
    print(f"Loaded keys  : {len(manager.as_legacy_dict())}")
    print(f"Status       : {report.status}")
    print(f"Errors       : {len(report.errors)}")
    print(f"Warnings     : {len(report.warnings)}")
    print("-" * 78)

    for issue in report.issues:
        marker = {
            ValidationSeverity.ERROR: "[ERROR]",
            ValidationSeverity.WARNING: "[WARN ]",
            ValidationSeverity.INFO: "[INFO ]",
        }[issue.severity]
        key_text = f" ({issue.key})" if issue.key else ""
        print(f"{marker} {issue.code}{key_text}: {issue.message}")

    if args.show_values:
        print("-" * 78)
        print(json.dumps(manager.config.to_dict(mask_secrets=True), indent=2))

    if args.json_path:
        destination = manager.export_snapshot(args.json_path)
        print("-" * 78)
        print(f"Masked snapshot written to: {destination}")

    return 0 if report.is_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
