from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional


def _safe_float(value, default=None):
    try:
        return float(value)
    except Exception:
        return default


def run_validation_for_sku(
    media_path: str,
    sku_name: str,
    training_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    F-019 backend.

    Preferred input:
        validation_metrics.json created by AI/training pipeline.

    Expected JSON structure:
        {
          "confusion_matrix": [[10, 1], [2, 12]],
          "precision": {"good": 0.83, "defect": 0.92},
          "recall": {"good": 0.91, "defect": 0.86},
          "f1": {"good": 0.87, "defect": 0.89},
          "f1_macro": 0.88,
          "accepted": true
        }
    """
    training_summary = training_summary or {}

    media = Path(media_path)
    sku_folder = sku_name.strip()

    report_dir = media / "validation_reports" / sku_folder
    report_dir.mkdir(parents=True, exist_ok=True)

    candidates = []

    if training_summary.get("validation_metrics_path"):
        candidates.append(Path(training_summary["validation_metrics_path"]))

    if training_summary.get("summary_path"):
        summary_path = Path(training_summary["summary_path"])
        candidates.append(summary_path.parent / "validation_metrics.json")

    candidates.append(media / "new_sku_images" / sku_folder / "validation_metrics.json")
    candidates.append(media / "calibration" / sku_folder / "validation_metrics.json")

    metrics = None
    used_path = ""

    for path in candidates:
        if path and path.exists():
            with open(path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            used_path = str(path)
            break

    if metrics is None:
        metrics = {
            "status": "PENDING_VALIDATION_DATA",
            "message": (
                "Validation metrics file not found. "
                "AI validation pipeline must generate validation_metrics.json."
            ),
            "confusion_matrix": [],
            "precision": {},
            "recall": {},
            "f1": {},
            "f1_macro": None,
            "accepted": False,
        }

    f1_macro = _safe_float(metrics.get("f1_macro"), None)

    result = {
        "sku_name": sku_name,
        "status": metrics.get("status", "COMPLETED"),
        "metrics_source": used_path,
        "confusion_matrix": metrics.get("confusion_matrix", []),
        "precision": metrics.get("precision", {}),
        "recall": metrics.get("recall", {}),
        "f1": metrics.get("f1", {}),
        "f1_macro": f1_macro,
        "accepted": bool(metrics.get("accepted", False)),
        "message": metrics.get("message", ""),
        "validated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    out_path = report_dir / "latest_validation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    result["report_path"] = str(out_path)
    return result