from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from src.COMMON.db import (
    get_collection,
    ensure_collection,
    ACTION_CODE_CATALOG_COLLECTION,
    AI_DEFECT_CATALOG_MAP_COLLECTION,
    ACTION_DECISION_RULES_COLLECTION,
    INSPECTION_ACTION_DECISIONS_COLLECTION,
)


# ============================================================
# COLLECTION GETTERS
# ============================================================
def action_catalog_col():
    return get_collection(ACTION_CODE_CATALOG_COLLECTION)


def ai_map_col():
    return get_collection(AI_DEFECT_CATALOG_MAP_COLLECTION)


def rules_col():
    return get_collection(ACTION_DECISION_RULES_COLLECTION)


def inspection_action_col():
    return get_collection(INSPECTION_ACTION_DECISIONS_COLLECTION)


# ============================================================
# INIT / INDEXES
# ============================================================
def ensure_action_catalog_collections() -> None:
    ensure_collection(ACTION_CODE_CATALOG_COLLECTION)
    ensure_collection(AI_DEFECT_CATALOG_MAP_COLLECTION)
    ensure_collection(ACTION_DECISION_RULES_COLLECTION)
    ensure_collection(INSPECTION_ACTION_DECISIONS_COLLECTION)

    action_catalog_col().create_index(
        [
            ("catalog_code", 1),
            ("revision_no", 1),
            ("condition_code", 1),
        ],
        unique=True,
        name="uniq_catalog_revision_condition",
    )

    ai_map_col().create_index(
        [
            ("ai_label", 1),
            ("side", 1),
            ("model_version", 1),
        ],
        name="idx_ai_label_side_model",
    )

    rules_col().create_index(
        [
            ("catalog_code", 1),
            ("revision_no", 1),
            ("priority", -1),
        ],
        name="idx_rule_lookup",
    )


# ============================================================
# DEFAULT 3 OQC CATALOG SECTIONS
# This is your current UI data moved into MongoDB.
# Action values are kept as TBD. Fill real customer-approved values later.
# ============================================================
DEFAULT_HEADER = {
    "document_name": "Global Off Standard Catalogue for PCR Tyres",
    "document_no": "SOP-GQ&BE-001",
    "revision_no": "03",
    "document_status": "Approved",
    "process_owner": "Corporate",
    "security_classification": "Internal",
    "date_of_release": "",
    "date_of_applicability": "",
}


DEFAULT_ACTION_CATALOG_ROWS = [
    # ---------------- 101 ----------------
    {
        "catalog_code": "101",
        "section_name": "Tread blisters",
        "side": "tread",
        "condition_code": "101.1",
        "condition": "",
        "description": "No blisters in tread",
        "action_code": "TBD",
        "classification": "TBD",
        "replacement": "TBD",
        "scrap": "TBD",
    },
    {
        "catalog_code": "101",
        "section_name": "Tread blisters",
        "side": "tread",
        "condition_code": "101.2",
        "condition": "",
        "description": "Air entrapment between tread and cap strip",
        "action_code": "TBD",
        "classification": "TBD",
        "replacement": "TBD",
        "scrap": "TBD",
    },
    {
        "catalog_code": "101",
        "section_name": "Tread blisters",
        "side": "tread",
        "condition_code": "101.3",
        "condition": "",
        "description": "Air entrapment between tread and steel belt",
        "action_code": "TBD",
        "classification": "TBD",
        "replacement": "TBD",
        "scrap": "TBD",
    },

    # ---------------- 102 ----------------
    {
        "catalog_code": "102",
        "section_name": "Tread lightness",
        "side": "tread",
        "condition_code": "102.1",
        "condition": "",
        "description": "Rounding of imperfection < 2 mm",
        "action_code": "TBD",
        "classification": "TBD",
        "replacement": "TBD",
        "scrap": "TBD",
    },
    {
        "catalog_code": "102",
        "section_name": "Tread lightness",
        "side": "tread",
        "condition_code": "102.2",
        "condition": "",
        "description": "Length of imperfection ≤ 5 mm",
        "action_code": "TBD",
        "classification": "TBD",
        "replacement": "TBD",
        "scrap": "TBD",
    },
    {
        "catalog_code": "102",
        "section_name": "Tread lightness",
        "side": "tread",
        "condition_code": "102.3",
        "condition": "",
        "description": "Max. 2 imperfections in non-successive blocks",
        "action_code": "TBD",
        "classification": "TBD",
        "replacement": "TBD",
        "scrap": "TBD",
    },
    {
        "catalog_code": "102",
        "section_name": "Tread lightness",
        "side": "tread",
        "condition_code": "102.4",
        "condition": "",
        "description": "More than above",
        "action_code": "TBD",
        "classification": "TBD",
        "replacement": "TBD",
        "scrap": "TBD",
    },
    {
        "catalog_code": "102",
        "section_name": "Tread lightness",
        "side": "tread",
        "condition_code": "102.5",
        "condition": "",
        "description": "5 or more imperfections which are longer than 10mm and rounded 5 mm",
        "action_code": "TBD",
        "classification": "TBD",
        "replacement": "TBD",
        "scrap": "TBD",
    },

    # ---------------- 103 ----------------
    {
        "catalog_code": "103",
        "section_name": "Segment to segment flash (radial flash)",
        "side": "tread",
        "condition_code": "103.1",
        "condition": "",
        "description": "Thickness of flash < 0.5 mm",
        "action_code": "TBD",
        "classification": "TBD",
        "replacement": "TBD",
        "scrap": "TBD",
    },
    {
        "catalog_code": "103",
        "section_name": "Segment to segment flash (radial flash)",
        "side": "tread",
        "condition_code": "103.2",
        "condition": "",
        "description": "Height of flash < 0.5 mm",
        "action_code": "TBD",
        "classification": "TBD",
        "replacement": "TBD",
        "scrap": "TBD",
    },
    {
        "catalog_code": "103",
        "section_name": "Segment to segment flash (radial flash)",
        "side": "tread",
        "condition_code": "103.3",
        "condition": "",
        "description": "Height of flash between 0.5 mm and 1 mm",
        "action_code": "TBD",
        "classification": "TBD",
        "replacement": "TBD",
        "scrap": "TBD",
    },
    {
        "catalog_code": "103",
        "section_name": "Segment to segment flash (radial flash)",
        "side": "tread",
        "condition_code": "103.4",
        "condition": "",
        "description": "Height of flash between 1 mm and 1.5 mm",
        "action_code": "TBD",
        "classification": "TBD",
        "replacement": "TBD",
        "scrap": "TBD",
    },
    {
        "catalog_code": "103",
        "section_name": "Segment to segment flash (radial flash)",
        "side": "tread",
        "condition_code": "103.5",
        "condition": "",
        "description": "More than above",
        "action_code": "TBD",
        "classification": "TBD",
        "replacement": "TBD",
        "scrap": "TBD",
    },
    {
        "catalog_code": "103",
        "section_name": "Segment to segment flash (radial flash)",
        "side": "tread",
        "condition_code": "103.6",
        "condition": "",
        "description": "*all flash that seals off a groove must be cut",
        "action_code": "TBD",
        "classification": "TBD",
        "replacement": "TBD",
        "scrap": "TBD",
    },
]


DEFAULT_AI_MAP = [
    {
        "ai_label": "tread_blister",
        "side": "tread",
        "model_version": "v1.0",
        "catalog_code": "101",
        "catalog_defect_name": "Tread blisters",
        "min_confidence": 0.75,
        "active": True,
    },
    {
        "ai_label": "tread_lightness",
        "side": "tread",
        "model_version": "v1.0",
        "catalog_code": "102",
        "catalog_defect_name": "Tread lightness",
        "min_confidence": 0.75,
        "active": True,
    },
    {
        "ai_label": "segment_flash",
        "side": "tread",
        "model_version": "v1.0",
        "catalog_code": "103",
        "catalog_defect_name": "Segment to segment flash (radial flash)",
        "min_confidence": 0.75,
        "active": True,
    },
]


DEFAULT_RULES = [
    {
        "rule_id": "RULE_101_DEFAULT",
        "catalog_code": "101",
        "revision_no": "03",
        "rule_name": "Tread blister default rule",
        "measurement_field": "confidence",
        "operator": ">=",
        "value": 0.75,
        "action_code": "TBD",
        "final_decision": "REVIEW",
        "priority": 10,
        "active": True,
    },
    {
        "rule_id": "RULE_102_DEFAULT",
        "catalog_code": "102",
        "revision_no": "03",
        "rule_name": "Tread lightness default rule",
        "measurement_field": "confidence",
        "operator": ">=",
        "value": 0.75,
        "action_code": "TBD",
        "final_decision": "REVIEW",
        "priority": 10,
        "active": True,
    },
    {
        "rule_id": "RULE_103_DEFAULT",
        "catalog_code": "103",
        "revision_no": "03",
        "rule_name": "Segment flash default rule",
        "measurement_field": "confidence",
        "operator": ">=",
        "value": 0.75,
        "action_code": "TBD",
        "final_decision": "REVIEW",
        "priority": 10,
        "active": True,
    },
]


def seed_default_action_catalog(force: bool = False) -> Dict[str, Any]:
    """
    Insert default 3 OQC catalog sections into MongoDB.

    force=False:
        Does not duplicate existing rows.

    force=True:
        Deletes old Rev 03 catalog rows and inserts fresh data.
    """
    ensure_action_catalog_collections()

    now = datetime.utcnow()

    if force:
        action_catalog_col().delete_many({"revision_no": DEFAULT_HEADER["revision_no"]})
        ai_map_col().delete_many({"model_version": "v1.0"})
        rules_col().delete_many({"revision_no": DEFAULT_HEADER["revision_no"]})

    inserted_catalog = 0

    for row in DEFAULT_ACTION_CATALOG_ROWS:
        doc = {
            **DEFAULT_HEADER,
            **row,
            "active": True,
            "created_at": now,
            "updated_at": now,
        }

        result = action_catalog_col().update_one(
            {
                "catalog_code": doc["catalog_code"],
                "revision_no": doc["revision_no"],
                "condition_code": doc["condition_code"],
            },
            {"$setOnInsert": doc},
            upsert=True,
        )

        if result.upserted_id:
            inserted_catalog += 1

    inserted_maps = 0

    for row in DEFAULT_AI_MAP:
        doc = {
            **row,
            "created_at": now,
            "updated_at": now,
        }

        result = ai_map_col().update_one(
            {
                "ai_label": doc["ai_label"],
                "side": doc["side"],
                "model_version": doc["model_version"],
            },
            {"$setOnInsert": doc},
            upsert=True,
        )

        if result.upserted_id:
            inserted_maps += 1

    inserted_rules = 0

    for row in DEFAULT_RULES:
        doc = {
            **row,
            "created_at": now,
            "updated_at": now,
        }

        result = rules_col().update_one(
            {"rule_id": doc["rule_id"]},
            {"$setOnInsert": doc},
            upsert=True,
        )

        if result.upserted_id:
            inserted_rules += 1

    return {
        "ok": True,
        "inserted_catalog_rows": inserted_catalog,
        "inserted_ai_maps": inserted_maps,
        "inserted_rules": inserted_rules,
    }


# ============================================================
# FETCH FOR GUI
# ============================================================
def get_action_catalog_header(revision_no: str = "03") -> Dict[str, Any]:
    doc = action_catalog_col().find_one(
        {
            "revision_no": revision_no,
            "active": True,
        },
        {
            "_id": 0,
            "document_name": 1,
            "document_no": 1,
            "revision_no": 1,
            "document_status": 1,
            "process_owner": 1,
            "security_classification": 1,
            "date_of_release": 1,
            "date_of_applicability": 1,
        },
    )

    if not doc:
        return DEFAULT_HEADER.copy()

    return doc


def get_action_catalog_sections(revision_no: str = "03") -> List[Dict[str, Any]]:
    """
    Returns grouped catalog sections for GUI accordion.

    Output:
    [
        {
            "catalog_code": "101",
            "section_name": "Tread blisters",
            "rows": [...]
        }
    ]
    """
    cursor = action_catalog_col().find(
        {
            "revision_no": revision_no,
            "active": True,
        },
        {"_id": 0},
    ).sort(
        [
            ("catalog_code", 1),
            ("condition_code", 1),
        ]
    )

    grouped: Dict[str, Dict[str, Any]] = {}

    for row in cursor:
        code = str(row.get("catalog_code", ""))

        if code not in grouped:
            grouped[code] = {
                "catalog_code": code,
                "section_name": row.get("section_name", ""),
                "side": row.get("side", ""),
                "rows": [],
            }

        grouped[code]["rows"].append(row)

    return list(grouped.values())


def get_ai_catalog_mappings(model_version: str = "v1.0") -> List[Dict[str, Any]]:
    return list(
        ai_map_col().find(
            {
                "model_version": model_version,
                "active": True,
            },
            {"_id": 0},
        ).sort("catalog_code", 1)
    )


def get_action_decision_rules(revision_no: str = "03") -> List[Dict[str, Any]]:
    return list(
        rules_col().find(
            {
                "revision_no": revision_no,
                "active": True,
            },
            {"_id": 0},
        ).sort(
            [
                ("catalog_code", 1),
                ("priority", -1),
            ]
        )
    )


# ============================================================
# FUTURE: SAVE INSPECTION DECISION WITH CATALOG TRACEABILITY
# ============================================================
def save_inspection_action_decision(doc: Dict[str, Any]):
    """
    Use this later from live inference after AI detects defect.
    """
    payload = dict(doc)
    payload.setdefault("created_at", datetime.utcnow())
    return inspection_action_col().insert_one(payload)