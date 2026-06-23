from __future__ import annotations

import operator as _op
import re
import uuid
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from bson import ObjectId  # type: ignore
except Exception:  # pragma: no cover
    ObjectId = str  # type: ignore

try:
    from pymongo.errors import OperationFailure  # type: ignore
except Exception:  # pragma: no cover
    OperationFailure = Exception  # type: ignore
import os
from pathlib import Path

from src.COMMON.db import get_collection, ensure_collection, get_gridfs

# Existing collection kept for backward compatibility with your old page.
ACTION_CODE_CATALOG_COLLECTION = "Action Code Catalog"
AI_DEFECT_CATALOG_MAP_COLLECTION = "AI Defect Catalog Map"
ACTION_DECISION_RULES_COLLECTION = "Action Decision Rules"
INSPECTION_ACTION_DECISIONS_COLLECTION = "Inspection Action Decisions"

# New production-grade collections.
ACTION_CATALOG_VERSIONS_COLLECTION = "Action Catalog Versions"
ACTION_CATALOG_IMAGES_COLLECTION = "Action Catalog Images"
ACTION_CATALOG_AUDIT_COLLECTION = "Action Catalog Audit Log"

DISPOSITION_COLUMNS = ("oe", "replacement", "scrap")

DEFAULT_HEADER: Dict[str, Any] = {
    "document_name": "Global Off Standard Catalogue for PCR Tyres",
    "document_no": "SOP-GQ&BE-001",
    "revision_no": "03",
    "document_status": "Approved",
    "date_of_release": "05/07/2023",
    "date_of_applicability": "17/07/2023",
    "process_owner": "Corporate",
    "security_classification": "Internal",
}


def _now() -> datetime:
    return datetime.utcnow()


# -----------------------------------------------------------------------------
# Collection helpers
# -----------------------------------------------------------------------------
def action_catalog_col():
    return get_collection(ACTION_CODE_CATALOG_COLLECTION)


def catalog_versions_col():
    return get_collection(ACTION_CATALOG_VERSIONS_COLLECTION)


def catalog_images_col():
    return get_collection(ACTION_CATALOG_IMAGES_COLLECTION)


def ai_map_col():
    return get_collection(AI_DEFECT_CATALOG_MAP_COLLECTION)


def rules_col():
    return get_collection(ACTION_DECISION_RULES_COLLECTION)


def inspection_action_col():
    return get_collection(INSPECTION_ACTION_DECISIONS_COLLECTION)


def audit_col():
    return get_collection(ACTION_CATALOG_AUDIT_COLLECTION)



def _normalize_index_keys(keys):
    """Return PyMongo index key spec in a comparable list-of-tuples form."""
    if isinstance(keys, str):
        return [(keys, 1)]
    return [(str(k), int(v)) for k, v in keys]


def _drop_index_if_exists(collection, index_name: str) -> None:
    """Drop an index by name only when it exists. Safe for old deployed databases."""
    try:
        if index_name in collection.index_information():
            collection.drop_index(index_name)
    except Exception:
        # Index cleanup must never stop the application from opening.
        pass

def _safe_create_index(col, keys, name=None, **kwargs):
    """
    Mongo-safe index creation.

    If same key already exists with different name, do not create again.
    If same index name exists with different key, drop old named index and recreate.
    """
    wanted_keys = list(keys)

    for idx in col.list_indexes():
        existing_name = idx.get("name")
        existing_keys = list(idx.get("key", {}).items())

        # Same key already exists, maybe auto name:
        # version_id_1_catalog_code_1_image_order_1
        if existing_keys == wanted_keys:
            return existing_name

        # Same name but different key: drop and recreate
        if name and existing_name == name and existing_keys != wanted_keys:
            col.drop_index(existing_name)
            break

    return col.create_index(wanted_keys, name=name, **kwargs)

def _create_or_replace_index(collection, keys, *, name: str, **kwargs) -> None:
    """
    Create index safely in production.

    Reason:
        Old Apollo builds already created some indexes with the same name but
        different key specs. MongoDB then throws IndexKeySpecsConflict.
        This helper checks the existing named index, drops it only if the key
        spec/options are different, and recreates the correct production index.
    """
    desired_keys = _normalize_index_keys(keys)

    try:
        info = collection.index_information()
        existing = info.get(name)

        if existing:
            existing_keys = [(str(k), int(v)) for k, v in existing.get("key", [])]
            desired_unique = bool(kwargs.get("unique", False))
            existing_unique = bool(existing.get("unique", False))
            desired_partial = kwargs.get("partialFilterExpression")
            existing_partial = existing.get("partialFilterExpression")

            if (
                existing_keys != desired_keys
                or existing_unique != desired_unique
                or existing_partial != desired_partial
            ):
                collection.drop_index(name)

        collection.create_index(keys, name=name, **kwargs)

    except OperationFailure as exc:
        # Handles code 85/86 conflicts if a legacy index still exists with same name.
        code = getattr(exc, "code", None)
        if code in (85, 86):
            try:
                collection.drop_index(name)
            except Exception:
                pass
            collection.create_index(keys, name=name, **kwargs)
        else:
            raise

def ensure_action_catalog_collections() -> None:
    """Create collections and indexes used by the OSC/action-code module."""
    for name in (
        ACTION_CODE_CATALOG_COLLECTION,
        ACTION_CATALOG_VERSIONS_COLLECTION,
        ACTION_CATALOG_IMAGES_COLLECTION,
        AI_DEFECT_CATALOG_MAP_COLLECTION,
        ACTION_DECISION_RULES_COLLECTION,
        INSPECTION_ACTION_DECISIONS_COLLECTION,
        ACTION_CATALOG_AUDIT_COLLECTION,
    ):
        ensure_collection(name)

    _create_or_replace_index(catalog_versions_col(), "version_id", unique=True, name="uniq_version_id")
    _create_or_replace_index(catalog_versions_col(), [("is_current", 1), ("status", 1)], name="idx_current_status")
    _create_or_replace_index(catalog_versions_col(), [("revision_no", 1), ("local_version_no", 1)], name="idx_revision_local")

    # Remove old non-versioned unique index from earlier Apollo builds.
    # It blocks storing the same condition_code across multiple catalog versions.
    _drop_index_if_exists(action_catalog_col(), "uniq_catalog_revision_condition")

    # One row per condition per catalog version.
    _create_or_replace_index(
        action_catalog_col(),
        [("version_id", 1), ("condition_code", 1)],
        unique=True,
        # Ignore legacy rows that do not have version_id.
        partialFilterExpression={"version_id": {"$type": "string"}},
        name="uniq_version_condition",
    )
    _create_or_replace_index(
        action_catalog_col(),
        [("version_id", 1), ("catalog_code", 1), ("row_order", 1)],
        name="idx_version_section_order",
    )
    _create_or_replace_index(
        action_catalog_col(),
        [("version_id", 1), ("side", 1), ("active", 1)],
        name="idx_version_side_active",
    )

    _safe_create_index(
        catalog_images_col(),
        [("version_id", 1), ("catalog_code", 1), ("image_order", 1)],
        name="idx_images_section",
    )

    _safe_create_index(
        catalog_images_col(),
        [("gridfs_file_id", 1)],
        name="idx_catalog_images_gridfs_file_id",
    )

    _create_or_replace_index(
        ai_map_col(),
        [("ai_label", 1), ("side", 1), ("model_version", 1), ("active", 1)],
        name="idx_ai_label_side_model",
    )

    _create_or_replace_index(
        rules_col(),
        [("version_id", 1), ("catalog_code", 1), ("priority", -1), ("active", 1)],
        name="idx_rules_lookup",
    )

    _create_or_replace_index(
        inspection_action_col(),
        [("cycle_id", 1), ("side", 1), ("ai_label", 1)],
        name="idx_inspection_action_cycle",
    )


# -----------------------------------------------------------------------------
# Version model
# -----------------------------------------------------------------------------
def build_version_id(revision_no: str, local_version_no: str = "00") -> str:
    safe_rev = str(revision_no).strip().replace(" ", "_")
    safe_local = str(local_version_no).strip().replace(" ", "_")
    return f"OSC_REV_{safe_rev}_V{safe_local}"


def get_current_catalog_version() -> Optional[Dict[str, Any]]:
    ensure_action_catalog_collections()
    return catalog_versions_col().find_one(
        {"is_current": True, "status": "ACTIVE"},
        {"_id": 0},
        sort=[("published_at", -1), ("created_at", -1)],
    )


def get_catalog_versions(include_archived: bool = False) -> List[Dict[str, Any]]:
    ensure_action_catalog_collections()
    query: Dict[str, Any] = {}
    if not include_archived:
        query["status"] = {"$ne": "ARCHIVED"}
    return list(catalog_versions_col().find(query, {"_id": 0}).sort([("created_at", -1)]))


def get_version_or_current(version_id: Optional[str] = None) -> Dict[str, Any]:
    ensure_action_catalog_collections()
    if version_id:
        version = catalog_versions_col().find_one({"version_id": version_id}, {"_id": 0})
    else:
        version = get_current_catalog_version()
    if not version:
        raise RuntimeError("No OSC catalog version found. Import the SOP PDF or seed catalog first.")
    return version


def create_catalog_version(
    header: Optional[Dict[str, Any]] = None,
    *,
    version_id: Optional[str] = None,
    local_version_no: str = "00",
    source: str = "manual",
    status: str = "DRAFT",
    is_current: bool = False,
    created_by: str = "system",
    notes: str = "",
) -> Dict[str, Any]:
    ensure_action_catalog_collections()
    header = {**DEFAULT_HEADER, **(header or {})}
    version_id = version_id or build_version_id(header.get("revision_no", "03"), local_version_no)
    now = _now()

    if is_current:
        catalog_versions_col().update_many({}, {"$set": {"is_current": False}})

    doc = {
        "version_id": version_id,
        "revision_no": str(header.get("revision_no", "03")),
        "local_version_no": str(local_version_no),
        "source": source,
        "status": status,  # DRAFT / ACTIVE / ARCHIVED
        "is_current": bool(is_current),
        "locked": status == "ACTIVE",
        "header": header,
        "notes": notes,
        "created_by": created_by,
        "created_at": now,
        "updated_at": now,
        "published_at": now if status == "ACTIVE" else None,
    }
    catalog_versions_col().update_one({"version_id": version_id}, {"$setOnInsert": doc}, upsert=True)
    return catalog_versions_col().find_one({"version_id": version_id}, {"_id": 0}) or doc


def publish_catalog_version(version_id: str, operator: str = "operator") -> Dict[str, Any]:
    """Make a reviewed draft active. Production rule: publish, do not overwrite."""
    ensure_action_catalog_collections()
    version = catalog_versions_col().find_one({"version_id": version_id})
    if not version:
        raise RuntimeError(f"Catalog version not found: {version_id}")

    now = _now()
    catalog_versions_col().update_many(
        {"version_id": {"$ne": version_id}, "is_current": True},
        {"$set": {"is_current": False, "status": "ARCHIVED", "locked": True, "updated_at": now}},
    )
    catalog_versions_col().update_one(
        {"version_id": version_id},
        {"$set": {"is_current": True, "status": "ACTIVE", "locked": True, "published_at": now, "updated_at": now}},
    )
    audit_col().insert_one({
        "event": "PUBLISH_VERSION",
        "version_id": version_id,
        "operator": operator,
        "created_at": now,
    })
    return get_version_or_current(version_id)


def create_draft_from_version(base_version_id: Optional[str] = None, operator: str = "operator") -> Dict[str, Any]:
    """Clone an active version into a new editable draft."""
    ensure_action_catalog_collections()
    base = get_version_or_current(base_version_id)
    revision_no = base.get("revision_no", "03")
    # Local version uses timestamp to avoid collisions on plant PCs.
    local_version_no = datetime.now().strftime("%Y%m%d_%H%M%S")
    draft_version_id = build_version_id(revision_no, local_version_no)

    header = deepcopy(base.get("header") or DEFAULT_HEADER)
    draft = create_catalog_version(
        header,
        version_id=draft_version_id,
        local_version_no=local_version_no,
        source=f"draft_from:{base['version_id']}",
        status="DRAFT",
        is_current=False,
        created_by=operator,
        notes="Operator editable draft cloned from active OSC catalog.",
    )

    rows = list(action_catalog_col().find({"version_id": base["version_id"]}, {"_id": 0}))
    for row in rows:
        row["version_id"] = draft_version_id
        row["source_version_id"] = base["version_id"]
        row["created_at"] = _now()
        row["updated_at"] = _now()
    if rows:
        action_catalog_col().insert_many(rows, ordered=False)

    images = list(catalog_images_col().find({"version_id": base["version_id"]}, {"_id": 0}))
    for img in images:
        img["version_id"] = draft_version_id
        img["source_version_id"] = base["version_id"]
        img["created_at"] = _now()
        img["updated_at"] = _now()
    if images:
        catalog_images_col().insert_many(images, ordered=False)

    audit_col().insert_one({
        "event": "CREATE_DRAFT",
        "base_version_id": base["version_id"],
        "version_id": draft_version_id,
        "operator": operator,
        "created_at": _now(),
    })
    return draft


def delete_draft_catalog_version(version_id: str, operator: str = "operator") -> Dict[str, Any]:
    """
    Delete only an unpublished DRAFT catalog version.

    Safety:
    - ACTIVE / CURRENT versions cannot be deleted.
    - Image files are not deleted from disk because drafts reuse same reference images.
    - Only MongoDB draft rows, draft image references, draft rules, and draft version metadata are removed.
    """
    ensure_action_catalog_collections()

    version = catalog_versions_col().find_one({"version_id": version_id})
    if not version:
        raise RuntimeError(f"Catalog version not found: {version_id}")

    if version.get("status") != "DRAFT":
        raise RuntimeError("Only DRAFT versions can be deleted.")

    if version.get("is_current") or version.get("locked"):
        raise RuntimeError("Current/locked catalog version cannot be deleted.")

    now = _now()

    row_result = action_catalog_col().delete_many({"version_id": version_id})
    image_result = catalog_images_col().delete_many({"version_id": version_id})
    rule_result = rules_col().delete_many({"version_id": version_id})
    version_result = catalog_versions_col().delete_one({"version_id": version_id})

    audit_col().insert_one({
        "event": "DELETE_DRAFT",
        "version_id": version_id,
        "operator": operator,
        "deleted_rows": row_result.deleted_count,
        "deleted_images": image_result.deleted_count,
        "deleted_rules": rule_result.deleted_count,
        "deleted_versions": version_result.deleted_count,
        "created_at": now,
    })

    return {
        "ok": True,
        "version_id": version_id,
        "deleted_rows": row_result.deleted_count,
        "deleted_images": image_result.deleted_count,
        "deleted_rules": rule_result.deleted_count,
        "deleted_versions": version_result.deleted_count,
    }
# -----------------------------------------------------------------------------
# Import / save / fetch catalog rows
# -----------------------------------------------------------------------------
def import_catalog_payload(
    payload: Dict[str, Any],
    *,
    replace: bool = False,
    publish: bool = False,
    operator: str = "system",
) -> Dict[str, Any]:
    """Import parsed/curated SOP data.

    Expected payload:
    {
      "header": {...},
      "version_id": "OSC_REV_03_V00",
      "sections": [{"catalog_code": "101", "section_name": "...", "rows": [...]}],
      "images": [{"catalog_code": "101", "image_path": "...", ...}]
    }
    """
    ensure_action_catalog_collections()
    header = {**DEFAULT_HEADER, **payload.get("header", {})}
    version_id = payload.get("version_id") or build_version_id(header.get("revision_no", "03"), payload.get("local_version_no", "00"))
    status = "ACTIVE" if publish else "DRAFT"

    if replace:
        action_catalog_col().delete_many({"version_id": version_id})
        # Clean legacy non-versioned rows created by the earlier seed function.
        # Without this, old unique indexes/data can collide with the new versioned import.
        action_catalog_col().delete_many({
            "revision_no": str(header.get("revision_no", "03")),
            "$or": [
                {"version_id": {"$exists": False}},
                {"version_id": None},
                {"version_id": ""},
            ],
        })
        catalog_images_col().delete_many({"version_id": version_id})
        catalog_versions_col().delete_one({"version_id": version_id})

    create_catalog_version(
        header,
        version_id=version_id,
        local_version_no=payload.get("local_version_no", "00"),
        source=payload.get("source", "import"),
        status=status,
        is_current=publish,
        created_by=operator,
        notes=payload.get("notes", ""),
    )

    now = _now()
    docs: List[Dict[str, Any]] = []
    for sec_order, section in enumerate(payload.get("sections", []), start=1):
        code = str(section.get("catalog_code", "")).strip()
        section_name = str(section.get("section_name", "")).strip()
        side = str(section.get("side", infer_side_from_catalog_code(code))).strip()
        for row_order, row in enumerate(section.get("rows", []), start=1):
            condition_code = str(row.get("condition_code") or f"{code}.{row_order}").strip()
            docs.append({
                **header,
                "version_id": version_id,
                "revision_no": str(header.get("revision_no", "03")),
                "catalog_code": code,
                "section_name": section_name,
                "side": side,
                "condition_code": condition_code,
                "row_order": int(row.get("row_order", row_order)),
                "section_order": int(section.get("section_order", sec_order)),
                "description": str(row.get("description", "")).strip(),
                "action_code": str(row.get("action_code", "")).strip(),
                "classification": normalize_classification(row),
                "oe": bool(row.get("oe", False)),
                "replacement": bool(row.get("replacement", False)),
                "scrap": bool(row.get("scrap", False)),
                "critical_characteristic": bool(section.get("critical_characteristic", row.get("critical_characteristic", False))),
                "is_note": bool(row.get("is_note", False)),
                "active": bool(row.get("active", True)),
                "source_page": row.get("source_page") or section.get("source_page"),
                "created_at": now,
                "updated_at": now,
                "updated_by": operator,
            })

    if docs:
        for doc in docs:
            action_catalog_col().update_one(
                {"version_id": version_id, "condition_code": doc["condition_code"]},
                {"$set": doc},
                upsert=True,
            )

    image_docs = []
    for idx, img in enumerate(payload.get("images", []), start=1):
        image_docs.append({
            "version_id": version_id,
            "catalog_code": str(img.get("catalog_code", "")).strip(),
            "section_name": str(img.get("section_name", "")).strip(),
            "side": str(img.get("side", infer_side_from_catalog_code(str(img.get("catalog_code", ""))))).strip(),
            "description": img.get("description", ""),
            "condition_code": img.get("condition_code", ""),
            "action_code": img.get("action_code", ""),
            "classification": img.get("classification", ""),
            "image_order": int(img.get("image_order", idx)),
            "page_no": img.get("page_no"),

            # old local-path support
            "image_path": img.get("image_path"),

            # new MongoDB GridFS support
            "storage_type": img.get("storage_type", "gridfs" if img.get("gridfs_file_id") else "file"),
            "image_name": img.get("image_name") or img.get("file_name") or os.path.basename(str(img.get("image_path") or "")),
            "gridfs_bucket": img.get("gridfs_bucket") or "catalog_images_fs",
            "gridfs_file_id": img.get("gridfs_file_id"),
            "content_type": img.get("content_type", "image/png"),
            "file_size_bytes": img.get("file_size_bytes"),

            "bbox": img.get("bbox"),
            "active": True,
            "created_at": now,
            "updated_at": now,
        })
    if image_docs:
        catalog_images_col().insert_many(image_docs, ordered=False)

    if publish:
        publish_catalog_version(version_id, operator=operator)

    audit_col().insert_one({
        "event": "IMPORT_CATALOG",
        "version_id": version_id,
        "operator": operator,
        "row_count": len(docs),
        "image_count": len(image_docs),
        "replace": replace,
        "published": publish,
        "created_at": now,
    })
    return {"ok": True, "version_id": version_id, "row_count": len(docs), "image_count": len(image_docs), "published": publish}


def normalize_classification(row: Dict[str, Any]) -> str:
    if row.get("scrap"):
        return "SCRAP"
    if row.get("replacement") and row.get("oe"):
        return "OE / REPLACEMENT"
    if row.get("replacement"):
        return "REPLACEMENT"
    if row.get("oe"):
        return "OE"
    return str(row.get("classification", "")).strip()


def infer_side_from_catalog_code(code: str) -> str:
    code = str(code).strip()
    if not code or not code[0].isdigit():
        return "general"
    group = code[0]
    return {
        "1": "tread",
        "2": "shoulder",
        "3": "sidewall",
        "4": "bead",
        "5": "innerliner",
        "6": "curing",
        "7": "foreign_material",
    }.get(group, "general")


def get_action_catalog_header(version_id: Optional[str] = None) -> Dict[str, Any]:
    try:
        return dict(get_version_or_current(version_id).get("header") or DEFAULT_HEADER)
    except Exception:
        return DEFAULT_HEADER.copy()


def get_action_catalog_sections(version_id: Optional[str] = None, *, include_images: bool = True, include_inactive: bool = False) -> List[Dict[str, Any]]:
    version = get_version_or_current(version_id)
    query: Dict[str, Any] = {"version_id": version["version_id"]}
    if not include_inactive:
        query["active"] = True

    rows = list(action_catalog_col().find(query, {"_id": 0}).sort([("section_order", 1), ("catalog_code", 1), ("row_order", 1)]))
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        code = str(row.get("catalog_code", ""))
        if code not in grouped:
            grouped[code] = {
                "version_id": version["version_id"],
                "catalog_code": code,
                "section_name": row.get("section_name", ""),
                "side": row.get("side", ""),
                "section_order": row.get("section_order", 9999),
                "critical_characteristic": row.get("critical_characteristic", False),
                "rows": [],
                "images": [],
            }
        grouped[code]["rows"].append(row)

    if include_images and grouped:
        images = catalog_images_col().find({"version_id": version["version_id"], "active": True}, {"_id": 0}).sort([("catalog_code", 1), ("image_order", 1)])
        for img in images:
            code = str(img.get("catalog_code", ""))
            if code in grouped:
                grouped[code]["images"].append(img)

    return list(grouped.values())

def get_catalog_image_bytes(image_doc: Dict[str, Any]) -> Optional[bytes]:
    """
    Load OSC catalog image from MongoDB GridFS first.
    Fallback: old local image_path.
    """
    if not image_doc:
        return None

    gridfs_file_id = image_doc.get("gridfs_file_id")

    if gridfs_file_id:
        try:
            bucket = (
                image_doc.get("gridfs_bucket")
                or image_doc.get("bucket")
                or "catalog_images_fs"
            )

            file_id = gridfs_file_id
            if not isinstance(file_id, ObjectId):
                file_id = ObjectId(str(file_id))

            fs = get_gridfs(bucket=bucket)
            return fs.get(file_id).read()

        except Exception:
            return None

    image_path = image_doc.get("image_path")
    if image_path and os.path.exists(str(image_path)):
        try:
            return Path(str(image_path)).read_bytes()
        except Exception:
            return None

    return None

def save_header(version_id: str, header_updates: Dict[str, Any], operator: str = "operator") -> None:
    version = get_version_or_current(version_id)
    if version.get("locked"):
        raise RuntimeError("Active catalog is locked. Create a draft before editing.")
    header = {**(version.get("header") or {}), **header_updates}
    catalog_versions_col().update_one({"version_id": version_id}, {"$set": {"header": header, "updated_at": _now()}})
    audit_col().insert_one({"event": "SAVE_HEADER", "version_id": version_id, "operator": operator, "updates": header_updates, "created_at": _now()})


def save_catalog_rows(version_id: str, rows: Iterable[Dict[str, Any]], operator: str = "operator") -> Dict[str, Any]:
    version = get_version_or_current(version_id)
    if version.get("locked"):
        raise RuntimeError("Active catalog is locked. Create a draft before editing.")

    count = 0
    now = _now()
    for row in rows:
        condition_code = str(row.get("condition_code", "")).strip()
        if not condition_code:
            continue
        update = {
            "description": str(row.get("description", "")).strip(),
            "action_code": str(row.get("action_code", "")).strip(),
            "oe": bool(row.get("oe", False)),
            "replacement": bool(row.get("replacement", False)),
            "scrap": bool(row.get("scrap", False)),
            "classification": normalize_classification(row),
            "active": bool(row.get("active", True)),
            "updated_at": now,
            "updated_by": operator,
        }
        action_catalog_col().update_one({"version_id": version_id, "condition_code": condition_code}, {"$set": update})
        count += 1
    audit_col().insert_one({"event": "SAVE_ROWS", "version_id": version_id, "operator": operator, "row_count": count, "created_at": now})
    return {"ok": True, "updated_rows": count}


# -----------------------------------------------------------------------------
# AI label mapping + action decision resolver
# -----------------------------------------------------------------------------
def upsert_ai_catalog_mapping(
    *,
    ai_label: str,
    side: str,
    catalog_code: str,
    model_version: str = "v1.0",
    min_confidence: float = 0.0,
    active: bool = True,
    operator: str = "system",
) -> None:
    now = _now()
    ai_map_col().update_one(
        {"ai_label": ai_label, "side": side, "model_version": model_version},
        {"$set": {
            "ai_label": ai_label,
            "side": side,
            "model_version": model_version,
            "catalog_code": str(catalog_code),
            "min_confidence": float(min_confidence),
            "active": active,
            "updated_at": now,
            "updated_by": operator,
        }, "$setOnInsert": {"created_at": now}},
        upsert=True,
    )


def get_ai_catalog_mappings(model_version: str = "v1.0") -> List[Dict[str, Any]]:
    ensure_action_catalog_collections()
    return list(ai_map_col().find({"model_version": model_version, "active": True}, {"_id": 0}).sort([("side", 1), ("ai_label", 1)]))


_OPS = {
    ">": _op.gt,
    ">=": _op.ge,
    "<": _op.lt,
    "<=": _op.le,
    "==": _op.eq,
    "!=": _op.ne,
}


def _get_nested(data: Dict[str, Any], field: str, default: Any = None) -> Any:
    cur: Any = data
    for part in str(field).split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _rule_matches(rule: Dict[str, Any], measurements: Dict[str, Any]) -> bool:
    field = rule.get("measurement_field")
    op_name = str(rule.get("operator", ">=")).strip()
    expected = rule.get("value")
    actual = _get_nested(measurements, str(field), None)
    if actual is None:
        return False
    try:
        return bool(_OPS.get(op_name, _op.ge)(float(actual), float(expected)))
    except Exception:
        return False


def resolve_action_for_ai_defect(
    *,
    ai_label: str,
    side: str,
    measurements: Dict[str, Any],
    model_version: str = "v1.0",
    version_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve AI defect output into customer action/disposition with traceability.

    measurements example:
      {"confidence": 0.91, "length_mm": 7.2, "depth_mm": 0.6, "count": 2}
    """
    ensure_action_catalog_collections()
    version = get_version_or_current(version_id)

    mapping = ai_map_col().find_one({
        "ai_label": ai_label,
        "side": side,
        "model_version": model_version,
        "active": True,
    }, {"_id": 0})

    if not mapping:
        return {
            "resolved": False,
            "final_decision": "REVIEW",
            "reason": "No AI-to-OSC mapping found",
            "ai_label": ai_label,
            "side": side,
            "version_id": version["version_id"],
        }

    conf = float(measurements.get("confidence", 0.0) or 0.0)
    if conf < float(mapping.get("min_confidence", 0.0)):
        return {
            "resolved": False,
            "final_decision": "IGNORE_LOW_CONFIDENCE",
            "reason": "Below configured minimum confidence",
            "ai_label": ai_label,
            "side": side,
            "confidence": conf,
            "min_confidence": mapping.get("min_confidence"),
            "version_id": version["version_id"],
        }

    catalog_code = str(mapping.get("catalog_code"))
    rules = list(rules_col().find({
        "version_id": version["version_id"],
        "catalog_code": catalog_code,
        "active": True,
    }, {"_id": 0}).sort([("priority", -1)]))

    matched_rule = None
    for rule in rules:
        if _rule_matches(rule, measurements):
            matched_rule = rule
            break

    if matched_rule:
        condition_code = matched_rule.get("condition_code")
        row = action_catalog_col().find_one({"version_id": version["version_id"], "condition_code": condition_code}, {"_id": 0})
    else:
        # Safe fallback: choose the first row in that catalog section and keep human review.
        row = action_catalog_col().find_one({"version_id": version["version_id"], "catalog_code": catalog_code, "active": True}, {"_id": 0}, sort=[("row_order", 1)])

    if not row:
        return {
            "resolved": False,
            "final_decision": "REVIEW",
            "reason": "Mapped catalog section has no active rows",
            "ai_label": ai_label,
            "side": side,
            "catalog_code": catalog_code,
            "version_id": version["version_id"],
        }

    final_decision = "SCRAP" if row.get("scrap") else "REWORK_OR_REPLACEMENT" if row.get("replacement") else "ACCEPT_OR_REVIEW"
    return {
        "resolved": True,
        "version_id": version["version_id"],
        "revision_no": version.get("revision_no"),
        "ai_label": ai_label,
        "side": side,
        "model_version": model_version,
        "catalog_code": catalog_code,
        "section_name": row.get("section_name"),
        "condition_code": row.get("condition_code"),
        "condition_description": row.get("description"),
        "action_code": row.get("action_code"),
        "classification": row.get("classification"),
        "oe": bool(row.get("oe")),
        "replacement": bool(row.get("replacement")),
        "scrap": bool(row.get("scrap")),
        "final_decision": matched_rule.get("final_decision") if matched_rule else final_decision,
        "matched_rule_id": matched_rule.get("rule_id") if matched_rule else None,
        "measurements": measurements,
    }


def save_inspection_action_decision(doc: Dict[str, Any]):
    payload = dict(doc)
    payload.setdefault("created_at", _now())
    return inspection_action_col().insert_one(payload)


def resolve_and_save_inspection_action(
    *,
    cycle_id: str,
    sku_name: str,
    tyre_name: str,
    ai_label: str,
    side: str,
    measurements: Dict[str, Any],
    model_version: str = "v1.0",
    version_id: Optional[str] = None,
) -> Dict[str, Any]:
    decision = resolve_action_for_ai_defect(
        ai_label=ai_label,
        side=side,
        measurements=measurements,
        model_version=model_version,
        version_id=version_id,
    )
    decision.update({"cycle_id": cycle_id, "sku_name": sku_name, "tyre_name": tyre_name})
    save_inspection_action_decision(decision)
    return decision


# -----------------------------------------------------------------------------
# Backward compatible seed - only creates empty active version if DB is blank.
# Use tools/import_osc_catalog_from_pdf.py for full SOP import.
# -----------------------------------------------------------------------------
def seed_default_action_catalog(force: bool = False) -> Dict[str, Any]:
    ensure_action_catalog_collections()
    existing = get_current_catalog_version()
    if existing and not force:
        return {"ok": True, "message": "Catalog already exists", "version_id": existing["version_id"], "inserted_catalog_rows": 0}

    if force:
        action_catalog_col().delete_many({})
        catalog_versions_col().delete_many({})
        catalog_images_col().delete_many({})

    version_id = build_version_id(DEFAULT_HEADER["revision_no"], "00")
    create_catalog_version(DEFAULT_HEADER, version_id=version_id, local_version_no="00", source="empty_seed", status="ACTIVE", is_current=True)
    return {"ok": True, "message": "Created empty active catalog version. Import SOP PDF to load rows.", "version_id": version_id, "inserted_catalog_rows": 0}
