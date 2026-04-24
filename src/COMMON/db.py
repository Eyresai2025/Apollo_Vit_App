from __future__ import annotations

import hashlib
import os
from datetime import datetime
from typing import Any, Dict, Optional

import cv2  # type: ignore
import numpy as np  # type: ignore
from bson import ObjectId  # type: ignore
from gridfs import GridFS  # type: ignore
from pymongo import MongoClient  # type: ignore

from src.COMMON.common import load_env


# =========================
# ENV / CONFIG
# =========================
_env = load_env()

DB_URL: str = _env.get("DATABASE_URL", "mongodb://localhost:27017/")
DB_NAME: str = "EyresQC_Apollo"   # <-- force same DB for everything

GRIDFS_BUCKET: str = _env.get("GRIDFS_BUCKET", "fs")

TYRE_DETAILS_COLLECTION = "TYRE DETAILS"
NEW_SKU_META_COLLECTION = "New SKU"
ACCOUNTS_COLLECTION_NAME = "Accounts"
REPEATABILITY_COLLECTION = "Repeatability"


# =========================
# SINGLETON CLIENT
# =========================
_client: Optional[MongoClient] = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(DB_URL)
    return _client


def get_db(db_name: Optional[str] = None):
    name = db_name or DB_NAME
    return get_client()[name]


def get_collection(collection_name: str, db_name: Optional[str] = None):
    return get_db(db_name)[collection_name]


def get_gridfs(bucket: Optional[str] = None, db_name: Optional[str] = None) -> GridFS:
    return GridFS(get_db(db_name), collection=bucket or GRIDFS_BUCKET)


def ensure_collection(collection_name: str, db_name: Optional[str] = None) -> None:
    db = get_db(db_name)
    if collection_name not in db.list_collection_names():
        db.create_collection(collection_name)


# =========================
# FIXED COLLECTION HELPERS
# =========================
def get_tyre_details_collection():
    return get_collection(TYRE_DETAILS_COLLECTION)


def get_new_sku_collection():
    return get_collection(NEW_SKU_META_COLLECTION)


def get_accounts_collection():
    return get_collection(ACCOUNTS_COLLECTION_NAME)


def get_repeatability_collection():
    return get_collection(REPEATABILITY_COLLECTION)


# =========================
# ACCOUNTS
# =========================
def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def create_account(full_name: str, username: str, email: str, password: str):
    col = get_accounts_collection()
    existing = col.find_one({"$or": [{"username": username}, {"email": email}]})
    if existing:
        return False, "Username or Email already exists."

    doc = {
        "full_name": full_name,
        "username": username,
        "email": email,
        "password": _hash_password(password),
        "created_at": datetime.utcnow(),
        "is_active": True,
    }
    col.insert_one(doc)
    return True, "Account created successfully."


def authenticate_user(identifier: str, password: str):
    col = get_accounts_collection()
    user = col.find_one({"$or": [{"username": identifier}, {"email": identifier}]})

    if not user:
        return False, "Account not found.", None

    if user["password"] != _hash_password(password):
        return False, "Incorrect password.", None

    if not user.get("is_active", True):
        return False, "Account is disabled.", None

    return True, "Login successful.", user


def reset_password(identifier: str, new_password: str):
    col = get_accounts_collection()
    result = col.update_one(
        {"$or": [{"username": identifier}, {"email": identifier}]},
        {"$set": {"password": _hash_password(new_password)}},
    )

    if result.matched_count == 0:
        return False, "Account not found."

    return True, "Password updated successfully."


# =========================
# GENERIC IMAGE / GRIDFS HELPERS
# =========================
def nparray_to_bytes(db, img_array, filename, cycle):
    date = datetime.now().strftime("%d-%m-%Y")
    success, encoded = cv2.imencode(".jpg", img_array)
    if not success:
        raise ValueError("Failed to encode image array to JPG bytes.")

    image_bytes = encoded.tobytes()
    fs = GridFS(db)
    file_id = fs.put(
        image_bytes,
        filename=filename,
        cycle_no=cycle,
        inspection_date=date,
    )
    return file_id


def recent_cycle(mydb):
    file_collection = mydb[TYRE_DETAILS_COLLECTION]
    current_date = datetime.now()

    recent_document = file_collection.find_one({}, sort=[("inspectionDateTime", -1)])

    if recent_document:
        latest_inspection_date_str = recent_document.get("inspectionDate")

        if latest_inspection_date_str:
            try:
                latest_inspection_date = datetime.strptime(
                    latest_inspection_date_str, "%d-%m-%Y"
                )

                if current_date.date() != latest_inspection_date.date():
                    return "1"
                else:
                    return str(int(recent_document.get("cycle_no", 0)) + 1)

            except ValueError:
                print(
                    f"Error: Invalid date format in database: {latest_inspection_date_str}"
                )
                return "1"
        else:
            print("Warning: No inspectionDate found in the most recent document.")
            return "1"
    else:
        print("No previous documents found. Starting new cycle.")
        return "1"


def db_to_images(cycle, db, download_loc, date):
    os.makedirs(download_loc, exist_ok=True)

    file_collection = db[f"{GRIDFS_BUCKET}.files"]
    file_list = list(
        file_collection.find(
            {"cycle_no": cycle, "inspection_date": date},
            {"_id": False, "filename": True},
        )
    )

    fs = GridFS(db)

    for file in file_list:
        image_doc = fs.find_one(file)

        if image_doc:
            image_data = fs.get(image_doc._id).read()
            retrieved_image_data = np.frombuffer(image_data, dtype=np.uint8)
            retrieved_image = cv2.imdecode(retrieved_image_data, cv2.IMREAD_COLOR)

            if retrieved_image is not None:
                cv2.imwrite(
                    os.path.join(download_loc, file["filename"]),
                    retrieved_image,
                )
            else:
                print(f"Failed to decode image: {file['filename']}")
        else:
            print(f"Image with filename '{file}' not found.")

# =========================
# CYCLE METADATA IN MONGODB
# =========================
def _extract_cycle_no(cycle_id: str) -> str:
    try:
        return str(int(str(cycle_id).split("_")[-1]))
    except Exception:
        return str(cycle_id)


def _count_defect_sides(side_results: dict) -> int:
    count = 0
    for _, side_data in side_results.items():
        label = str(side_data.get("final_label", "")).upper()
        if label in ["DEFECT", "FAILED", "INVALID", "SUSPECT"]:
            count += 1
    return count


def save_cycle_metadata(result: dict):
    col = get_tyre_details_collection()

    now = datetime.now()
    inspection_datetime = now.strftime("%Y-%m-%d %H:%M:%S")
    inspection_date = now.strftime("%d-%m-%Y")

    cycle_id = str(result.get("cycle_id", ""))
    cycle_no = _extract_cycle_no(cycle_id)

    image_map = result.get("image_map", {}) or {}
    side_results = result.get("side_results", {}) or {}

    doc = {
        "cycle_no": cycle_no,
        "cycle_id": cycle_id,
        "inspectionDateTime": inspection_datetime,
        "inspectionDate": inspection_date,
        "sku_name": result.get("sku_name"),
        "tyre_name": result.get("tyre_name"),
        "cycle_decision": result.get("final_label"),
        "final_label": result.get("final_label"),
        "cycle_latency_sec": result.get("cycle_latency_sec"),
        "defect": str(result.get("final_label", "")).upper() != "OK",
        "numberOfDefects": _count_defect_sides(side_results),

        "sidewall1_image_name": os.path.basename(image_map["sidewall1"]) if image_map.get("sidewall1") else None,
        "sidewall2_image_name": os.path.basename(image_map["sidewall2"]) if image_map.get("sidewall2") else None,
        "innerwall_image_name": os.path.basename(image_map["innerwall"]) if image_map.get("innerwall") else None,
        "tread_image_name": os.path.basename(image_map["tread"]) if image_map.get("tread") else None,
        "bead_image_name": os.path.basename(image_map["bead"]) if image_map.get("bead") else None,

        "image_map": image_map,
        "side_results": side_results,
        "cycle_output_dir": result.get("cycle_dir"),
    }

    return col.insert_one(doc)


# =========================
# NEW SKU
# =========================
def save_new_sku_image(
    file_path: str,
    label: str,
    capture_id: str,
    sku_meta: Optional[Dict[str, Any]] = None,
    meta_collection: Optional[str] = None,
    gridfs_bucket: Optional[str] = None,
) -> ObjectId:
    sku_meta = sku_meta or {}
    meta_collection = meta_collection or NEW_SKU_META_COLLECTION
    gridfs_bucket = gridfs_bucket or GRIDFS_BUCKET

    ensure_collection(meta_collection)

    fs = get_gridfs(bucket=gridfs_bucket)
    meta_col = get_collection(meta_collection)

    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)

    ext = os.path.splitext(file_name)[1].lower()
    content_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"

    with open(file_path, "rb") as f:
        file_id = fs.put(
            f,
            filename=file_name,
            contentType=content_type,
            metadata={
                "capture_id": capture_id,
                "label": label,
                "sku_meta": sku_meta,
                "source_file_path": file_path,
                "file_size": file_size,
                "created_at": datetime.utcnow(),
            },
        )

    meta_doc = {
        "type": "image_meta",
        "capture_id": capture_id,
        "label": label,
        "file_name": file_name,
        "file_path": file_path,
        "file_size": file_size,
        "status": "stored",
        "created_at": datetime.utcnow(),
        "sku_meta": sku_meta,
        "gridfs_bucket": gridfs_bucket,
        "gridfs_file_id": file_id,
    }
    meta_col.insert_one(meta_doc)

    return file_id


# =========================
# REPEATABILITY
# =========================
def insert_repeatability_log(doc: dict):
    col = get_repeatability_collection()
    return col.insert_one(doc)


def fetch_gridfs_bytes(file_id: str | ObjectId, bucket: Optional[str] = None) -> bytes:
    fs = get_gridfs(bucket=bucket)
    oid = file_id if isinstance(file_id, ObjectId) else ObjectId(file_id)
    return fs.get(oid).read()