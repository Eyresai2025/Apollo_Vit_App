from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import mimetypes

from pymongo import MongoClient
from bson import ObjectId
import gridfs


# ============================================================
# CONFIG
# ============================================================

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "EyresQC_Apollo"

COLLECTION_NAME = "New SKU Images"
GRIDFS_BUCKET_NAME = "new_sku_images_fs"

REQUIRED_ZONES = ["sidewall1", "sidewall2", "tread", "inner", "bead"]


# ============================================================
# MANAGER CLASS
# ============================================================

class NewSKUImagesGridFSManager:
    def __init__(self, mongo_uri=MONGO_URI, db_name=DB_NAME):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]

        # Metadata collection
        self.new_sku_images_col = self.db[COLLECTION_NAME]

        # GridFS bucket for actual image binary data
        self.new_sku_images_fs = gridfs.GridFS(
            self.db,
            collection=GRIDFS_BUCKET_NAME
        )

    def now(self):
        return datetime.now()

    def guess_content_type(self, file_path: Path) -> str:
        content_type, _ = mimetypes.guess_type(str(file_path))
        if content_type:
            return content_type

        suffix = file_path.suffix.lower()

        if suffix in [".jpg", ".jpeg"]:
            return "image/jpeg"
        if suffix == ".png":
            return "image/png"
        if suffix == ".bmp":
            return "image/bmp"
        if suffix in [".tif", ".tiff"]:
            return "image/tiff"
        if suffix == ".webp":
            return "image/webp"

        return "application/octet-stream"

    def save_image_to_gridfs(
        self,
        image_path: str,
        sku_name: str,
        barcode: str,
        cycle_id: str,
        zone: str
    ) -> ObjectId:
        image_path_obj = Path(image_path)

        if not image_path_obj.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        content_type = self.guess_content_type(image_path_obj)
        file_size = image_path_obj.stat().st_size

        metadata = {
            "sku_name": sku_name,
            "barcode": barcode,
            "cycle_id": cycle_id,
            "zone": zone,
            "image_type": "new_sku_image",
            "original_filename": image_path_obj.name,
            "file_size_bytes": file_size,
            "content_type": content_type,
            "uploaded_at": self.now()
        }

        with open(image_path_obj, "rb") as f:
            gridfs_file_id = self.new_sku_images_fs.put(
                f,
                filename=image_path_obj.name,
                content_type=content_type,
                metadata=metadata
            )

        return gridfs_file_id

    def save_new_sku_cycle_images(
        self,
        sku_name: str,
        barcode: str,
        cycle_id: str,
        image_paths: Dict[str, str],
        operator: str = "admin"
    ) -> ObjectId:
        """
        Save one cycle of new SKU images.
        Required zones:
        sidewall1, sidewall2, tread, inner, bead
        """

        images_doc = {}

        for zone in REQUIRED_ZONES:
            image_path = image_paths.get(zone)

            if not image_path:
                images_doc[zone] = {
                    "available": False,
                    "image_name": None,
                    "gridfs_bucket": GRIDFS_BUCKET_NAME,
                    "gridfs_file_id": None,
                    "file_size_bytes": 0,
                    "content_type": None
                }
                continue

            image_path_obj = Path(image_path)

            gridfs_file_id = self.save_image_to_gridfs(
                image_path=image_path,
                sku_name=sku_name,
                barcode=barcode,
                cycle_id=cycle_id,
                zone=zone
            )

            images_doc[zone] = {
                "available": True,
                "image_name": image_path_obj.name,
                "gridfs_bucket": GRIDFS_BUCKET_NAME,
                "gridfs_file_id": gridfs_file_id,
                "file_size_bytes": image_path_obj.stat().st_size,
                "content_type": self.guess_content_type(image_path_obj)
            }

            print(f"[OK] Saved {zone}: {image_path_obj.name} -> GridFS ID: {gridfs_file_id}")

        doc = {
            "sku_name": sku_name,
            "barcode": barcode,
            "cycle_id": cycle_id,
            "type": "new_sku_images",
            "storage_type": "gridfs",
            "images": images_doc,
            "operator": operator,
            "created_at": self.now(),
            "updated_at": self.now()
        }

        inserted = self.new_sku_images_col.insert_one(doc)

        print("\n============================================================")
        print("[DONE] New SKU cycle images saved in MongoDB.")
        print(f"Collection Name : {COLLECTION_NAME}")
        print(f"Metadata Doc ID : {inserted.inserted_id}")
        print(f"GridFS Bucket   : {GRIDFS_BUCKET_NAME}")
        print("============================================================")

        return inserted.inserted_id

    def download_zone_image(
        self,
        cycle_doc_id: str,
        zone: str,
        output_path: str
    ):
        """
        Download one zone image back from MongoDB GridFS.
        """

        if zone not in REQUIRED_ZONES:
            raise ValueError(f"Invalid zone: {zone}")

        doc = self.new_sku_images_col.find_one({
            "_id": ObjectId(cycle_doc_id)
        })

        if not doc:
            raise RuntimeError(f"Cycle document not found: {cycle_doc_id}")

        zone_info = doc["images"].get(zone)

        if not zone_info or not zone_info.get("gridfs_file_id"):
            raise RuntimeError(f"No GridFS image found for zone: {zone}")

        gridfs_file_id = zone_info["gridfs_file_id"]

        grid_out = self.new_sku_images_fs.get(gridfs_file_id)
        data = grid_out.read()

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path_obj, "wb") as f:
            f.write(data)

        print(f"[OK] Downloaded {zone} image to: {output_path_obj}")

    def create_indexes(self):
        self.new_sku_images_col.create_index(
            [("sku_name", 1), ("cycle_id", 1)],
            unique=True
        )
        self.new_sku_images_col.create_index("barcode")
        self.new_sku_images_col.create_index("created_at")

        self.db[f"{GRIDFS_BUCKET_NAME}.files"].create_index(
            [("metadata.sku_name", 1), ("metadata.cycle_id", 1), ("metadata.zone", 1)]
        )

        print("[OK] Indexes created.")


# ============================================================
# MANUAL RUN EXAMPLE
# ============================================================

if __name__ == "__main__":
    manager = NewSKUImagesGridFSManager()
    manager.create_indexes()

    manager.save_new_sku_cycle_images(
        sku_name="SKU_001",
        barcode="APOLLO12345",
        cycle_id="CYCLE_20260617_001",
        operator="admin",
        image_paths={
            "sidewall1": r"media/raw images/1.jpg",
            "sidewall2": r"media/raw images/2.jpg",
            "tread": r"media/raw images/3.jpg",
            "inner": r"media/raw images/4.jpg",
            "bead": r"media/raw images/5.jpg",
        }
    )