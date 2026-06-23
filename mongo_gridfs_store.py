from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

from pymongo import MongoClient
from bson import ObjectId
import gridfs


MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "EyresQC_Apollo"


class MongoGridFSManager:
    def __init__(self, mongo_uri: str = MONGO_URI, db_name: str = DB_NAME):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]

        # Separate GridFS buckets for clean organization
        self.input_images_fs = gridfs.GridFS(self.db, collection="input_images_fs")
        self.output_images_fs = gridfs.GridFS(self.db, collection="output_images_fs")
        self.ai_models_fs = gridfs.GridFS(self.db, collection="ai_models_fs")
        self.catalog_images_fs = gridfs.GridFS(self.db, collection="catalog_images_fs")

        # Metadata collections
        self.ai_models_col = self.db["AI Models"]
        self.sku_input_images_col = self.db["Input Images"]
        self.sku_output_images_col = self.db["Output Images"]
        self.action_catalog_images_col = self.db["Action Catalog Images"]

    def _guess_content_type(self, file_path: str) -> str:
        suffix = Path(file_path).suffix.lower()

        if suffix in [".jpg", ".jpeg"]:
            return "image/jpeg"
        if suffix == ".png":
            return "image/png"
        if suffix in [".bmp"]:
            return "image/bmp"
        if suffix in [".tif", ".tiff"]:
            return "image/tiff"
        if suffix in [".pt", ".pth"]:
            return "application/octet-stream"
        if suffix in [".onnx"]:
            return "application/onnx"
        if suffix in [".engine"]:
            return "application/octet-stream"

        return "application/octet-stream"

    def save_file_to_gridfs(
        self,
        file_path: str,
        bucket_type: str,
        metadata: Optional[dict] = None
    ) -> ObjectId:
        file_path_obj = Path(file_path)

        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if bucket_type == "input":
            fs = self.input_images_fs
        elif bucket_type == "output":
            fs = self.output_images_fs
        elif bucket_type == "model":
            fs = self.ai_models_fs
        elif bucket_type == "catalog":
            fs = self.catalog_images_fs
        else:
            raise ValueError("bucket_type must be input, output, model, or catalog")

        metadata = metadata or {}
        metadata.update({
            "original_path": str(file_path_obj),
            "file_size_bytes": file_path_obj.stat().st_size,
            "uploaded_at": datetime.now(),
            "content_type": self._guess_content_type(str(file_path_obj))
        })

        with open(file_path_obj, "rb") as f:
            file_id = fs.put(
                f,
                filename=file_path_obj.name,
                metadata=metadata,
                content_type=metadata["content_type"]
            )

        return file_id

    def save_ai_model(
        self,
        sku_name: str,
        model_path: str,
        model_name: str,
        model_version: str,
        model_type: str = "YOLOv8 Segmentation",
        classes: Optional[list] = None,
        confidence_threshold: float = 0.30,
        active: bool = True
    ) -> ObjectId:
        metadata = {
            "sku_name": sku_name,
            "model_name": model_name,
            "model_version": model_version,
            "model_type": model_type
        }

        model_gridfs_id = self.save_file_to_gridfs(
            file_path=model_path,
            bucket_type="model",
            metadata=metadata
        )

        doc = {
            "sku_name": sku_name,
            "model_name": model_name,
            "model_version": model_version,
            "model_type": model_type,
            "classes": classes or [],
            "confidence_threshold": confidence_threshold,
            "active": active,

            # Actual model binary stored in MongoDB GridFS
            "model_gridfs_file_id": model_gridfs_id,

            # Optional reference only
            "original_model_path": str(Path(model_path)),

            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }

        inserted = self.ai_models_col.insert_one(doc)
        print(f"[OK] AI model saved.")
        print(f"     Metadata ID     : {inserted.inserted_id}")
        print(f"     GridFS Model ID : {model_gridfs_id}")

        return inserted.inserted_id

    def save_sku_input_images(
        self,
        sku_name: str,
        barcode: str,
        cycle_id: str,
        image_paths: Dict[str, str]
    ) -> ObjectId:
        required_zones = ["sidewall1", "sidewall2", "tread", "inner", "bead"]

        images_doc = {}

        for zone in required_zones:
            path = image_paths.get(zone)

            if not path:
                images_doc[zone] = {
                    "available": False,
                    "image_name": None,
                    "gridfs_file_id": None,
                    "original_path": None
                }
                continue

            metadata = {
                "sku_name": sku_name,
                "barcode": barcode,
                "cycle_id": cycle_id,
                "zone": zone,
                "image_type": "input"
            }

            gridfs_file_id = self.save_file_to_gridfs(
                file_path=path,
                bucket_type="input",
                metadata=metadata
            )

            images_doc[zone] = {
                "available": True,
                "image_name": Path(path).name,
                "gridfs_file_id": gridfs_file_id,
                "original_path": str(Path(path))
            }

        doc = {
            "sku_name": sku_name,
            "barcode": barcode,
            "cycle_id": cycle_id,
            "type": "input_images",
            "images": images_doc,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }

        inserted = self.sku_input_images_col.insert_one(doc)
        print(f"[OK] SKU input images saved.")
        print(f"     Metadata ID: {inserted.inserted_id}")

        return inserted.inserted_id

    def save_sku_output_images(
        self,
        sku_name: str,
        barcode: str,
        cycle_id: str,
        image_paths: Dict[str, str],
        ai_model_version: str,
        final_result: str,
        defect_summary: Optional[Dict[str, dict]] = None
    ) -> ObjectId:
        required_zones = ["sidewall1", "sidewall2", "tread", "inner", "bead"]

        images_doc = {}
        defect_summary = defect_summary or {}

        for zone in required_zones:
            path = image_paths.get(zone)
            zone_defects = defect_summary.get(zone, {})

            if not path:
                images_doc[zone] = {
                    "available": False,
                    "output_image_name": None,
                    "gridfs_file_id": None,
                    "original_path": None,
                    "defect_count": 0,
                    "defects": []
                }
                continue

            metadata = {
                "sku_name": sku_name,
                "barcode": barcode,
                "cycle_id": cycle_id,
                "zone": zone,
                "image_type": "output",
                "ai_model_version": ai_model_version
            }

            gridfs_file_id = self.save_file_to_gridfs(
                file_path=path,
                bucket_type="output",
                metadata=metadata
            )

            images_doc[zone] = {
                "available": True,
                "output_image_name": Path(path).name,
                "gridfs_file_id": gridfs_file_id,
                "original_path": str(Path(path)),
                "defect_count": zone_defects.get("defect_count", 0),
                "defects": zone_defects.get("defects", [])
            }

        doc = {
            "sku_name": sku_name,
            "barcode": barcode,
            "cycle_id": cycle_id,
            "type": "output_images",
            "ai_model_version": ai_model_version,
            "final_result": final_result,
            "images": images_doc,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }

        inserted = self.sku_output_images_col.insert_one(doc)
        print(f"[OK] SKU output images saved.")
        print(f"     Metadata ID: {inserted.inserted_id}")

        return inserted.inserted_id

    def save_action_catalog_image_to_gridfs(
        self,
        action_catalog_image_doc_id: str,
        image_file_path: str
    ) -> ObjectId:
        doc_id = ObjectId(action_catalog_image_doc_id)

        existing_doc = self.action_catalog_images_col.find_one({"_id": doc_id})
        if not existing_doc:
            raise RuntimeError(f"Action Catalog Images document not found: {doc_id}")

        metadata = {
            "source_collection": "Action Catalog Images",
            "source_doc_id": str(doc_id),
            "catalog_code": existing_doc.get("catalog_code"),
            "section_name": existing_doc.get("section_name"),
            "side": existing_doc.get("side"),
            "version_id": existing_doc.get("version_id"),
            "image_order": existing_doc.get("image_order")
        }

        gridfs_file_id = self.save_file_to_gridfs(
            file_path=image_file_path,
            bucket_type="catalog",
            metadata=metadata
        )

        self.action_catalog_images_col.update_one(
            {"_id": doc_id},
            {
                "$set": {
                    "gridfs_file_id": gridfs_file_id,
                    "image_path": str(Path(image_file_path)),
                    "updated_at": datetime.now()
                }
            }
        )

        print("[OK] Action catalog image binary saved into MongoDB GridFS.")
        print(f"     Action Catalog Doc ID : {doc_id}")
        print(f"     GridFS Image ID       : {gridfs_file_id}")

        return gridfs_file_id

    def download_file_from_gridfs(
        self,
        gridfs_file_id: str,
        bucket_type: str,
        output_path: str
    ):
        file_id = ObjectId(gridfs_file_id)

        if bucket_type == "input":
            fs = self.input_images_fs
        elif bucket_type == "output":
            fs = self.output_images_fs
        elif bucket_type == "model":
            fs = self.ai_models_fs
        elif bucket_type == "catalog":
            fs = self.catalog_images_fs
        else:
            raise ValueError("bucket_type must be input, output, model, or catalog")

        grid_out = fs.get(file_id)

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path_obj, "wb") as f:
            f.write(grid_out.read())

        print(f"[OK] Downloaded file from GridFS to: {output_path}")


if __name__ == "__main__":
    manager = MongoGridFSManager()

    # -------------------------------
    # Example 1: Save AI model in DB
    # -------------------------------
    # manager.save_ai_model(
    #     sku_name="SKU_001",
    #     model_path=r"c:\Users\Hi\Downloads\democlassification_model.pt",
    #     model_name="Apollo_Tyre_Defect_Model",
    #     model_version="v1.0",
    #     classes=["crack", "flash", "blow"],
    #     confidence_threshold=0.30,
    #     active=True
    # )

    # --------------------------------
    # Example 2: Save SKU input images
    # --------------------------------
    # manager.save_sku_input_images(
    #     sku_name="SKU_001",
    #     barcode="APOLLO12345",
    #     cycle_id="CYCLE_20260617_001",
    #     image_paths={
    #         "sidewall1": r"media/raw images/1.jpg",
    #         "sidewall2": r"media/raw images/2.jpg",
    #         "tread": r"media/raw images/3.jpg",
    #         "inner": r"media/raw images/4.jpg",
    #         "bead": r"media/raw images/5.jpg",
    #     }
    # )

    # ---------------------------------
    # Example 3: Save SKU output images
    # ---------------------------------
    # manager.save_sku_output_images(
    #     sku_name="SKU_001",
    #     barcode="APOLLO12345",
    #     cycle_id="CYCLE_20260617_001",
    #     ai_model_version="v1.0",
    #     final_result="DEFECT",
    #     image_paths={
    #         "sidewall1": r"media/raw images/1.jpg",
    #         "sidewall2": r"media/raw images/2.jpg",
    #         "tread": r"media/raw images/3.jpg",
    #         "inner": r"media/raw images/4.jpg",
    #         "bead": r"media/raw images/5.jpg",
    #     },
    #     defect_summary={
    #         "sidewall1": {"defect_count": 2, "defects": ["flash", "crack"]},
    #         "sidewall2": {"defect_count": 0, "defects": []},
    #         "tread": {"defect_count": 1, "defects": ["blow"]},
    #         "inner": {"defect_count": 0, "defects": []},
    #         "bead": {"defect_count": 0, "defects": []},
    #     }
    # )

    # print("MongoDB GridFS manager ready.")