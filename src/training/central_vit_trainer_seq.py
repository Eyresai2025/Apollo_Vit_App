import os
import sys
import json
import random
import importlib
import traceback
from pathlib import Path
from typing import Callable, Dict, Optional

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

PIPELINE_MODULE_MAP = {
    "sidewall1": "full_sidewall1_training_pipeline",
    "sidewall2": "full_sidewall2_training_pipeline",
    "innerwall": "full_innerwall_training_pipeline",
    "Tread": "full_tread_training_pipeline",
}


def _safe_name(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return "unknown_sku"
    bad = '<>:"/\\|?*'
    for ch in bad:
        text = text.replace(ch, "_")
    text = "_".join(text.split())
    text = text.strip("._")
    return text or "unknown_sku"


def _log(logger: Optional[Callable[[str], None]], msg: str) -> None:
    print(msg, flush=True)
    if logger:
        try:
            logger(msg)
        except Exception:
            pass


def _pick_reference_image(serial_root: Path) -> Optional[Path]:
    """
    Pick one random image only from the serial root folder itself.
    This means it will use the images saved directly under:
        .../<sku>/<serial>/
    and NOT from train/good.
    """
    candidates = [
        p for p in serial_root.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    if not candidates:
        return None
    return random.choice(candidates)


def _validate_train_good(train_good_dir: Path) -> None:
    if not train_good_dir.exists():
        raise FileNotFoundError(f"Missing train/good folder: {train_good_dir}")

    img_count = sum(
        1 for p in train_good_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if img_count == 0:
        raise ValueError(f"No images found in train/good: {train_good_dir}")


def _import_pipeline_module(vit_training_root: str, module_name: str):
    vit_training_root = Path(vit_training_root).resolve()

    if not vit_training_root.exists():
        raise FileNotFoundError(f"VIT training root not found: {vit_training_root}")

    expected_module_file = vit_training_root / f"{module_name}.py"
    if not expected_module_file.exists():
        raise FileNotFoundError(f"Pipeline file not found: {expected_module_file}")

    inserted = False
    vit_training_root_str = str(vit_training_root)

    if vit_training_root_str not in sys.path:
        sys.path.insert(0, vit_training_root_str)
        inserted = True

    try:
        module = importlib.import_module(module_name)
        module = importlib.reload(module)
        return module
    finally:
        if inserted and vit_training_root_str in sys.path:
            try:
                sys.path.remove(vit_training_root_str)
            except Exception:
                pass


def _run_one_training(
    vit_training_root: str,
    pipeline_kind: str,
    sku_name: str,
    serial: str,
    serial_root: Path,
    yolo_r_path: str,
    device: str,
    rebuild_dataset: bool,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict:
    if pipeline_kind not in PIPELINE_MODULE_MAP:
        raise ValueError(f"Unknown pipeline kind: {pipeline_kind}")

    module_name = PIPELINE_MODULE_MAP[pipeline_kind]
    module = _import_pipeline_module(vit_training_root, module_name)

    train_good_dir = serial_root / "train" / "good"
    _validate_train_good(train_good_dir)

    reference_image = _pick_reference_image(serial_root)
    if reference_image is None:
        raise FileNotFoundError(
            f"No direct reference image found in serial root: {serial_root}\n"
            f"Expected at least one image directly inside the serial folder."
        )

    # SAVE DIRECTLY INSIDE SERIAL FOLDER
    dataset_root = serial_root / "Prepared_Dataset"
    run_name = f"{_safe_name(sku_name)}_{serial}_{pipeline_kind}"

    dataset_root.mkdir(parents=True, exist_ok=True)

    cfg = module.CFG
    cfg.train_raw_dir = str(train_good_dir)
    cfg.test_raw_dir = None
    cfg.reference_image_path = str(reference_image)
    cfg.dataset_root = str(dataset_root)
    cfg.yolo_r_path = str(yolo_r_path)

    cfg.run_name = run_name
    cfg.device = device
    cfg.rebuild_dataset = rebuild_dataset
    cfg.skip_test_preparation = True
    cfg.resume_checkpoint = None
    cfg.num_workers = 0

    config_dump = {
        "sku_name": sku_name,
        "camera_serial": serial,
        "pipeline_kind": pipeline_kind,
        "train_raw_dir": str(train_good_dir),
        "reference_image_path": str(reference_image),
        "dataset_root": str(dataset_root),
        "run_name": run_name,
        "yolo_r_path": str(yolo_r_path),
        "device": device,
        "rebuild_dataset": rebuild_dataset,
    }

    config_path = serial_root / f"{run_name}_launch_config.json"
    config_path.write_text(json.dumps(config_dump, indent=2), encoding="utf-8")

    _log(logger, f"[TRAIN] serial={serial} | pipeline={pipeline_kind}")
    _log(logger, f"[TRAIN] train_raw_dir={train_good_dir}")
    _log(logger, f"[TRAIN] reference_image={reference_image}")
    _log(logger, f"[TRAIN] dataset_root={dataset_root}")

    module.main()

    run_dir = dataset_root / "runs" / run_name

    return {
        "success": True,
        "sku_name": sku_name,
        "camera_serial": serial,
        "pipeline_kind": pipeline_kind,
        "train_raw_dir": str(train_good_dir),
        "reference_image_path": str(reference_image),
        "dataset_root": str(dataset_root),
        "run_dir": str(run_dir),
        "config_path": str(config_path),
    }


def run_training_for_sku(
    media_path: str,
    sku_name: str,
    serial_pipeline_map: Dict[str, Optional[str]],
    vit_training_root: str,
    yolo_r_path: str,
    device: str = "cuda",
    rebuild_dataset: bool = True,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict:
    """
    Expected capture structure:
        media/new_sku_images/<sku_name>/<serial>/train/good/   -> first 10 images
        media/new_sku_images/<sku_name>/<serial>/              -> next 10 images

    Training uses:
        train_raw_dir      = <serial>/train/good
        reference_image    = one random image from <serial>/ root
        outputs            = <serial>/VIT_Training/...
    """
    sku_folder = _safe_name(sku_name)
    sku_root = Path(media_path) / "new_sku_images" / sku_folder

    if not sku_root.exists():
        raise FileNotFoundError(f"SKU folder not found: {sku_root}")

    summary = {
        "sku_name": sku_name,
        "sku_folder": sku_folder,
        "sku_root": str(sku_root),
        "results": [],
        "skipped": [],
    }

    _log(logger, f"[TRAIN] Starting SKU training: {sku_name}")
    _log(logger, f"[TRAIN] SKU root: {sku_root}")

    for serial, pipeline_kind in serial_pipeline_map.items():
        if not pipeline_kind:
            summary["skipped"].append({
                "camera_serial": serial,
                "reason": "No pipeline configured"
            })
            _log(logger, f"[SKIP] serial={serial} | no pipeline configured")
            continue

        serial_root = sku_root / str(serial)
        if not serial_root.exists():
            summary["skipped"].append({
                "camera_serial": serial,
                "reason": f"Serial folder not found: {serial_root}"
            })
            _log(logger, f"[SKIP] serial={serial} | folder missing")
            continue

        try:
            result = _run_one_training(
                vit_training_root=vit_training_root,
                pipeline_kind=pipeline_kind,
                sku_name=sku_name,
                serial=str(serial),
                serial_root=serial_root,
                yolo_r_path=yolo_r_path,
                device=device,
                rebuild_dataset=rebuild_dataset,
                logger=logger,
            )
            summary["results"].append(result)
            _log(logger, f"[DONE] serial={serial} | pipeline={pipeline_kind}")

        except Exception as e:
            err = {
                "success": False,
                "camera_serial": str(serial),
                "pipeline_kind": pipeline_kind,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            summary["results"].append(err)
            _log(logger, f"[FAIL] serial={serial} | {e}")

    summary_path = sku_root / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)

    _log(logger, f"[TRAIN] Summary saved: {summary_path}")
    return summary