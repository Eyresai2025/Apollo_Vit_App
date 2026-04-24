import os
import sys
import json
import random
import importlib
import traceback
from pathlib import Path
from typing import Callable, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

PIPELINE_MODULE_MAP = {
    "sidewall1": "full_sidewall1_training_pipeline",
    "sidewall2": "full_sidewall2_training_pipeline",
    "innerwall": "full_innerwall_training_pipeline",
    "Tread": "full_tread_training_pipeline",
}

# -----------------------------
# SAFE MODE SETTINGS
# -----------------------------
# Dataset prep runs in parallel on CPU
PREP_MAX_WORKERS = 2
PREP_DEVICE = "cpu"

# Actual model training runs one-by-one on GPU
TRAIN_DEVICE = "cuda"


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


def _configure_module_cfg(
    module,
    sku_name: str,
    serial: str,
    serial_root: Path,
    yolo_r_path: str,
    device: str,
    rebuild_dataset: bool,
):
    train_good_dir = serial_root / "train" / "good"
    _validate_train_good(train_good_dir)

    reference_image = _pick_reference_image(serial_root)
    if reference_image is None:
        raise FileNotFoundError(
            f"No direct reference image found in serial root: {serial_root}\n"
            f"Expected at least one image directly inside the serial folder."
        )

    dataset_root = serial_root / "Prepared_Dataset"
    run_name = f"{_safe_name(sku_name)}_{serial}"

    cfg = module.CFG
    cfg.train_raw_dir = str(train_good_dir)
    cfg.test_raw_dir = None
    cfg.reference_image_path = str(reference_image)
    cfg.dataset_root = str(dataset_root)
    cfg.yolo_r_path = str(yolo_r_path)

    cfg.run_name = f"{run_name}_{PIPELINE_MODULE_MAP_INV[module.__name__]}"
    cfg.device = device
    cfg.rebuild_dataset = rebuild_dataset
    cfg.skip_test_preparation = True
    cfg.resume_checkpoint = None
    cfg.num_workers = 0

    return cfg, train_good_dir, reference_image, dataset_root


# reverse lookup once
PIPELINE_MODULE_MAP_INV = {v: k for k, v in PIPELINE_MODULE_MAP.items()}


def _prepare_one_dataset(
    vit_training_root: str,
    pipeline_kind: str,
    sku_name: str,
    serial: str,
    serial_root: Path,
    yolo_r_path: str,
    rebuild_dataset: bool,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict:
    if pipeline_kind not in PIPELINE_MODULE_MAP:
        raise ValueError(f"Unknown pipeline kind: {pipeline_kind}")

    module_name = PIPELINE_MODULE_MAP[pipeline_kind]
    module = _import_pipeline_module(vit_training_root, module_name)

    if not hasattr(module, "run_dataset_preparation"):
        raise AttributeError(f"{module_name} does not expose run_dataset_preparation(cfg, device)")

    cfg, train_good_dir, reference_image, dataset_root = _configure_module_cfg(
        module=module,
        sku_name=sku_name,
        serial=serial,
        serial_root=serial_root,
        yolo_r_path=yolo_r_path,
        device=PREP_DEVICE,
        rebuild_dataset=rebuild_dataset,
    )

    config_dump = {
        "phase": "prepare",
        "sku_name": sku_name,
        "camera_serial": serial,
        "pipeline_kind": pipeline_kind,
        "train_raw_dir": str(train_good_dir),
        "reference_image_path": str(reference_image),
        "dataset_root": str(dataset_root),
        "yolo_r_path": str(yolo_r_path),
        "device": PREP_DEVICE,
        "rebuild_dataset": rebuild_dataset,
    }

    config_path = serial_root / f"{_safe_name(sku_name)}_{serial}_{pipeline_kind}_prepare_config.json"
    config_path.write_text(json.dumps(config_dump, indent=2), encoding="utf-8")

    _log(logger, f"[PREP] serial={serial} | pipeline={pipeline_kind} | device={PREP_DEVICE}")
    _log(logger, f"[PREP] train_raw_dir={train_good_dir}")
    _log(logger, f"[PREP] reference_image={reference_image}")
    _log(logger, f"[PREP] dataset_root={dataset_root}")

    module.run_dataset_preparation(cfg, PREP_DEVICE)

    return {
        "sku_name": sku_name,
        "camera_serial": serial,
        "pipeline_kind": pipeline_kind,
        "train_raw_dir": str(train_good_dir),
        "reference_image_path": str(reference_image),
        "dataset_root": str(dataset_root),
        "prepare_config_path": str(config_path),
    }


def _train_one_prepared(
    vit_training_root: str,
    pipeline_kind: str,
    sku_name: str,
    serial: str,
    serial_root: Path,
    yolo_r_path: str,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict:
    if pipeline_kind not in PIPELINE_MODULE_MAP:
        raise ValueError(f"Unknown pipeline kind: {pipeline_kind}")

    module_name = PIPELINE_MODULE_MAP[pipeline_kind]
    module = _import_pipeline_module(vit_training_root, module_name)

    if not hasattr(module, "run_original_training"):
        raise AttributeError(f"{module_name} does not expose run_original_training(cfg, device)")

    cfg, train_good_dir, reference_image, dataset_root = _configure_module_cfg(
        module=module,
        sku_name=sku_name,
        serial=serial,
        serial_root=serial_root,
        yolo_r_path=yolo_r_path,
        device=TRAIN_DEVICE,
        rebuild_dataset=False,   # IMPORTANT: do not rebuild, dataset is already prepared
    )

    cfg.run_name = f"{_safe_name(sku_name)}_{serial}_{pipeline_kind}"

    config_dump = {
        "phase": "train",
        "sku_name": sku_name,
        "camera_serial": serial,
        "pipeline_kind": pipeline_kind,
        "train_raw_dir": str(train_good_dir),
        "reference_image_path": str(reference_image),
        "dataset_root": str(dataset_root),
        "run_name": cfg.run_name,
        "yolo_r_path": str(yolo_r_path),
        "device": TRAIN_DEVICE,
        "rebuild_dataset": False,
    }

    config_path = serial_root / f"{cfg.run_name}_train_config.json"
    config_path.write_text(json.dumps(config_dump, indent=2), encoding="utf-8")

    _log(logger, f"[TRAIN] serial={serial} | pipeline={pipeline_kind} | device={TRAIN_DEVICE}")
    _log(logger, f"[TRAIN] dataset_root={dataset_root}")

    run_dir = module.run_original_training(cfg, TRAIN_DEVICE)

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
    device: str = "cuda",   # retained for API compatibility with GUI
    rebuild_dataset: bool = True,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict:
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
    _log(logger, f"[MODE] Parallel dataset preparation: ON | workers={PREP_MAX_WORKERS} | device={PREP_DEVICE}")
    _log(logger, f"[MODE] Sequential model training: ON | device={TRAIN_DEVICE}")

    jobs = []
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

        jobs.append({
            "serial": str(serial),
            "pipeline_kind": pipeline_kind,
            "serial_root": serial_root,
        })

    # --------------------------------------
    # PHASE 1: parallel dataset preparation
    # --------------------------------------
    prep_ok = {}

    if jobs:
        max_workers = min(PREP_MAX_WORKERS, len(jobs))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_map = {
                ex.submit(
                    _prepare_one_dataset,
                    vit_training_root,
                    job["pipeline_kind"],
                    sku_name,
                    job["serial"],
                    job["serial_root"],
                    yolo_r_path,
                    rebuild_dataset,
                    logger,
                ): job for job in jobs
            }

            for fut in as_completed(future_map):
                job = future_map[fut]
                serial = job["serial"]
                pipeline_kind = job["pipeline_kind"]

                try:
                    prep_res = fut.result()
                    prep_ok[serial] = prep_res
                    _log(logger, f"[PREP-DONE] serial={serial} | pipeline={pipeline_kind}")
                except Exception as e:
                    err = {
                        "success": False,
                        "camera_serial": str(serial),
                        "pipeline_kind": pipeline_kind,
                        "phase": "prepare",
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }
                    summary["results"].append(err)
                    _log(logger, f"[PREP-FAIL] serial={serial} | {e}")

    # --------------------------------------
    # PHASE 2: sequential GPU training
    # --------------------------------------
    for job in jobs:
        serial = job["serial"]
        pipeline_kind = job["pipeline_kind"]
        serial_root = job["serial_root"]

        if serial not in prep_ok:
            continue

        try:
            result = _train_one_prepared(
                vit_training_root=vit_training_root,
                pipeline_kind=pipeline_kind,
                sku_name=sku_name,
                serial=serial,
                serial_root=serial_root,
                yolo_r_path=yolo_r_path,
                logger=logger,
            )
            summary["results"].append(result)
            _log(logger, f"[DONE] serial={serial} | pipeline={pipeline_kind}")

        except Exception as e:
            err = {
                "success": False,
                "camera_serial": str(serial),
                "pipeline_kind": pipeline_kind,
                "phase": "train",
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