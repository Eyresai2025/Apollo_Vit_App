#!/usr/bin/env python3
"""
Full sidewall1 pipeline:
1) prepare dataset from raw sidewall1 images
2) train ViT autoencoder on prepared train/good patches

Place this file in the same folder as:
- prepare_dataset_from_raw_sidewall1.py
- R_Detection_align_crop.py
- polarizer.py
- patchify_utils.py
- vit_autoencoder.py

Run:
    python full_sidewall1_training_pipeline.py
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from utils.checkpoint import load_checkpoint
from utils.data_loader import get_data_loaders
from train.train import train_model
from models.utils import get_vgg_model

from src.training.VIT_Training import prepare_dataset_from_raw_sidewall1 as prep
from vit_autoencoder import ViTEncoderDecoder, freeze_vit_layers

# ============================================================
# USER CONFIG
# ============================================================
@dataclass
class PipelineConfig:
    # ---------------- dataset preparation ----------------
    train_raw_dir: str = r"C:\Users\eyres\OneDrive - radometech.com\Desktop\Apollo\VIT+Autoencoder_AE\VIT_Training\Dataset\Train\Good"
    test_raw_dir: Optional[str] = None
    reference_image_path: str = r"C:\Users\eyres\OneDrive - radometech.com\Desktop\Apollo\VIT+Autoencoder_AE\VIT_Training\Reference\ref_sidewall1.png"
    dataset_root: str = r"C:\Users\eyres\OneDrive - radometech.com\Desktop\Apollo\VIT+Autoencoder_AE\VIT_Training\Prepared_Dataset"
    yolo_r_path: str = r"C:\Users\eyres\Downloads\R_Detection.pt"

    use_alignment: bool = True
    conf_thres_r: float = 0.3
    resize_crop_to: Tuple[int, int] = (2000, 10000)
    big_patch_h: int = 200
    big_patch_w: int = 200
    big_step_h: int = 200
    big_step_w: int = 200
    cover_edges: bool = True

    # ---------------- training ----------------
    run_name: str = "sidewall1_vit_autoencoder_run_01"
    epochs: int = 1
    batch_size: int = 2
    lr: float = 1e-4
    weight_decay: float = 1e-3
    num_workers: int = 0
    image_size: int = 224
    checkpoint_every: int = 10
    resume_checkpoint: Optional[str] = None

    # ---------------- pipeline control ----------------
    rebuild_dataset: bool = False
    device: str = "cuda"
    skip_test_preparation: bool = True

    #--------------- Augmentation ------------------------
    apply_augmentation: bool = True
    augment_train_good: bool = True
    augment_test_good: bool = False
    augment_test_anomalous: bool = False
    augment_in_subfolder: bool = False
    augment_subfolder_name: str = "Augmentation"
    brightness_up_factor: float = 1.5
    brightness_down_factor: float = 0.7


CFG = PipelineConfig()


# ============================================================
# BASIC HELPERS
# ============================================================
def log(msg: str) -> None:
    print(msg, flush=True)


def resolve_device(requested: str) -> str:
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested == "cuda":
        log("[WARN] CUDA not available. Falling back to CPU.")
    return "cpu"


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def folder_has_files(path_str: Optional[str]) -> bool:
    if not path_str:
        return False
    p = Path(path_str)
    return p.exists() and any(x.is_file() for x in p.rglob("*"))

def validate_dataset_root(dataset_root: Path, require_test: bool = False) -> None:
    required_dirs = [dataset_root / "train" / "good"]

    if require_test:
        required_dirs.extend([
            dataset_root / "test" / "good",
            dataset_root / "test" / "anomalous",
        ])

    missing = [str(p) for p in required_dirs if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing prepared dataset folders:\n" + "\n".join(missing))

    train_good_count = sum(1 for p in (dataset_root / "train" / "good").rglob("*") if p.is_file())
    if train_good_count == 0:
        raise ValueError(f"No patches found in {(dataset_root / 'train' / 'good')}")

    log(f"[DATASET] train/good patches: {train_good_count}")

    if require_test:
        test_good_count = sum(1 for p in (dataset_root / "test" / "good").rglob("*") if p.is_file())
        test_anom_count = sum(1 for p in (dataset_root / "test" / "anomalous").rglob("*") if p.is_file())
        log(f"[DATASET] test/good patches: {test_good_count}")
        log(f"[DATASET] test/anomalous patches: {test_anom_count}")

def configure_prep_module(cfg: PipelineConfig, device: str) -> None:
    prep.TRAIN_RAW_DIR = cfg.train_raw_dir

    if cfg.skip_test_preparation or not folder_has_files(cfg.test_raw_dir):
        prep.TEST_RAW_DIR = None
        log("[PREP] Test folder missing/empty. Skipping test dataset preparation.")
    else:
        prep.TEST_RAW_DIR = cfg.test_raw_dir

    prep.DATASET_ROOT = cfg.dataset_root
    prep.REF_IMAGE_PATH = cfg.reference_image_path
    prep.YOLO_R_PATH = cfg.yolo_r_path
    prep.DEVICE = device
    prep.USE_ALIGNMENT = cfg.use_alignment
    prep.CONF_THRES_R = cfg.conf_thres_r
    prep.RESIZE_CROP_TO = cfg.resize_crop_to
    prep.BIG_PATCH_H = cfg.big_patch_h
    prep.BIG_PATCH_W = cfg.big_patch_w
    prep.BIG_STEP_H = cfg.big_step_h
    prep.BIG_STEP_W = cfg.big_step_w
    prep.COVER_EDGES = cfg.cover_edges
    prep.TMP_STRIPS_ROOT = str(Path(cfg.dataset_root) / "_tmp_strips")

    prep.APPLY_AUGMENTATION = cfg.apply_augmentation
    prep.AUGMENT_TRAIN_GOOD = cfg.augment_train_good
    prep.AUGMENT_TEST_GOOD = cfg.augment_test_good
    prep.AUGMENT_TEST_ANOMALOUS = cfg.augment_test_anomalous
    prep.AUGMENT_IN_SUBFOLDER = cfg.augment_in_subfolder
    prep.AUGMENT_SUBFOLDER_NAME = cfg.augment_subfolder_name
    prep.BRIGHTNESS_UP_FACTOR = cfg.brightness_up_factor
    prep.BRIGHTNESS_DOWN_FACTOR = cfg.brightness_down_factor


def rebuild_dataset_root(dataset_root: Path) -> None:
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
        log(f"[DATASET] Removed old dataset root: {dataset_root}")
    dataset_root.mkdir(parents=True, exist_ok=True)


def run_dataset_preparation(cfg: PipelineConfig, device: str) -> None:
    dataset_root = Path(cfg.dataset_root)
    if cfg.rebuild_dataset:
        rebuild_dataset_root(dataset_root)

    configure_prep_module(cfg, device)
    log("[PIPELINE] Starting dataset preparation...")
    prep.main()

    require_test = (not cfg.skip_test_preparation) and folder_has_files(cfg.test_raw_dir)
    validate_dataset_root(dataset_root, require_test=require_test)

    log("[PIPELINE] Dataset preparation complete.")

def run_original_training(cfg: PipelineConfig, device: str) -> Path:
    dataset_root = Path(cfg.dataset_root)
    run_dir = dataset_root / "runs" / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    save_json(run_dir / "config.json", asdict(cfg))

    train_root = str(dataset_root / "train")

    train_loader, _ = get_data_loaders(
        train_data_path=train_root,
        test_data_path=None,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    model = ViTEncoderDecoder().to(device)
    freeze_vit_layers(model)
    vgg = get_vgg_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    start_epoch = 0
    if cfg.resume_checkpoint:
        start_epoch = load_checkpoint(model, optimizer, cfg.resume_checkpoint) + 1

    log(f"[TRAIN] Device: {device}")
    log(f"[TRAIN] Run dir: {run_dir}")
    log(f"[TRAIN] Epochs: {cfg.epochs}, Batch size: {cfg.batch_size}")
    log(f"[TRAIN] Train samples: {len(train_loader.dataset)}")
    log("[TRAIN] Entering train_model()")
    train_model(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        num_epochs=cfg.epochs,
        model_save_path=str(run_dir),
        start_epoch=start_epoch,
        vgg=vgg,
        device=device,
    )
    log("[TRAIN] train_model() returned")

    return run_dir

# ============================================================
# MAIN
# ============================================================
def main() -> None:
    device = resolve_device(CFG.device)
    log("[PIPELINE] Full sidewall1 pipeline started")

    run_dataset_preparation(CFG, device)

    run_dir = run_original_training(CFG, device)

    log("[PIPELINE] Training completed successfully")
    log(f"[PIPELINE] Outputs saved in: {run_dir}")


if __name__ == "__main__":
    main()


