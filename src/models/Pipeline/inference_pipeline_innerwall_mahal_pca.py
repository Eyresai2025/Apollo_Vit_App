"""
Full pipeline for Innerwall Template Matching using ViT embeddings with:
1) Preprocessing using polarizer-like reflection removal
2) Alignment and cropping to a reference image using R detection
3) Patch embedding extraction using ViT
4) Distance metrics:
      - cosine
      - euclidean
      - mahalanobis
      - mahalanobis_pca
5) Threshold calibration from good images
6) Inference on Images compared to the threshold set
7) YOLO on ViT-selected defect patches

IMPORTANT:
- If you change DISTANCE_METRIC, rerun MODE="calibrate" first.
- For mahalanobis_pca, this script fits a GLOBAL PCA and then computes per-(r,c)
  Mahalanobis statistics in PCA space.
  
"""

import os
import re
import json
import shutil
import time
from pathlib import Path
from collections import defaultdict
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from sklearn.decomposition import PCA
import uuid
import tempfile

from src.COMMON.common import innerwall_dimensions
from src.models.defect_dimension import area_defect_innerwall, cor_innerwall

from src.models.Pipeline.polarizer import polarizer_optimized

from src.models.Pipeline.R_inner_mapping_alignment import (
    build_r_detector,
    align_and_crop_to_reference_fixed_band,
    get_reference_r_band,
    crop_from_reference_band_info,
)

from src.models.Pipeline.patchify_utils import patchify_index_grouped
from src.models.Pipeline.vit_autoencoder import ViTEncoderDecoder
from src.models.Pipeline.yolo_patch_classifier import load_yolo_seg, segment_patch_paths

try:
    from src.models.Pipeline.checkpoint import load_checkpoint
except Exception:
    from src.models.Pipeline.checkpoint import load_checkpoint

# =========================================================
# CONFIG
# =========================================================
MODE = "calibrate"   # "calibrate" or "infer"+
DEBUG_SAVE_INTERMEDIATE = False

# calibration good raw images
CALIB_GOOD_DIR = r"C:\Users\DELL\Downloads\sidewall_qutrac\sidewall_def\calib"

# new incoming tires
PROD_RAW_DIR = r"C:\Users\DELL\Downloads\sidewall_qutrac\sidewall_def\prod"
REF_IMAGE_PATH = r"C:\Users\DELL\Downloads\sidewall_qutrac\sidewall_def\reference\ref_innerwall.png"
OUTPUT_DIR = r"C:\Users\DELL\Downloads\sidewall_qutrac\sidewall_def\output"

CHECKPOINT_PATH = r"C:\Users\DELL\Downloads\ssl_epoch_50.pth"
YOLO_R_PATH = r"C:\Users\DELL\Downloads\R_Detection.pt"
YOLO_SEG_MODEL_PATH = r"C:\Users\DELL\OneDrive - radometech.com\Desktop\Apollo\VIT+Autoencoder\Model\best_classify_5def.pt"

DEVICE = "cuda"
SEG_DEVICE = DEVICE

IMG_SIZE = 224
BATCH_SIZE = 128

# "cosine" or "euclidean" or "mahalanobis" or "mahalanobis_pca"
DISTANCE_METRIC = "mahalanobis_pca"

# safe default:
# cosine normalizes internally anyway, but keep False here so euclidean/mahal work naturally
NORMALIZE_EMBEDDINGS = False

# PCA config
PCA_N_COMPONENTS = 32
PCA_FIT_ON_MAP_ONLY = True

# Mahalanobis config
MAHALANOBIS_MODE = "diag"       # "diag" or "full"
MAHALANOBIS_REG_EPS = 1e-3
MAHALANOBIS_MIN_SAMPLES = 3

USE_INTERMEDIATE_BLOCKS = True
TARGET_BLOCK_INDICES = [4, 5, 6, 7, 8, 9]

# "mean" or "concat"
BLOCK_FUSION = "concat"

# optional: normalize each block feature before fusion
NORMALIZE_EACH_BLOCK = False

USE_ALIGNMENT = True
RESIZE_CROP_TO = (2000, 10000)  # (W, H)
FINAL_STITCHED_SIZE = (2000, 10000) 
SLICE_H = 4200
SLICE_W = 4096
CONF_THRES_R = 0.3

BIG_PATCH_H = 200
BIG_PATCH_W = 200
BIG_STEP_H = 200
BIG_STEP_W = 200
COVER_EDGES = True

# with 5 images total:
# first 5 -> embedding map
# next 5 -> threshold calibration
MAP_IMAGE_COUNT = 5
THRESH_IMAGE_COUNT = 5

# patchwise threshold settings
LOCAL_PERCENTILE = 99.0
Z_SCORE_THRESHOLD = 3.0
SIGMA_FLOOR = 0.01
DEFECT_DIMENSION_DECIMALS = 2

# =========================================================
# IMPROVED THRESHOLDING CONFIG
# =========================================================
USE_LEAVE_ONE_OUT_THRESHOLDS = True

MAD_FLOOR = 0.01

# =========================================================
# AUGMENTATION CONFIG (CALIBRATION ONLY)
# =========================================================
AUGMENT_CALIB = False
AUGMENT_MAP = False
AUGMENT_THRESH = False

AUG_TRANSLATIONS = [(-3, 0), (3, 0), (0, -3), (0, 3)]
AUG_ROTATIONS = [-2.0, 2.0]
AUG_BRIGHTNESS_FACTORS = [0.95, 1.05]
AUG_CONTRAST_FACTORS = [0.95, 1.05]

# =========================================================
# YOLO STAGE
# =========================================================
USE_YOLO_SEG = False
SEG_CONF_THRESHOLD = 0.84
KEEP_SEG_CLASSES = None

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
RC_RE = re.compile(r"__r(\d+)_c(\d+)\.(png|jpg|jpeg|bmp|tif|tiff)$", re.IGNORECASE)

# =========================================================
# DEFECTIVE CALIB IMAGE SUPPORT
# =========================================================
USE_DEFECT_CALIB_IMAGES = True
DEFECT_CALIB_PREFIXES = ("def",)   # def1, def2, def3 ...

# Patches to ignore ONLY for def* calibration images
DEFECT_IGNORE_RCS = {
    (40, 3),
    (40, 4),
    (40, 5),
    (40, 6),
}

# =========================================================
# SIMPLE OUTLIER-AWARE THRESHOLDING
# =========================================================
REMOVE_TOP_OUTLIER_PER_RC = True
OUTLIER_RATIO = 1.8   # remove largest if largest > 1.8 * second_largest

LOCAL_PERCENTILE_AFTER_CLEAN = 95.0

# =========================================================
# UTILITIES
# =========================================================
def make_model():
    model = ViTEncoderDecoder(
        vit_model_name="vit_base_patch16_224",
        image_size=224,
    )
    return model


def _build_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])


def _list_images(root_dir):
    paths = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(root, f))
    return sorted(paths)


def _batched(paths, batch_size=BATCH_SIZE):
    batch = []
    for p in paths:
        batch.append(p)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _reset_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def translate_image_bgr(img, tx, ty):
    h, w = img.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    out = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return out


def rotate_image_bgr(img, angle_deg):
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    out = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return out


def adjust_brightness_contrast_bgr(img, brightness_factor=1.0, contrast_factor=1.0):
    x = img.astype(np.float32)
    x = (x - 127.5) * contrast_factor + 127.5
    x = x * brightness_factor
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def generate_calibration_augmentations(crop_bgr):
    variants = []

    for tx, ty in AUG_TRANSLATIONS:
        aug = translate_image_bgr(crop_bgr, tx=tx, ty=ty)
        sx = f"{tx:+d}".replace("+", "p").replace("-", "m")
        sy = f"{ty:+d}".replace("+", "p").replace("-", "m")
        suffix = f"t{sx}_{sy}"
        variants.append((suffix, aug))

    for ang in AUG_ROTATIONS:
        aug = rotate_image_bgr(crop_bgr, angle_deg=ang)
        sa = f"{ang:+.0f}".replace("+", "p").replace("-", "m")
        suffix = f"r{sa}"
        variants.append((suffix, aug))

    for bf in AUG_BRIGHTNESS_FACTORS:
        aug = adjust_brightness_contrast_bgr(crop_bgr, brightness_factor=bf, contrast_factor=1.0)
        suffix = f"b{int(round(bf * 100))}"
        variants.append((suffix, aug))

    for cf in AUG_CONTRAST_FACTORS:
        aug = adjust_brightness_contrast_bgr(crop_bgr, brightness_factor=1.0, contrast_factor=cf)
        suffix = f"c{int(round(cf * 100))}"
        variants.append((suffix, aug))

    return variants


def create_augmented_patch_dirs_from_crop(crop_bgr, base_name, aug_root_dir):
    os.makedirs(aug_root_dir, exist_ok=True)

    aug_patch_dirs = []
    variants = generate_calibration_augmentations(crop_bgr)

    for suffix, aug_bgr in variants:
        single_aug_dir = os.path.join(aug_root_dir, f"{base_name}_{suffix}")
        _reset_dir(single_aug_dir)

        aug_path = os.path.join(single_aug_dir, f"{base_name}.png")
        cv2.imwrite(aug_path, aug_bgr)

        aug_patches_dir = patchify_index_grouped(
            single_aug_dir,
            patch_h=BIG_PATCH_H,
            patch_w=BIG_PATCH_W,
            step_h=BIG_STEP_H,
            step_w=BIG_STEP_W,
            cover_edges=COVER_EDGES,
        )
        aug_patch_dirs.append(aug_patches_dir)

    return aug_patch_dirs


def parse_rc_from_patch_name(fname):
    m = RC_RE.search(os.path.basename(fname))
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))

def get_feature_dim():
    if USE_INTERMEDIATE_BLOCKS:
        if BLOCK_FUSION == "mean":
            return 768
        elif BLOCK_FUSION == "concat":
            return 768 * len(TARGET_BLOCK_INDICES)
        else:
            raise ValueError(f"Unsupported BLOCK_FUSION: {BLOCK_FUSION}")
    else:
        return 768


def to_gray(img):
    if img is None:
        raise ValueError("Input image is None")

    if img.ndim == 2:
        return img.copy()

    if img.ndim == 3 and img.shape[2] == 1:
        return img[:, :, 0].copy()

    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    raise ValueError(f"Unsupported image shape for to_gray: {img.shape}")


def choose_threshold_and_stats(
    key,
    thresholds_by_rc,
    mu_by_rc,
    sigma_by_rc,
):
    if key not in thresholds_by_rc:
        return None

    local_thr = float(thresholds_by_rc[key])

    return {
        "threshold_source": "local",
        "threshold_used": local_thr,
        "mu_used": float(mu_by_rc.get(key, local_thr)),
        "sigma_used": max(float(sigma_by_rc.get(key, SIGMA_FLOOR)), SIGMA_FLOOR),
        "local_threshold_used": local_thr,
    }

def remove_ignored_rc_patches_from_dir(patches_dir, ignore_rcs):
    """
    Physically delete masked RC patch files from a patch directory.
    After this, all downstream bank/stat/threshold code will naturally ignore them.
    """
    removed = 0
    for p in _list_images(patches_dir):
        r, c = parse_rc_from_patch_name(os.path.basename(p))
        if r is None or c is None:
            continue
        if (r, c) in ignore_rcs:
            try:
                os.remove(p)
                removed += 1
            except Exception as e:
                print(f"[WARN] failed removing masked patch {p} | {e}")
    return removed

def is_defect_calib_image(path):
    stem = Path(path).stem.lower()
    return any(stem.startswith(prefix.lower()) for prefix in DEFECT_CALIB_PREFIXES)



# =========================================================
# VIT EMBEDDINGS
# =========================================================

@torch.inference_mode()
def extract_vit_features(model, batch, target_block_indices, fusion="concat", normalize_each_block=False, normalize_final=False):
    """
    Extract pooled patch-token embeddings from multiple ViT blocks.

    Args:
        model: ViTEncoderDecoder
        batch: [B,3,H,W]
        target_block_indices: list like [4,5,6,7,8,9]
        fusion: "mean" or "concat"
        normalize_each_block: normalize each block embedding before fusion
        normalize_final: normalize final fused embedding

    Returns:
        emb: [B, D]
    """
    enc = model.encoder

    x = enc.patch_embed(batch)

    if hasattr(enc, "_pos_embed"):
        x = enc._pos_embed(x)

    if hasattr(enc, "patch_drop"):
        x = enc.patch_drop(x)

    if hasattr(enc, "norm_pre"):
        x = enc.norm_pre(x)

    wanted = set(target_block_indices)
    collected = []

    for idx, blk in enumerate(enc.blocks):
        x = blk(x)

        if idx in wanted:
            patch_tokens = x[:, 1:, :]
            emb = patch_tokens.mean(dim=1)

            if normalize_each_block:
                emb = F.normalize(emb, dim=1)

            collected.append((idx, emb))

    if len(collected) == 0:
        raise RuntimeError(f"No block outputs collected for indices: {target_block_indices}")

    idx_to_emb = {idx: emb for idx, emb in collected}
    ordered_embs = [idx_to_emb[idx] for idx in target_block_indices if idx in idx_to_emb]

    if fusion == "mean":
        fused = torch.stack(ordered_embs, dim=0).mean(dim=0)
    elif fusion == "concat":
        fused = torch.cat(ordered_embs, dim=1)
    else:
        raise ValueError(f"Unsupported fusion mode: {fusion}")

    if normalize_final:
        fused = F.normalize(fused, dim=1)

    return fused

@torch.inference_mode()
def get_patch_embeddings(model, paths, device, tfm=None):
    if tfm is None:
        tfm = _build_transform()
    imgs = []
    valid_paths = []

    for p in paths:
        try:
            pil = Image.open(p).convert("RGB")
            imgs.append(tfm(pil))
            valid_paths.append(p)
        except Exception as e:
            print(f"[WARN] failed to load patch: {p} | {e}")

    feat_dim = get_feature_dim()

    if not imgs:
        return torch.empty(0, feat_dim), []

    batch = torch.stack(imgs).to(device, non_blocking=True)

    if device == "cuda":
        batch = batch.half() 

    if USE_INTERMEDIATE_BLOCKS:
        emb = extract_vit_features(
            model=model,
            batch=batch,
            target_block_indices=TARGET_BLOCK_INDICES,
            fusion=BLOCK_FUSION,
            normalize_each_block=NORMALIZE_EACH_BLOCK,
            normalize_final=NORMALIZE_EMBEDDINGS,
        )
    else:
        tokens = model.encoder.forward_features(batch)
        patch_tokens = tokens[:, 1:, :]
        emb = patch_tokens.mean(dim=1)

        if NORMALIZE_EMBEDDINGS:
            emb = F.normalize(emb, dim=1)

    return emb.detach().cpu(), valid_paths

@torch.inference_mode()
def get_patch_embeddings_from_arrays(model, patch_records, device, tfm=None):
    if tfm is None:
        tfm = _build_transform()

    # Detect if model is TRT engine
    if hasattr(model, 'extract'):
        # TRT path: batch all patches together
        imgs = []
        valid_records = []
        for rec in patch_records:
            try:
                rgb = cv2.cvtColor(rec["patch"], cv2.COLOR_GRAY2RGB)
                pil = Image.fromarray(rgb)
                imgs.append(tfm(pil))
                valid_records.append(rec)
            except Exception:
                pass
        if not imgs:
            return torch.empty(0, get_feature_dim()), []
        batch = torch.stack(imgs).cpu()  # keep on CPU for TRT
        # TRT expects float32 input (will convert inside extract)
        embeddings = model.extract(batch)  # returns torch float32 tensor
        return embeddings, valid_records

    # Original PyTorch path
    imgs = []
    valid_records = []
    for rec in patch_records:
        try:
            rgb = cv2.cvtColor(rec["patch"], cv2.COLOR_GRAY2RGB)
            pil = Image.fromarray(rgb)
            imgs.append(tfm(pil))
            valid_records.append(rec)
        except Exception:
            pass

    if not imgs:
        return torch.empty(0, get_feature_dim()), []

    batch = torch.stack(imgs).to(device, non_blocking=True)
    if device == "cuda":
        batch = batch.half()

    if USE_INTERMEDIATE_BLOCKS:
        emb = extract_vit_features(
            model=model,
            batch=batch,
            target_block_indices=TARGET_BLOCK_INDICES,
            fusion=BLOCK_FUSION,
            normalize_each_block=NORMALIZE_EACH_BLOCK,
            normalize_final=NORMALIZE_EMBEDDINGS,
        )
    else:
        tokens = model.encoder.forward_features(batch)
        patch_tokens = tokens[:, 1:, :]
        emb = patch_tokens.mean(dim=1)
        if NORMALIZE_EMBEDDINGS:
            emb = F.normalize(emb, dim=1)

    return emb.detach().cpu(), valid_records

def is_nonblack_patch(path, black_thresh=10, min_nonblack_ratio=0.25):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    nonblack_ratio = float((img > black_thresh).mean())
    return nonblack_ratio >= min_nonblack_ratio

def is_nonblack_patch_array(patch_gray, black_thresh=10, min_nonblack_ratio=0.25):
    if patch_gray is None or patch_gray.size == 0:
        return False
    nonblack = np.count_nonzero(patch_gray > black_thresh)
    ratio = nonblack / float(patch_gray.size)
    return ratio >= min_nonblack_ratio

# =========================================================
# PCA HELPERS
# =========================================================
def collect_embeddings_for_pca(model, patch_dirs, device):
    all_embs = []

    for pdir in patch_dirs:
        all_paths = _list_images(pdir)

        for batch_paths in _batched(all_paths):
            emb, paths = get_patch_embeddings(model, batch_paths, device, tfm=None)

            for i, p in enumerate(paths):
                r, c = parse_rc_from_patch_name(p)
                if r is None or c is None:
                    continue
                if not is_nonblack_patch(p, black_thresh=10, min_nonblack_ratio=0.25):
                    continue
                all_embs.append(emb[i].clone().float())

    if len(all_embs) == 0:
        raise RuntimeError("No valid embeddings found for PCA fitting")

    X = torch.stack(all_embs, dim=0).cpu().numpy().astype(np.float32)
    return X


def fit_global_pca_from_patch_dirs(model, patch_dirs, device, n_components=32):
    X = collect_embeddings_for_pca(model, patch_dirs, device)

    n_components = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components, svd_solver="auto")
    pca.fit(X)

    pca_artifact = {
        "mean": torch.from_numpy(pca.mean_.astype(np.float32)),
        "components": torch.from_numpy(pca.components_.astype(np.float32)),
        "explained_variance": torch.from_numpy(pca.explained_variance_.astype(np.float32)),
        "n_components": int(n_components),
    }
    print(f"[PCA] fitted global PCA with {n_components} components")
    return pca_artifact


def pca_transform_embedding(x, pca_artifact):
    x = x.detach().cpu().float()
    mean = pca_artifact["mean"].float()
    comps = pca_artifact["components"].float()
    z = torch.matmul(comps, (x - mean))
    return z


# =========================================================
# DISTANCE HELPERS
# =========================================================
def mahalanobis_distance(query_emb, stats_obj):
    x = query_emb.detach().cpu().float()
    mu = stats_obj["mean"].detach().cpu().float()
    diff = x - mu

    mode = stats_obj.get("mode", "diag")

    if mode == "diag":
        inv_var = stats_obj["inv_var"].detach().cpu().float()
        dist2 = torch.sum((diff * diff) * inv_var)

    elif mode == "full":
        inv_cov = stats_obj["inv_cov"].detach().cpu().float()
        dist2 = diff.unsqueeze(0) @ inv_cov @ diff.unsqueeze(1)
        dist2 = dist2.squeeze()

    else:
        raise ValueError(f"Unsupported mahalanobis mode: {mode}")

    dist2 = torch.clamp(dist2, min=0.0)
    return float(torch.sqrt(dist2).item())


def nearest_distance_to_bank(query_emb, bank_embs, metric="cosine", mahalanobis_stats=None):
    if metric in ["mahalanobis", "mahalanobis_pca"]:
        if mahalanobis_stats is None:
            return None, None
        best_dist = mahalanobis_distance(query_emb, mahalanobis_stats)
        return None, best_dist

    if bank_embs is None or len(bank_embs) == 0:
        return None, None

    if metric == "cosine":
        q = F.normalize(query_emb.unsqueeze(0), dim=1)[0]
        b = F.normalize(bank_embs, dim=1)
        sims = torch.matmul(b, q)
        best_idx = int(torch.argmax(sims).item())
        best_sim = float(sims[best_idx].item())
        best_dist = float(1.0 - best_sim)
        return best_sim, best_dist

    elif metric == "euclidean":
        dists = torch.norm(bank_embs - query_emb.unsqueeze(0), dim=1)
        best_idx = int(torch.argmin(dists).item())
        best_dist = float(dists[best_idx].item())
        return None, best_dist

    else:
        raise ValueError(f"Unsupported metric: {metric}")


def all_distances_to_bank(query_emb, bank_embs, metric="cosine", mahalanobis_stats=None):
    if metric in ["mahalanobis", "mahalanobis_pca"]:
        if mahalanobis_stats is None:
            return [], [], None, None, None
        dist_val = mahalanobis_distance(query_emb, mahalanobis_stats)
        return None, [dist_val], None, dist_val, 0

    if bank_embs is None or len(bank_embs) == 0:
        return [], [], None, None, None

    if metric == "cosine":
        q = F.normalize(query_emb.unsqueeze(0), dim=1)[0]
        b = F.normalize(bank_embs, dim=1)
        sims = torch.matmul(b, q)
        dists = 1.0 - sims

        sims_list = sims.detach().cpu().numpy().astype(float).tolist()
        dists_list = dists.detach().cpu().numpy().astype(float).tolist()

        best_idx = int(torch.argmax(sims).item())
        best_sim = float(sims[best_idx].item())
        best_dist = float(dists[best_idx].item())
        return sims_list, dists_list, best_sim, best_dist, best_idx

    elif metric == "euclidean":
        dists = torch.norm(bank_embs - query_emb.unsqueeze(0), dim=1)
        dists_list = dists.detach().cpu().numpy().astype(float).tolist()

        best_idx = int(torch.argmin(dists).item())
        best_dist = float(dists[best_idx].item())
        return None, dists_list, None, best_dist, best_idx

    else:
        raise ValueError(f"Unsupported metric: {metric}")

# =========================================================
# EMBEDDING BANKS / MAHALANOBIS STATS
# =========================================================

def build_embedding_bank_from_patch_dirs(model, patch_dirs, device, return_meta=False):
    bank_lists = defaultdict(list)
    meta_lists = defaultdict(list)

    for pdir in patch_dirs:
        all_paths = _list_images(pdir)

        for batch_paths in _batched(all_paths):
            emb, paths = get_patch_embeddings(model, batch_paths, device, tfm=None)

            for i, p in enumerate(paths):
                r, c = parse_rc_from_patch_name(p)
                if r is None or c is None:
                    continue

                if not is_nonblack_patch(p, black_thresh=10, min_nonblack_ratio=0.25):
                    continue

                key = (r, c)
                vec = emb[i]
                bank_lists[key].append(vec.clone())

                meta_lists[key].append({
                    "source_patch_path": p,
                    "source_group": str(Path(pdir).parent.name),
                    "is_augmented": "augmented_crops" in p.replace("\\", "/"),
                })

    reference_bank = {}
    reference_bank_meta = {}

    for key, vec_list in bank_lists.items():
        if len(vec_list) == 0:
            continue
        reference_bank[key] = torch.stack(vec_list, dim=0)
        reference_bank_meta[key] = meta_lists[key]

    print(f"[BANK] built for {len(reference_bank)} RC locations")

    if return_meta:
        return reference_bank, reference_bank_meta
    return reference_bank

def build_mahalanobis_stats_from_patch_dirs(
    model,
    patch_dirs,
    device,
    mode="diag",
    reg_eps=1e-3,
    min_samples=3,
    pca_artifact=None,
):
    emb_lists = defaultdict(list)

    for pdir in patch_dirs:
        all_paths = _list_images(pdir)

        for batch_paths in _batched(all_paths):
            emb, paths = get_patch_embeddings(model, batch_paths, device, tfm=None)

            for i, p in enumerate(paths):
                r, c = parse_rc_from_patch_name(p)
                if r is None or c is None:
                    continue

                if not is_nonblack_patch(p, black_thresh=10, min_nonblack_ratio=0.25):
                    continue

                vec = emb[i].clone().float()
                if pca_artifact is not None:
                    vec = pca_transform_embedding(vec, pca_artifact)

                emb_lists[(r, c)].append(vec)

    mahalanobis_stats = {}

    for key, vec_list in emb_lists.items():
        if len(vec_list) < min_samples:
            continue

        X = torch.stack(vec_list, dim=0).float()
        mu = X.mean(dim=0)
        xc = X - mu
        n, d = X.shape

        if mode == "diag":
            var = torch.mean(xc * xc, dim=0) + reg_eps
            inv_var = 1.0 / var

            mahalanobis_stats[key] = {
                "mode": "diag",
                "mean": mu.cpu(),
                "inv_var": inv_var.cpu(),
                "num_samples": int(n),
            }

        elif mode == "full":
            cov = (xc.T @ xc) / max(n - 1, 1)
            cov = cov + reg_eps * torch.eye(d, dtype=cov.dtype, device=cov.device)
            inv_cov = torch.linalg.pinv(cov)

            mahalanobis_stats[key] = {
                "mode": "full",
                "mean": mu.cpu(),
                "inv_cov": inv_cov.cpu(),
                "num_samples": int(n),
            }

        else:
            raise ValueError(f"Unsupported Mahalanobis mode: {mode}")

    print(f"[MAHAL] built stats for {len(mahalanobis_stats)} RC locations")
    return mahalanobis_stats

def build_mahalanobis_stats_from_vectors(
    vec_list,
    mode="diag",
    reg_eps=1e-3,
    min_samples=3,
):
    if len(vec_list) < min_samples:
        return None

    X = torch.stack([v.clone().float() for v in vec_list], dim=0)
    mu = X.mean(dim=0)
    xc = X - mu
    n, d = X.shape

    if mode == "diag":
        var = torch.mean(xc * xc, dim=0) + reg_eps
        inv_var = 1.0 / var
        return {
            "mode": "diag",
            "mean": mu.cpu(),
            "inv_var": inv_var.cpu(),
            "num_samples": int(n),
        }

    elif mode == "full":
        cov = (xc.T @ xc) / max(n - 1, 1)
        cov = cov + reg_eps * torch.eye(d, dtype=cov.dtype, device=cov.device)
        inv_cov = torch.linalg.pinv(cov)
        return {
            "mode": "full",
            "mean": mu.cpu(),
            "inv_cov": inv_cov.cpu(),
            "num_samples": int(n),
        }

    else:
        raise ValueError(f"Unsupported Mahalanobis mode: {mode}")
    
def build_image_patch_feature_dict(model, patch_dirs, device, pca_artifact=None):
    """
    Returns:
        image_patch_features[image_name][(r,c)] = {
            "feature": tensor[D],
            "patch_path": path,
        }
    """
    image_patch_features = defaultdict(dict)

    for pdir in patch_dirs:
        image_name = Path(pdir).parent.name
        all_paths = _list_images(pdir)

        for batch_paths in _batched(all_paths):
            emb, paths = get_patch_embeddings(model, batch_paths, device, tfm=None)

            for i, p in enumerate(paths):
                r, c = parse_rc_from_patch_name(p)
                if r is None or c is None:
                    continue

                if not is_nonblack_patch(p, black_thresh=10, min_nonblack_ratio=0.25):
                    continue

                vec = emb[i].clone().float()
                if pca_artifact is not None:
                    vec = pca_transform_embedding(vec, pca_artifact)

                image_patch_features[image_name][(r, c)] = {
                    "feature": vec,
                    "patch_path": p,
                }

    return image_patch_features


def collect_good_distances_by_rc_leave_one_out(
    image_patch_features,
    metric="mahalanobis_pca",
    mahalanobis_mode="diag",
    reg_eps=1e-3,
    min_samples=3,
):
    """
    Leave-one-out calibration:
      for each good image patch at (r,c), compare against model built from all OTHER good images at same (r,c)

    Returns:
      dist_by_rc, dist_by_col, dist_by_row, all_distances, rc_rows
    """
    dist_by_rc = defaultdict(list)
    dist_by_col = defaultdict(list)
    dist_by_row = defaultdict(list)
    all_distances = []
    rc_rows = []

    image_names = sorted(list(image_patch_features.keys()))
    all_keys = set()
    for img_name in image_names:
        all_keys.update(image_patch_features[img_name].keys())

    for (r, c) in sorted(all_keys):
        available_imgs = [img for img in image_names if (r, c) in image_patch_features[img]]

        for anchor_img in available_imgs:
            query_vec = image_patch_features[anchor_img][(r, c)]["feature"]
            patch_path = image_patch_features[anchor_img][(r, c)]["patch_path"]

            ref_vecs = [
                image_patch_features[other_img][(r, c)]["feature"]
                for other_img in available_imgs
                if other_img != anchor_img
            ]

            if len(ref_vecs) < min_samples:
                continue

            if metric in ["mahalanobis", "mahalanobis_pca"]:
                stats_obj = build_mahalanobis_stats_from_vectors(
                    ref_vecs,
                    mode=mahalanobis_mode,
                    reg_eps=reg_eps,
                    min_samples=min_samples,
                )
                if stats_obj is None:
                    continue

                _, best_dist = nearest_distance_to_bank(
                    query_emb=query_vec,
                    bank_embs=None,
                    metric=metric,
                    mahalanobis_stats=stats_obj,
                )

            else:
                ref_bank = torch.stack(ref_vecs, dim=0)
                _, best_dist = nearest_distance_to_bank(
                    query_emb=query_vec,
                    bank_embs=ref_bank,
                    metric=metric,
                )

            if best_dist is None:
                continue

            best_dist = float(best_dist)
            key = (r, c)

            dist_by_rc[key].append(best_dist)
            dist_by_col[c].append(best_dist)
            dist_by_row[r].append(best_dist)
            all_distances.append(best_dist)

            rc_rows.append({
                "r": int(r),
                "c": int(c),
                "patch_path": patch_path,
                "image_group": anchor_img,
                "metric": metric,
                "distance": best_dist,
            })

    return dist_by_rc, dist_by_col, dist_by_row, all_distances, rc_rows


def _robust_stats(vals, sigma_floor=0.01, mad_floor=0.01):
    vals = np.asarray(vals, dtype=np.float32)
    mu = float(np.mean(vals))
    sigma = max(float(np.std(vals)), sigma_floor)

    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    mad = max(mad, mad_floor)
    mad_sigma = 1.4826 * mad

    return mu, sigma, med, mad, mad_sigma

def _robust_threshold(vals, percentile=99.0, k_sigma=4.0, k_mad=4.0, sigma_floor=0.01, mad_floor=0.01):
    vals = np.asarray(vals, dtype=np.float32)
    pct = float(np.percentile(vals, percentile))
    mu, sigma, med, mad, mad_sigma = _robust_stats(vals, sigma_floor=sigma_floor, mad_floor=mad_floor)

    thr_sigma = mu + k_sigma * sigma
    thr_mad = med + k_mad * mad_sigma

    thr = max(pct, thr_sigma, thr_mad)
    return thr, mu, sigma, med, mad

def remove_one_top_outlier(vals, ratio=1.8):
    """
    Remove only one top outlier if it is clearly separated from the second-largest value.
    Returns cleaned numpy array and a flag.
    """
    vals = np.asarray(vals, dtype=np.float32)

    if len(vals) < 4:
        return vals, False

    vals_sorted = np.sort(vals)

    largest = float(vals_sorted[-1])
    second_largest = float(vals_sorted[-2])

    if second_largest <= 0:
        return vals_sorted, False

    if largest > ratio * second_largest:
        return vals_sorted[:-1], True

    return vals_sorted, False

def collect_good_distances_by_rc(
    model,
    patch_dirs,
    reference_bank,
    device,
    mahalanobis_stats=None,
    pca_artifact=None,
):
    dist_by_rc = defaultdict(list)
    dist_by_col = defaultdict(list)
    dist_by_row = defaultdict(list)
    all_distances = []
    rc_rows = []

    for pdir in patch_dirs:
        image_group = str(Path(pdir).parent.name)
        all_paths = _list_images(pdir)

        for batch_paths in _batched(all_paths):
            emb, paths = get_patch_embeddings(model, batch_paths, device, tfm=None)

            for i, p in enumerate(paths):
                r, c = parse_rc_from_patch_name(p)
                if r is None or c is None:
                    continue

                if not is_nonblack_patch(p, black_thresh=10, min_nonblack_ratio=0.25):
                    continue

                key = (r, c)
                query_vec = emb[i]

                if DISTANCE_METRIC == "mahalanobis_pca":
                    query_vec = pca_transform_embedding(query_vec, pca_artifact)

                if DISTANCE_METRIC in ["mahalanobis", "mahalanobis_pca"]:
                    if mahalanobis_stats is None or key not in mahalanobis_stats:
                        continue

                    best_sim, best_dist = nearest_distance_to_bank(
                        query_emb=query_vec,
                        bank_embs=None,
                        metric=DISTANCE_METRIC,
                        mahalanobis_stats=mahalanobis_stats[key],
                    )
                else:
                    if key not in reference_bank:
                        continue

                    best_sim, best_dist = nearest_distance_to_bank(
                        query_emb=query_vec,
                        bank_embs=reference_bank[key],
                        metric=DISTANCE_METRIC,
                    )

                if best_dist is None:
                    continue

                best_dist = float(best_dist)

                dist_by_rc[key].append(best_dist)
                dist_by_col[c].append(best_dist)
                dist_by_row[r].append(best_dist)
                all_distances.append(best_dist)

                rc_rows.append({
                    "r": int(r),
                    "c": int(c),
                    "patch_path": p,
                    "image_group": image_group,
                    "metric": DISTANCE_METRIC,
                    "distance": best_dist,
                })

    return dist_by_rc, dist_by_col, dist_by_row, all_distances, rc_rows

def collect_all_pairwise_patch_distances(model, patch_dirs, device):
    image_patch_embs = defaultdict(dict)
    image_names = []

    for pdir in patch_dirs:
        image_name = Path(pdir).parent.name
        image_names.append(image_name)

        all_paths = _list_images(pdir)

        for batch_paths in _batched(all_paths):
            emb, paths = get_patch_embeddings(model, batch_paths, device, tfm=None)

            for i, p in enumerate(paths):
                r, c = parse_rc_from_patch_name(p)
                if r is None or c is None:
                    continue

                    # keep only informative patches
                if not is_nonblack_patch(p, black_thresh=10, min_nonblack_ratio=0.25):
                    continue

                image_patch_embs[image_name][(r, c)] = {
                    "embedding": emb[i].clone(),
                    "patch_path": p,
                }

    image_names = sorted(list(set(image_names)))

    all_keys = set()
    for img_name in image_names:
        all_keys.update(image_patch_embs[img_name].keys())

    pairwise_rows = []

    for (r, c) in sorted(all_keys):
        available_imgs = [img for img in image_names if (r, c) in image_patch_embs[img]]

        for anchor_img in available_imgs:
            q_emb = image_patch_embs[anchor_img][(r, c)]["embedding"]

            for compare_img in available_imgs:
                if anchor_img == compare_img:
                    continue

                ref_emb = image_patch_embs[compare_img][(r, c)]["embedding"]

                if DISTANCE_METRIC == "cosine":
                    qn = F.normalize(q_emb.unsqueeze(0), dim=1)[0]
                    rn = F.normalize(ref_emb.unsqueeze(0), dim=1)[0]
                    sim_val = float(torch.sum(qn * rn).item())
                    dist_val = float(max(0.0, 1.0 - sim_val))

                    pairwise_rows.append({
                        "r": int(r),
                        "c": int(c),
                        "anchor_image": anchor_img,
                        "compare_image": compare_img,
                        "metric": "cosine",
                        "sim": sim_val,
                        "dist": dist_val,
                    })

                elif DISTANCE_METRIC == "euclidean":
                    dist_val = float(torch.norm(q_emb - ref_emb, p=2).item())

                    pairwise_rows.append({
                        "r": int(r),
                        "c": int(c),
                        "anchor_image": anchor_img,
                        "compare_image": compare_img,
                        "metric": "euclidean",
                        "sim": None,
                        "dist": dist_val,
                    })

                else:
                    raise ValueError(f"Unsupported metric for pairwise export: {DISTANCE_METRIC}")

    return pairwise_rows

# =========================================================
# THRESHOLD BUILDING
# =========================================================

def build_patchwise_thresholds_simple(
    dist_by_rc,
    local_percentile=95.0,
    remove_top_outlier=True,
    outlier_ratio=1.8,
):
    thresholds_by_rc = {}
    mu_by_rc = {}
    sigma_by_rc = {}

    cleaned_dist_by_rc = {}
    local_debug_rows = []
    all_cleaned_distances = []

    for key, vals in dist_by_rc.items():
        vals = np.asarray(vals, dtype=np.float32)
        if len(vals) == 0:
            continue

        raw_count = int(len(vals))

        if remove_top_outlier:
            cleaned_vals, removed_flag = remove_one_top_outlier(vals, ratio=outlier_ratio)
        else:
            cleaned_vals, removed_flag = vals, False

        if len(cleaned_vals) == 0:
            continue

        thr = float(np.percentile(cleaned_vals, local_percentile))
        mu = float(np.mean(cleaned_vals))
        sigma = float(np.std(cleaned_vals))
        sigma = max(sigma, SIGMA_FLOOR)

        thresholds_by_rc[key] = thr
        mu_by_rc[key] = mu
        sigma_by_rc[key] = sigma
        cleaned_dist_by_rc[key] = cleaned_vals.tolist()
        all_cleaned_distances.extend(cleaned_vals.tolist())

        local_debug_rows.append({
            "r": int(key[0]),
            "c": int(key[1]),
            "raw_count": raw_count,
            "cleaned_count": int(len(cleaned_vals)),
            "outlier_removed": bool(removed_flag),
            "raw_min": float(np.min(vals)),
            "raw_max": float(np.max(vals)),
            "cleaned_min": float(np.min(cleaned_vals)),
            "cleaned_max": float(np.max(cleaned_vals)),
            "local_threshold": float(thr),
            "mu_cleaned": float(mu),
            "sigma_cleaned": float(sigma),
        })

    return (
        thresholds_by_rc,
        mu_by_rc,
        sigma_by_rc,
        cleaned_dist_by_rc,
        local_debug_rows,
    )

# =========================================================
# INFERENCE ON PATCHES
# =========================================================

@torch.inference_mode()
def infer_patches_generic(
    model,
    patches_dir,
    reference_bank,
    reference_bank_meta,
    thresholds_by_rc,
    mu_by_rc,
    sigma_by_rc,
    mahalanobis_stats,
    pca_artifact,
    save_dir,
    device,
    patch_transform=None,
):
    os.makedirs(save_dir, exist_ok=True)

    all_paths = _list_images(patches_dir)
    rows = []
    patch_records = []
    raw_compare_rows = []

    total = 0
    paired = 0
    skipped_no_rc = 0
    skipped_no_ref = 0
    skipped_black_bg = 0
    skipped_no_threshold = 0

    nonblack_paths = [p for p in all_paths if is_nonblack_patch(p)]

    for batch_paths in _batched(all_paths):
        emb, paths = get_patch_embeddings(model, batch_paths, device, tfm=patch_transform)

        for i, p in enumerate(paths):
            total += 1
            fname = os.path.basename(p)
            r, c = parse_rc_from_patch_name(fname)

            if r is None or c is None:
                skipped_no_rc += 1
                rows.append({
                    "filename": fname,
                    "full_path": p,
                    "r": None,
                    "c": None,
                    "metric": DISTANCE_METRIC,
                    "similarity": None,
                    "distance": None,
                    "all_similarities": None,
                    "all_distances": None,
                    "num_ref_patches": 0,
                    "best_match_index": None,
                    "best_match_patch_path": None,
                    "best_match_group": None,
                    "best_match_is_augmented": None,
                    "mahalanobis_mode": None,
                    "mahalanobis_num_samples": None,
                    "ang_dist_rad": None,
                    "local_threshold_used": None,
                    "threshold_source": None,
                    "threshold_used": None,
                    "mu_used": None,
                    "sigma_used": None,
                    "z_score": None,
                    "classification": "SKIP_NO_RC",
                })
                continue

            key = (r, c)

            if DISTANCE_METRIC in ["mahalanobis", "mahalanobis_pca"]:
                has_ref = mahalanobis_stats is not None and key in mahalanobis_stats
            else:
                has_ref = reference_bank is not None and key in reference_bank

            if not has_ref:
                skipped_no_ref += 1
                rows.append({
                    "filename": fname,
                    "full_path": p,
                    "r": r,
                    "c": c,
                    "metric": DISTANCE_METRIC,
                    "similarity": None,
                    "distance": None,
                    "all_similarities": None,
                    "all_distances": None,
                    "num_ref_patches": 0,
                    "best_match_index": None,
                    "best_match_patch_path": None,
                    "best_match_group": None,
                    "best_match_is_augmented": None,
                    "mahalanobis_mode": None,
                    "mahalanobis_num_samples": None,
                    "ang_dist_rad": None,
                    "local_threshold_used": None,
                    "threshold_source": None,
                    "threshold_used": None,
                    "mu_used": None,
                    "sigma_used": None,
                    "z_score": None,
                    "classification": "SKIP_NO_REF",
                })
                continue

            ref_meta_list = reference_bank_meta.get(key, []) if reference_bank_meta is not None else []
            mahal_stats = mahalanobis_stats.get(key) if mahalanobis_stats is not None else None

            query_vec = emb[i]
            if DISTANCE_METRIC == "mahalanobis_pca":
                query_vec = pca_transform_embedding(query_vec, pca_artifact)

            all_sims, all_dists, best_sim, best_dist, best_idx = all_distances_to_bank(
                query_emb=query_vec,
                bank_embs=None if DISTANCE_METRIC in ["mahalanobis", "mahalanobis_pca"] else reference_bank[key],
                metric=DISTANCE_METRIC,
                mahalanobis_stats=mahal_stats,
            )

            if DISTANCE_METRIC in ["mahalanobis", "mahalanobis_pca"]:
                raw_compare_rows.append({
                    "filename": fname,
                    "full_path": p,
                    "r": int(r),
                    "c": int(c),
                    "reference_index": 0,
                    "reference_patch_path": None,
                    "reference_group": "mahalanobis_distribution",
                    "reference_is_augmented": None,
                    "metric": DISTANCE_METRIC,
                    "similarity": None,
                    "distance": float(best_dist) if best_dist is not None else None,
                    "mahalanobis_mode": mahal_stats.get("mode") if mahal_stats else None,
                    "mahalanobis_num_samples": int(mahal_stats.get("num_samples", 0)) if mahal_stats else 0,
                })
            else:
                if DISTANCE_METRIC == "cosine":
                    iter_pairs = zip(all_sims, all_dists)
                else:
                    iter_pairs = zip([None] * len(all_dists), all_dists)

                for ref_idx, (sim_val, dist_val) in enumerate(iter_pairs):
                    meta = ref_meta_list[ref_idx] if ref_idx < len(ref_meta_list) else {}

                    raw_compare_rows.append({
                        "filename": fname,
                        "full_path": p,
                        "r": int(r),
                        "c": int(c),
                        "reference_index": int(ref_idx),
                        "reference_patch_path": meta.get("source_patch_path"),
                        "reference_group": meta.get("source_group"),
                        "reference_is_augmented": meta.get("is_augmented"),
                        "metric": DISTANCE_METRIC,
                        "similarity": float(sim_val) if sim_val is not None else None,
                        "distance": float(dist_val),
                    })

            if best_dist is None:
                skipped_no_ref += 1
                rows.append({
                    "filename": fname,
                    "full_path": p,
                    "r": r,
                    "c": c,
                    "metric": DISTANCE_METRIC,
                    "similarity": None,
                    "distance": None,
                    "all_similarities": None,
                    "all_distances": None,
                    "num_ref_patches": 0,
                    "best_match_index": None,
                    "best_match_patch_path": None,
                    "best_match_group": None,
                    "best_match_is_augmented": None,
                    "mahalanobis_mode": None,
                    "mahalanobis_num_samples": None,
                    "ang_dist_rad": None,
                    "local_threshold_used": None,
                    "column_threshold_used": None,
                    "row_threshold_used": None,
                    "threshold_source": None,
                    "threshold_used": None,
                    "mu_used": None,
                    "sigma_used": None,
                    "z_score": None,
                    "classification": "SKIP_NO_REF",
                })
                continue

            if DISTANCE_METRIC == "cosine" and best_sim is not None:
                sim_clip = float(np.clip(best_sim, -1.0, 1.0))
                ang_dist = float(np.arccos(sim_clip))
            else:
                ang_dist = None

            threshold_info = choose_threshold_and_stats(
                key=key,
                thresholds_by_rc=thresholds_by_rc,
                mu_by_rc=mu_by_rc,
                sigma_by_rc=sigma_by_rc,
            )

            if threshold_info is None:
                skipped_no_threshold += 1
                rows.append({
                    "filename": fname,
                    "full_path": p,
                    "r": r,
                    "c": c,
                    "metric": DISTANCE_METRIC,
                    "similarity": float(best_sim) if best_sim is not None else None,
                    "distance": float(best_dist),
                    "all_similarities": ",".join(f"{x:.8f}" for x in all_sims) if all_sims is not None else None,
                    "all_distances": ",".join(f"{x:.8f}" for x in all_dists) if all_dists is not None else None,
                    "num_ref_patches": int(len(all_dists)) if all_dists is not None else 0,
                    "best_match_index": int(best_idx) if best_idx is not None else None,
                    "best_match_patch_path": None,
                    "best_match_group": None,
                    "best_match_is_augmented": None,
                    "mahalanobis_mode": mahal_stats.get("mode") if mahal_stats is not None else None,
                    "mahalanobis_num_samples": int(mahal_stats.get("num_samples", 0)) if mahal_stats is not None else None,
                    "ang_dist_rad": float(ang_dist) if ang_dist is not None else None,
                    "local_threshold_used": None,
                    "threshold_source": None,
                    "threshold_used": None,
                    "mu_used": None,
                    "sigma_used": None,
                    "z_score": None,
                    "classification": "SKIP_NO_THRESHOLD",
                })
                continue

            thr = float(threshold_info["threshold_used"])
            mu = float(threshold_info["mu_used"])
            sigma_eff = max(float(threshold_info["sigma_used"]), SIGMA_FLOOR)

            
            is_defect = float(best_dist) > float(thr)
            z_score = (float(best_dist) - mu) / sigma_eff

            paired += 1

            if DISTANCE_METRIC in ["mahalanobis", "mahalanobis_pca"]:
                num_ref_patches = int(mahal_stats.get("num_samples", 0)) if mahal_stats is not None else 0
                best_match_index = None
                best_match_patch_path = None
                best_match_group = None
                best_match_is_augmented = None
            else:
                best_ref_meta = ref_meta_list[best_idx] if (best_idx is not None and best_idx < len(ref_meta_list)) else {}
                num_ref_patches = int(len(all_dists))
                best_match_index = int(best_idx) if best_idx is not None else None
                best_match_patch_path = best_ref_meta.get("source_patch_path")
                best_match_group = best_ref_meta.get("source_group")
                best_match_is_augmented = best_ref_meta.get("is_augmented")

            row = {
                "filename": fname,
                "full_path": p,
                "r": r,
                "c": c,
                "metric": DISTANCE_METRIC,
                "similarity": float(best_sim) if best_sim is not None else None,
                "distance": float(best_dist),
                "all_similarities": ",".join(f"{x:.8f}" for x in all_sims) if all_sims is not None else None,
                "all_distances": ",".join(f"{x:.8f}" for x in all_dists) if all_dists is not None else None,
                "num_ref_patches": num_ref_patches,
                "best_match_index": best_match_index,
                "best_match_patch_path": best_match_patch_path,
                "best_match_group": best_match_group,
                "best_match_is_augmented": best_match_is_augmented,
                "mahalanobis_mode": mahal_stats.get("mode") if mahal_stats is not None else None,
                "mahalanobis_num_samples": int(mahal_stats.get("num_samples", 0)) if mahal_stats is not None else None,
                "ang_dist_rad": float(ang_dist) if ang_dist is not None else None,
                "local_threshold_used": threshold_info["local_threshold_used"],
                "threshold_source": threshold_info["threshold_source"],
                "threshold_used": float(thr),
                "mu_used": float(mu),
                "sigma_used": float(sigma_eff),
                "z_score": float(z_score),
                "classification": "DEFECT" if is_defect else "GOOD",
            }
            rows.append(row)
            patch_records.append(row)

    print(
        f"[{DISTANCE_METRIC.upper()}] total={total} | paired={paired} | "
        f"skip_no_rc={skipped_no_rc} | skip_no_ref={skipped_no_ref} | "
        f"skip_black_bg={skipped_black_bg} | skip_no_threshold={skipped_no_threshold}"
    )

    df = pd.DataFrame(rows)
    out_csv = os.path.join(save_dir, "patch_distance_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"[SAVE] {out_csv}")

    raw_compare_csv = os.path.join(save_dir, "patch_all_reference_distances.csv")
    pd.DataFrame(raw_compare_rows).to_csv(raw_compare_csv, index=False)
    print(f"[SAVE] {raw_compare_csv}")

    stitched_template_path = None
    if patch_records:
        sample = cv2.imread(patch_records[0]["full_path"])
        if sample is not None:
            ph, pw = sample.shape[:2]
            max_r = max(int(x["r"]) for x in patch_records if x["r"] is not None)
            max_c = max(int(x["c"]) for x in patch_records if x["c"] is not None)

            canvas = np.zeros(((max_r + 1) * ph, (max_c + 1) * pw, 3), dtype=np.uint8)

            for rec in patch_records:
                patch = cv2.imread(rec["full_path"])
                if patch is None:
                    continue

                    # stitch in RC layout
                y0 = int(rec["r"]) * ph
                x0 = int(rec["c"]) * pw
                canvas[y0:y0 + ph, x0:x0 + pw] = patch

                if rec["classification"] == "DEFECT":
                    color = (0, 0, 255)
                    cv2.rectangle(canvas, (x0, y0), (x0 + pw, y0 + ph), color, 2)
                    cv2.putText(
                        canvas,
                        f"{rec['distance']:.2f}",
                        (x0 + 5, y0 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        1,
                        cv2.LINE_AA,
                    )

            stitched_template_path = os.path.join(save_dir, "template_stitched.png")
            cv2.imwrite(stitched_template_path, canvas)
            print(f"[SAVE] {stitched_template_path}")

    return df, stitched_template_path

# =========================================================
# HEATMAP / BBOX HELPERS
# =========================================================
def compute_topk_image_score_from_df(vit_df, k=5):
    valid = vit_df[vit_df["classification"].isin(["GOOD", "DEFECT"])].copy()
    if valid.empty:
        return None

    dists = valid["distance"].dropna().astype(float).values
    if len(dists) == 0:
        return None

    dists_sorted = np.sort(dists)[::-1]
    topk = dists_sorted[:min(k, len(dists_sorted))]
    return float(np.mean(topk))

# ===========================================================
# INNERWALL DIMENSIONS
# ===========================================================

def normalize_tyre_name_for_dimensions(tyre_name):
    if tyre_name is None:
        return None

    norm = re.sub(r"[^0-9R]", "", str(tyre_name).upper())
    return norm or None

def tyre_bboxes(img_path):
    """
    Find outer tyre bounding box from thresholded image.
    Returns x, y, w, h, area
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    img_blur = cv2.medianBlur(img, 5)
    _, th1 = cv2.threshold(img_blur, 15, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError(f"No contours found in image: {img_path}")

    cnt = sorted(contours, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(cnt[0])
    area = w * h

    print(f"[TYRE BOX] {os.path.basename(img_path)} -> x:{x}, y:{y}, w:{w}, h:{h}, area:{area}")
    return x, y, w, h, area

def save_tyre_bbox_debug_image(img_path, save_path, bbox):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[TYRE BOX][WARN] could not read for debug save: {img_path}")
        return

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    x, y, w, h, area = bbox
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(
        vis,
        f"x={x}, y={y}, w={w}, h={h}, area={area}",
        (10, max(25, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(save_path, vis)
    print(f"[TYRE BOX][SAVE] {save_path}")

def enrich_innerwall_yolo_df_with_dimensions(seg_df, crop_path, tyre_name, debug_save_dir=None):
    dim_summary = {
        "tyre_name": tyre_name,
        "tyre_name_normalized": None,
        "dimensioned_detections": 0,
        "max_defect_height_mm": None,
        "max_defect_width_mm": None,
        "max_defect_area_mm2": None,
        "sum_defect_area_mm2": None,
    }

    if seg_df is None or len(seg_df) == 0:
        return seg_df, dim_summary

    if crop_path is None or not os.path.isfile(crop_path):
        print("[DIM][WARN] crop_path missing, skipping innerwall dimension calculation")
        return seg_df, dim_summary

    tyre_name_normalized = normalize_tyre_name_for_dimensions(tyre_name)
    dim_summary["tyre_name_normalized"] = tyre_name_normalized

    if not tyre_name_normalized:
        print("[DIM][WARN] tyre_name missing, skipping innerwall dimension calculation")
        return seg_df, dim_summary

    try:
        innerwall_width_mm, innerwall_height_mm, innerwall_area_mm2 = innerwall_dimensions(tyre_name_normalized)
        dimension_basis = "innerwall_dimensions"
    except Exception as e:
        print(f"[DIM][WARN] failed parsing tyre_name={tyre_name}: {e}")
        return seg_df, dim_summary

    crop_gray = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)
    if crop_gray is None:
        print(f"[DIM][WARN] failed to read crop_path={crop_path}")
        return seg_df, dim_summary

    crop_h, crop_w = int(crop_gray.shape[0]), int(crop_gray.shape[1])

    try:
        x, y, w, h, area = tyre_bboxes(crop_path)
        tyre_box = {"x": x, "y": y, "w": w, "h": h, "area": area}
        base_w = max(int(w), 1)
        base_h = max(int(h), 1)
        total_pix = max(int(area), 1)
        dimension_basis = "tyre_bbox"
    except Exception as e:
        print(f"[DIM][WARN] tyre bbox not found, using full crop fallback | {e}")
        base_w = max(crop_w, 1)
        base_h = max(crop_h, 1)
        total_pix = max(int(crop_h * crop_w), 1)
        dimension_basis = "full_crop_fallback"
        tyre_box = {"x": 0, "y": 0, "w": crop_w, "h": crop_h, "area": total_pix}

    if debug_save_dir is not None:
        os.makedirs(debug_save_dir, exist_ok=True)
        debug_path = os.path.join(debug_save_dir, "tyre_bbox_debug.png")
        save_tyre_bbox_debug_image(
            crop_path,
            debug_path,
            (tyre_box["x"], tyre_box["y"], tyre_box["w"], tyre_box["h"], tyre_box["area"]),
        )

    out_df = seg_df.copy()
    out_df["tyre_name"] = str(tyre_name)
    out_df["tyre_name_normalized"] = str(tyre_name_normalized)

    out_df["crop_width_px"] = crop_w
    out_df["crop_height_px"] = crop_h
    out_df["crop_area_px"] = int(crop_h * crop_w)

    out_df["dimension_basis"] = dimension_basis
    out_df["tyre_bbox_x_px"] = int(tyre_box["x"])
    out_df["tyre_bbox_y_px"] = int(tyre_box["y"])
    out_df["tyre_bbox_width_px"] = int(tyre_box["w"])
    out_df["tyre_bbox_height_px"] = int(tyre_box["h"])
    out_df["tyre_bbox_area_px"] = int(tyre_box["area"])

    out_df["innerwall_width_mm"] = float(innerwall_width_mm)
    out_df["innerwall_height_mm"] = float(innerwall_height_mm)
    out_df["innerwall_area_mm2"] = float(innerwall_area_mm2)

    out_df["bbox_width_px"] = (out_df["bbox_x2_px"] - out_df["bbox_x1_px"]).clip(lower=0).astype(float)
    out_df["bbox_height_px"] = (out_df["bbox_y2_px"] - out_df["bbox_y1_px"]).clip(lower=0).astype(float)
    out_df["bbox_area_px"] = (out_df["bbox_width_px"] * out_df["bbox_height_px"]).astype(float)

    out_df["global_bbox_x1_px"] = (out_df["c"].astype(float) * BIG_PATCH_W) + out_df["bbox_x1_px"].astype(float)
    out_df["global_bbox_y1_px"] = (out_df["r"].astype(float) * BIG_PATCH_H) + out_df["bbox_y1_px"].astype(float)
    out_df["global_bbox_x2_px"] = (out_df["c"].astype(float) * BIG_PATCH_W) + out_df["bbox_x2_px"].astype(float)
    out_df["global_bbox_y2_px"] = (out_df["r"].astype(float) * BIG_PATCH_H) + out_df["bbox_y2_px"].astype(float)

    defect_height_mm = []
    defect_width_mm = []
    defect_diag_mm = []
    defect_area_mm2 = []

    for _, row in out_df.iterrows():
        rdlen_mm, rdwid_mm = cor_innerwall(
            iwid=base_w,
            ilen=base_h,
            dwid=float(row["bbox_width_px"]),
            dlen=float(row["bbox_height_px"]),
            innerwallHeight=innerwall_height_mm,
            innerwallWidth=innerwall_width_mm,
        )

        area_mm2 = area_defect_innerwall(
            t_pix=total_pix,
            d_pix=float(row["bbox_area_px"]),
            areaOfInnerwall=innerwall_area_mm2,
        )

        defect_height_mm.append(float(rdlen_mm))
        defect_width_mm.append(float(rdwid_mm))
        defect_diag_mm.append(float(np.hypot(rdlen_mm, rdwid_mm)))
        defect_area_mm2.append(float(area_mm2))

    out_df["defect_height_mm"] = np.round(defect_height_mm, DEFECT_DIMENSION_DECIMALS)
    out_df["defect_width_mm"] = np.round(defect_width_mm, DEFECT_DIMENSION_DECIMALS)
    out_df["defect_diagonal_mm"] = np.round(defect_diag_mm, DEFECT_DIMENSION_DECIMALS)
    out_df["defect_area_mm2"] = np.round(defect_area_mm2, DEFECT_DIMENSION_DECIMALS)

    dim_summary.update({
        "dimensioned_detections": int(len(out_df)),
        "max_defect_height_mm": float(out_df["defect_height_mm"].max()) if len(out_df) else None,
        "max_defect_width_mm": float(out_df["defect_width_mm"].max()) if len(out_df) else None,
        "max_defect_area_mm2": float(out_df["defect_area_mm2"].max()) if len(out_df) else None,
        "sum_defect_area_mm2": float(np.round(out_df["defect_area_mm2"].sum(), DEFECT_DIMENSION_DECIMALS)) if len(out_df) else None,
    })

    print(
        f"[DIM] tyre={tyre_name} | normalized={tyre_name_normalized} | "
        f"basis={dimension_basis} | detections={dim_summary['dimensioned_detections']} | "
        f"max_h_mm={dim_summary['max_defect_height_mm']} | "
        f"max_w_mm={dim_summary['max_defect_width_mm']} | "
        f"max_area_mm2={dim_summary['max_defect_area_mm2']}"
    )

    return out_df, dim_summary
# =========================================================
# YOLO ONLY ON VIT DEFECT PATCHES
# =========================================================
def run_yolo_on_vit_defect_patches(
    vit_df,
    save_dir,
    seg_models,                 # now a dict
    conf_threshold=SEG_CONF_THRESHOLD,
    crop_path=None,
    tyre_name=None,
):
    os.makedirs(save_dir, exist_ok=True)
    dim_summary = {
        "dimensioned_detections": 0,
        "max_defect_height_mm": None,
        "max_defect_width_mm": None,
        "max_defect_area_mm2": None,
        "sum_defect_area_mm2": None,
    }

    if not seg_models:
        print("[YOLO] no segmentation models provided, skipping")
        return pd.DataFrame(), None, dim_summary

    # Filter defect patches with high confidence (optional performance filter)
    defect_df = vit_df[vit_df["classification"] == "DEFECT"].copy() 

    if defect_df.empty:
        print("[YOLO] No ViT defect patches after filtering")
        return pd.DataFrame(), None, dim_summary

    patch_paths = defect_df["full_path"].dropna().tolist()

    all_seg_rows = []
    combined_overlay_cache = {}  # path -> combined overlay image

    for model_key, seg_model in seg_models.items():
        if seg_model is None:
            continue

        seg_results = segment_patch_paths(
            seg_model,
            patch_paths,
            conf_threshold=conf_threshold,
        )

        for _, row in defect_df.iterrows():
            path = row["full_path"]
            if path not in seg_results:
                continue

            info = seg_results[path]

            if KEEP_SEG_CLASSES is not None:
                if not any(name in KEEP_SEG_CLASSES for name in info["cls_names"]):
                    continue

            # Combine overlays (first model sets base, others blend)
            if path not in combined_overlay_cache:
                combined_overlay_cache[path] = info["overlay"].copy()
            else:
                cv2.addWeighted(combined_overlay_cache[path], 0.5, info["overlay"], 0.5, 0)

            for box_xyxy, cid, cname, conf in zip(
                info.get("boxes_xyxy", []),
                info["cls_ids"],
                info["cls_names"],
                info["confs"],
            ):
                x1, y1, x2, y2 = box_xyxy
                all_seg_rows.append({
                    "filename": row["filename"],
                    "full_path": path,
                    "r": int(row["r"]),
                    "c": int(row["c"]),
                    "distance": float(row["distance"]),
                    "cls_id": int(cid),
                    "cls_name": cname,
                    "cls_conf": float(conf),
                    "bbox_x1_px": float(x1),
                    "bbox_y1_px": float(y1),
                    "bbox_x2_px": float(x2),
                    "bbox_y2_px": float(y2),
                    "model_key": model_key,
                })

    seg_df = pd.DataFrame(all_seg_rows)
    if not seg_df.empty:
        # Use the appropriate enrichment function (already present in file)
        seg_df, dim_summary = enrich_innerwall_yolo_df_with_dimensions(   # * = sidewall/innerwall/etc.
            seg_df=seg_df,
            crop_path=crop_path,
            tyre_name=tyre_name,
            debug_save_dir=None,
        )


    # Stitched image using combined overlays
    stitched_path = None
    if combined_overlay_cache:
        sample = list(combined_overlay_cache.values())[0]
        ph, pw = sample.shape[:2]
        max_r = max(int(x["r"]) for _, x in defect_df.iterrows())
        max_c = max(int(x["c"]) for _, x in defect_df.iterrows())
        canvas = np.zeros(((max_r + 1) * ph, (max_c + 1) * pw, 3), dtype=np.uint8)

        for _, row in defect_df.iterrows():
            path = row["full_path"]
            overlay = combined_overlay_cache.get(path)
            if overlay is None:
                continue
            if overlay.shape[:2] != (ph, pw):
                overlay = cv2.resize(overlay, (pw, ph), interpolation=cv2.INTER_LINEAR)
            y0 = int(row["r"]) * ph
            x0 = int(row["c"]) * pw
            canvas[y0:y0 + ph, x0:x0 + pw] = overlay
            cv2.putText(canvas, f"{row['distance']:.6f}", (x0+5, y0+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1, cv2.LINE_AA)

        stitched_path = os.path.join(save_dir, "final_stitched.png")
        canvas_to_save = cv2.resize(canvas, FINAL_STITCHED_SIZE, interpolation=cv2.INTER_AREA)
        cv2.imwrite(stitched_path, canvas_to_save)
        print(f"[SAVE] {stitched_path}")

    return seg_df, stitched_path, dim_summary

def load_given_reference_preprocessed(ref_image_path):
    """
    Load user-given fixed reference image and polarize it.
    """
    if not os.path.isfile(ref_image_path):
        raise RuntimeError(f"Reference image not found: {ref_image_path}")

    ref_raw_bgr = cv2.imread(ref_image_path)
    if ref_raw_bgr is None:
        raise RuntimeError(f"Cannot read reference image: {ref_image_path}")

    ref_pre_bgr = polarizer_optimized(ref_raw_bgr)
    if ref_pre_bgr is None:
        raise RuntimeError(f"Polarizer failed for reference image: {ref_image_path}")

    return ref_pre_bgr

# =========================================================
# CALIBRATION
# =========================================================
def read_and_polarize(raw_path):
    raw_bgr = cv2.imread(raw_path)
    if raw_bgr is None:
        raise RuntimeError(f"Cannot read image: {raw_path}")

    pre_bgr = polarizer_optimized(raw_bgr)
    if pre_bgr is None:
        raise RuntimeError(f"Polarizer failed for image: {raw_path}")

    return raw_bgr, pre_bgr


def align_crop_from_preprocessed(
    pre_bgr,
    ref_pre_bgr,
    r_detector,
    save_template_path=None,
    ref_info=None,
    use_incoming_r_detection=False,
):
    if USE_ALIGNMENT:
        if use_incoming_r_detection:
            crop_bgr, aligned_bgr, crop_meta = align_and_crop_to_reference_fixed_band(
                image_bgr=pre_bgr,
                reference_bgr=ref_pre_bgr,
                det_model=r_detector,
                slice_h=SLICE_H,
                slice_w=SLICE_W,
                target_size=RESIZE_CROP_TO,
                ref_info=ref_info,
            )
        else:
            crop_bgr, aligned_bgr, crop_meta = crop_from_reference_band_info(
                image_bgr=pre_bgr,
                ref_info=ref_info,
                target_size=RESIZE_CROP_TO,
            )

        if crop_bgr is None:
            raise RuntimeError(crop_meta)
    else:
        crop_bgr = cv2.resize(pre_bgr, RESIZE_CROP_TO, interpolation=cv2.INTER_LINEAR)
        aligned_bgr = crop_bgr.copy()

    if save_template_path is not None:
        cv2.imwrite(save_template_path, aligned_bgr)

    return crop_bgr

def build_calibration_pipeline(model, r_detector, device, gpu_sem=None):
    calib_root = os.path.join(OUTPUT_DIR, "calibration")
    template_dir = os.path.join(calib_root, "template_result")
    crop_dir = os.path.join(calib_root, "cropped")
    art_dir = os.path.join(calib_root, "artifacts")
    summary_dir = os.path.join(calib_root, "summary")

    for d in [template_dir, crop_dir, art_dir, summary_dir]:
        os.makedirs(d, exist_ok=True)

    all_calib_paths = _list_images(CALIB_GOOD_DIR)

    pure_good_paths = [p for p in all_calib_paths if not is_defect_calib_image(p)]
    defect_calib_paths = [p for p in all_calib_paths if is_defect_calib_image(p)] if USE_DEFECT_CALIB_IMAGES else []

    needed_good = MAP_IMAGE_COUNT + THRESH_IMAGE_COUNT
    if len(pure_good_paths) < needed_good:
        raise RuntimeError(
            f"Need at least {needed_good} PURE GOOD images in CALIB_GOOD_DIR "
            f"(excluding def* images). Found only {len(pure_good_paths)}."
        )

    # only pure good images are used for mandatory map/threshold split
    pure_good_paths = pure_good_paths[:needed_good]
    map_raw_paths = pure_good_paths[:MAP_IMAGE_COUNT]
    thr_raw_paths = pure_good_paths[MAP_IMAGE_COUNT:MAP_IMAGE_COUNT + THRESH_IMAGE_COUNT]

    print(f"[CALIB] pure good images used for map     : {len(map_raw_paths)}")
    print(f"[CALIB] pure good images used for thresh  : {len(thr_raw_paths)}")
    print(f"[CALIB] extra defect images included      : {len(defect_calib_paths)}")

    ref_pre_bgr = load_given_reference_preprocessed(REF_IMAGE_PATH)
    ref_pre_path = os.path.join(art_dir, "alignment_reference_polarized.png")
    cv2.imwrite(ref_pre_path, ref_pre_bgr)

    map_patch_dirs = []
    thr_patch_dirs = []
    def_patch_dirs = []

    aug_root_dir = os.path.join(calib_root, "augmented_crops")
    os.makedirs(aug_root_dir, exist_ok=True)

    processing_items = []

    for p in map_raw_paths:
        processing_items.append({
            "raw_path": p,
            "role": "map",
            "is_defect_calib": False,
        })

    for p in thr_raw_paths:
        processing_items.append({
            "raw_path": p,
            "role": "thr",
            "is_defect_calib": False,
        })

    for p in defect_calib_paths:
        processing_items.append({
            "raw_path": p,
            "role": "def_extra",
            "is_defect_calib": True,
        })

    for idx, item in enumerate(processing_items):
        raw_path = item["raw_path"]
        role = item["role"]
        is_defect_calib = item["is_defect_calib"]

        base_stem = Path(raw_path).stem
        name = f"{role}_{base_stem}"

        single_template_dir = os.path.join(template_dir, name)
        single_crop_dir = os.path.join(crop_dir, name)

        _reset_dir(single_template_dir)
        _reset_dir(single_crop_dir)

        template_path = os.path.join(single_template_dir, f"{name}_tm.png")
        crop_path = os.path.join(single_crop_dir, f"{name}_crop.png")

        _, _, crop_bgr = align_crop_from_preprocessed(
            raw_path=raw_path,
            ref_pre_bgr=ref_pre_bgr,
            r_detector=r_detector,
            save_template_path=template_path,
        )

        crop_gray = to_gray(crop_bgr)
        cv2.imwrite(crop_path, crop_gray)

        patches_dir = patchify_index_grouped(
            single_crop_dir,
            patch_h=BIG_PATCH_H,
            patch_w=BIG_PATCH_W,
            step_h=BIG_STEP_H,
            step_w=BIG_STEP_W,
            cover_edges=COVER_EDGES,
        )

        # IMPORTANT:
        # for def* calibration images, remove only the known defective RC patches
        if is_defect_calib:
            removed = remove_ignored_rc_patches_from_dir(
                patches_dir=patches_dir,
                ignore_rcs=DEFECT_IGNORE_RCS,
            )
            print(f"[DEF-CALIB] {name} -> removed {removed} masked defect patches")

        if role == "map":
            map_patch_dirs.append(patches_dir)
        elif role == "thr":
            thr_patch_dirs.append(patches_dir)
        elif role == "def_extra":
            def_patch_dirs.append(patches_dir)
        else:
            raise ValueError(f"Unknown calibration role: {role}")

        if AUGMENT_CALIB:
            aug_patch_dirs = create_augmented_patch_dirs_from_crop(
                crop_bgr=crop_bgr,
                base_name=name,
                aug_root_dir=aug_root_dir,
            )

            # if augmented defect calib images are ever used, mask them too
            if is_defect_calib:
                for apd in aug_patch_dirs:
                    removed = remove_ignored_rc_patches_from_dir(
                        patches_dir=apd,
                        ignore_rcs=DEFECT_IGNORE_RCS,
                    )
                    print(f"[DEF-CALIB-AUG] {name} -> removed {removed} masked patches from augmented dir")

            if role == "map" and AUGMENT_MAP:
                map_patch_dirs.extend(aug_patch_dirs)
            elif role == "thr" and AUGMENT_THRESH:
                thr_patch_dirs.extend(aug_patch_dirs)
            elif role == "def_extra":
                # use augmented defect-calib good patches in both bank + threshold, if wanted
                if AUGMENT_MAP:
                    def_patch_dirs.extend(aug_patch_dirs)
                if AUGMENT_THRESH:
                    def_patch_dirs.extend(aug_patch_dirs)

    # =====================================================
    # Include good patches from defect-calib images
    # =====================================================
    bank_source_dirs = map_patch_dirs + def_patch_dirs
    threshold_source_dirs = thr_patch_dirs 

    gpu_ctx = gpu_sem if gpu_sem is not None else nullcontext()

    with gpu_ctx:

        if DISTANCE_METRIC == "mahalanobis_pca":
            pca_source_dirs = bank_source_dirs if PCA_FIT_ON_MAP_ONLY else (bank_source_dirs + threshold_source_dirs)
            pca_artifact = fit_global_pca_from_patch_dirs(
                model=model,
                patch_dirs=pca_source_dirs,
                device=device,
                n_components=PCA_N_COMPONENTS,
            )
        else:
            pca_artifact = None

        if DISTANCE_METRIC in ["mahalanobis", "mahalanobis_pca"]:
            reference_bank = {}
            reference_bank_meta = {}
            mahalanobis_stats = build_mahalanobis_stats_from_patch_dirs(
                model=model,
                patch_dirs=bank_source_dirs,
                device=device,
                mode=MAHALANOBIS_MODE,
                reg_eps=MAHALANOBIS_REG_EPS,
                min_samples=MAHALANOBIS_MIN_SAMPLES,
                pca_artifact=pca_artifact if DISTANCE_METRIC == "mahalanobis_pca" else None,
            )
        else:
            reference_bank, reference_bank_meta = build_embedding_bank_from_patch_dirs(
                model=model,
                patch_dirs=bank_source_dirs,
                device=device,
                return_meta=True,
            )
            mahalanobis_stats = None

        if USE_LEAVE_ONE_OUT_THRESHOLDS:
            image_patch_features = build_image_patch_feature_dict(
                model=model,
                patch_dirs=threshold_source_dirs,
                device=device,
                pca_artifact=pca_artifact if DISTANCE_METRIC == "mahalanobis_pca" else None,
            )

            dist_by_rc, dist_by_col, dist_by_row, all_distances, rc_rows = collect_good_distances_by_rc_leave_one_out(
                image_patch_features=image_patch_features,
                metric=DISTANCE_METRIC,
                mahalanobis_mode=MAHALANOBIS_MODE,
                reg_eps=MAHALANOBIS_REG_EPS,
                min_samples=MAHALANOBIS_MIN_SAMPLES,
            )
        else:
            dist_by_rc, dist_by_col, dist_by_row, all_distances, rc_rows = collect_good_distances_by_rc(
                model=model,
                patch_dirs=threshold_source_dirs,
                reference_bank=reference_bank,
                device=device,
                mahalanobis_stats=mahalanobis_stats,
                pca_artifact=pca_artifact if DISTANCE_METRIC == "mahalanobis_pca" else None,
            )

        (
        thresholds_by_rc,
        mu_by_rc,
        sigma_by_rc,
        cleaned_dist_by_rc,
        local_debug_rows,
    ) = build_patchwise_thresholds_simple(
        dist_by_rc=dist_by_rc,
        local_percentile=LOCAL_PERCENTILE_AFTER_CLEAN,
        remove_top_outlier=REMOVE_TOP_OUTLIER_PER_RC,
        outlier_ratio=OUTLIER_RATIO,
    )

        torch.save(reference_bank, os.path.join(art_dir, "embedding_bank.pt"))
        torch.save(reference_bank_meta, os.path.join(art_dir, "embedding_bank_meta.pt"))

        if mahalanobis_stats is not None:
            torch.save(mahalanobis_stats, os.path.join(art_dir, "mahalanobis_stats.pt"))
        if pca_artifact is not None:
            torch.save(pca_artifact, os.path.join(art_dir, "pca_artifact.pt"))

        torch.save(
        {
            "thresholds_by_rc": thresholds_by_rc,
            "mu_by_rc": mu_by_rc,
            "sigma_by_rc": sigma_by_rc,
        },
        os.path.join(art_dir, "thresholds_by_rc.pt"),
    )
        pd.DataFrame(local_debug_rows).to_csv(
        os.path.join(summary_dir, "calibration_local_threshold_debug.csv"),
        index=False,
    )

        col_summary_rows = []
        for c, vals in dist_by_col.items():
            vals_np = np.array(vals, dtype=np.float32)
            if len(vals_np) == 0:
                continue
            col_summary_rows.append({
                "c": int(c),
                "count": int(len(vals_np)),
                "min_dist": float(np.min(vals_np)),
                "max_dist": float(np.max(vals_np)),
                "mean_dist": float(np.mean(vals_np)),
                "std_dist": float(np.std(vals_np)),
                "p95": float(np.percentile(vals_np, 95)),
                "p99": float(np.percentile(vals_np, 99)),
            })

        pd.DataFrame(col_summary_rows).to_csv(
            os.path.join(summary_dir, "calibration_column_summary.csv"),
            index=False,
        )

        row_summary_rows = []
        for r, vals in dist_by_row.items():
            vals_np = np.array(vals, dtype=np.float32)
            if len(vals_np) == 0:
                continue
            row_summary_rows.append({
                "r": int(r),
                "count": int(len(vals_np)),
                "min_dist": float(np.min(vals_np)),
                "max_dist": float(np.max(vals_np)),
                "mean_dist": float(np.mean(vals_np)),
                "std_dist": float(np.std(vals_np)),
                "p95": float(np.percentile(vals_np, 95)),
                "p99": float(np.percentile(vals_np, 99)),
            })

        pd.DataFrame(row_summary_rows).to_csv(
            os.path.join(summary_dir, "calibration_row_summary.csv"),
            index=False,
        )

        print("[DONE] calibration pipeline finished")

def patchify_array_indexed(img_gray, patch_h, patch_w, step_h, step_w, cover_edges=True):
    H, W = img_gray.shape[:2]
    ys = list(range(0, max(H - patch_h + 1, 1), step_h))
    xs = list(range(0, max(W - patch_w + 1, 1), step_w))

    if cover_edges:
        if ys[-1] != H - patch_h:
            ys.append(max(H - patch_h, 0))
        if xs[-1] != W - patch_w:
            xs.append(max(W - patch_w, 0))

    records = []
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            patch = img_gray[y:y+patch_h, x:x+patch_w].copy()
            records.append({
                "r": r,
                "c": c,
                "patch": patch,
                "name": f"patch__r{r:03d}_c{c:03d}.png"
            })
    return records

# =========================================================
# LOAD ARTIFACTS
# =========================================================
def load_calibration_artifacts_from_dir(output_dir, ref_image_path_override=None):
    """
    Dynamic artifact loading for Maincycle infer mode.

    If Maincycle passes:
        media/calibration/<SKU_NAME>/artifacts/alignment_reference_polarized.png

    then load all artifacts from that SKU artifacts folder.
    """

    if ref_image_path_override:
        art_dir = os.path.dirname(ref_image_path_override)
    else:
        calib_root = os.path.join(output_dir, "calibration")
        art_dir = os.path.join(calib_root, "artifacts")

    meta_path = os.path.join(art_dir, "embedding_bank_meta.pt")
    ref_pre_path = os.path.join(art_dir, "alignment_reference_polarized.png")
    bank_path = os.path.join(art_dir, "embedding_bank.pt")
    thr_path = os.path.join(art_dir, "thresholds_by_rc.pt")
    mahal_path = os.path.join(art_dir, "mahalanobis_stats.pt")
    pca_path = os.path.join(art_dir, "pca_artifact.pt")
    reference_r_path = os.path.join(art_dir, "reference_r.pt")

    if not os.path.isfile(ref_pre_path):
        raise RuntimeError(f"Missing alignment reference: {ref_pre_path}")

    if not os.path.isfile(thr_path):
        raise RuntimeError(f"Missing thresholds: {thr_path}")

    if DISTANCE_METRIC in ["mahalanobis", "mahalanobis_pca"] and not os.path.isfile(mahal_path):
        raise RuntimeError(f"Missing mahalanobis stats: {mahal_path}")

    if DISTANCE_METRIC == "mahalanobis_pca" and not os.path.isfile(pca_path):
        raise RuntimeError(f"Missing PCA artifact: {pca_path}")

    ref_pre_bgr = cv2.imread(ref_pre_path)
    if ref_pre_bgr is None:
        raise RuntimeError(f"Cannot read alignment reference: {ref_pre_path}")

    reference_bank = torch.load(bank_path, map_location="cpu") if os.path.isfile(bank_path) else {}
    reference_bank_meta = torch.load(meta_path, map_location="cpu") if os.path.isfile(meta_path) else {}
    thr_obj = torch.load(thr_path, map_location="cpu")
    mahalanobis_stats = torch.load(mahal_path, map_location="cpu") if os.path.isfile(mahal_path) else None
    pca_artifact = torch.load(pca_path, map_location="cpu") if os.path.isfile(pca_path) else None
    reference_r = torch.load(reference_r_path, map_location="cpu") if os.path.isfile(reference_r_path) else None

    thresholds_by_rc = thr_obj["thresholds_by_rc"]
    mu_by_rc = thr_obj["mu_by_rc"]
    sigma_by_rc = thr_obj["sigma_by_rc"]

    print(f"[ARTIFACTS] loaded from: {art_dir}")

    return (
        ref_pre_bgr,
        reference_r,
        reference_bank,
        reference_bank_meta,
        thresholds_by_rc,
        mu_by_rc,
        sigma_by_rc,
        mahalanobis_stats,
        pca_artifact,
    )

def load_calibration_artifacts():
    return load_calibration_artifacts_from_dir(OUTPUT_DIR)

def load_runtime(
    device=None,
    seg_models=None,
    seg_model_override=None,
    r_detector_override=None,
    use_yolo_seg_override=None,
    checkpoint_path_override=None,
    output_dir_override=None,
    yolo_r_path_override=None,
    ref_image_path_override=None,
    tyre_name_override=None,
    load_artifacts=True,
    trt_vit=None,
    use_trt_vit=False,
):
    output_dir = output_dir_override or OUTPUT_DIR
    checkpoint_path = checkpoint_path_override or CHECKPOINT_PATH
    yolo_r_path = yolo_r_path_override or YOLO_R_PATH
    os.makedirs(output_dir, exist_ok=True)

    if device is None:
        device = DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, using CPU")
        device = "cpu"

    if r_detector_override is not None:
        r_detector = r_detector_override
        print("[RUNTIME] using provided R-detector")
    else:
        r_detector = build_r_detector(yolo_r_path, conf=CONF_THRES_R, device=device)

    if use_trt_vit and trt_vit is not None:
        model = trt_vit
        print("[RUNTIME] using TensorRT ViT engine")
    else:
        # IMPORTANT:
        # If checkpoint_path points to a TensorRT engine, do NOT try to load it with torch.load.
        # We are deferring TRT attachment until after runtime creation in Maincycle.
        if checkpoint_path and str(checkpoint_path).lower().endswith(".engine"):
            model = None
            print("[RUNTIME] deferring ViT model load until TRT engine is attached")
        else:
            model = make_model().to(device).eval()
            model = model.half()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            load_checkpoint(model, optimizer, checkpoint_path)

    patch_transform = _build_transform()
    use_yolo_seg = USE_YOLO_SEG if use_yolo_seg_override is None else bool(use_yolo_seg_override)

    if seg_models is not None:
        runtime_seg_models = seg_models
    elif seg_model_override is not None:
        runtime_seg_models = {"default": seg_model_override}
    else:
        runtime_seg_models = {}
        if use_yolo_seg:
            try:
                runtime_seg_models["default"] = load_yolo_seg(YOLO_SEG_MODEL_PATH, device=device)
                print("[YOLO] segmentation model loaded")
            except Exception as e:
                print(f"[YOLO][WARN] failed to load model: {e}")

    if load_artifacts:
        (
            ref_pre_bgr,
            reference_r,
            reference_bank,
            reference_bank_meta,
            thresholds_by_rc,
            mu_by_rc,
            sigma_by_rc,
            mahalanobis_stats,
            pca_artifact,
        ) = load_calibration_artifacts_from_dir(
            output_dir,
            ref_image_path_override=ref_image_path_override,
        )
    else:
        ref_pre_bgr = None
        reference_r = None
        reference_bank = {}
        reference_bank_meta = {}
        thresholds_by_rc = {}
        mu_by_rc = {}
        sigma_by_rc = {}
        mahalanobis_stats = None
        pca_artifact = None

    reference_band_info = None

    if ref_pre_bgr is not None and reference_r is not None and len(reference_r) >= 2:
        H, W = ref_pre_bgr.shape[:2]

        y1 = int(round(reference_r[0][1]))
        y2 = int(round(reference_r[1][1]))

        y1 = max(0, min(y1, H - 1))
        y2 = max(0, min(y2, H - 1))

        if y2 <= y1:
            raise RuntimeError(
                f"Invalid saved reference_r crop band: y1={y1}, y2={y2}, reference_r={reference_r}"
            )

        reference_band_info = {
            "status": "ok",
            "ref_r": reference_r,
            "y1": y1,
            "y2": y2,
            "ref_h": H,
            "ref_w": W,
        }
        print("[RUNTIME] using saved reference_r.pt for reference band")

    elif ref_pre_bgr is not None:
        reference_band_info = get_reference_r_band(
            ref_pre_bgr,
            r_detector,
            SLICE_H,
            SLICE_W,
        )
        if reference_band_info["status"] != "ok":
            raise RuntimeError(f"Failed to precompute reference band: {reference_band_info}")

    r_detector = None

    return {
        "device": device,
        "model": model,
        "patch_transform": patch_transform,
        "r_detector": r_detector,
        "seg_models": runtime_seg_models,
        "use_yolo_seg": use_yolo_seg,
        "ref_pre_bgr": ref_pre_bgr,
        "reference_bank": reference_bank,
        "reference_bank_meta": reference_bank_meta,
        "thresholds_by_rc": thresholds_by_rc,
        "reference_band_info": reference_band_info,
        "tyre_name": tyre_name_override,
        "mu_by_rc": mu_by_rc,
        "sigma_by_rc": sigma_by_rc,
        "mahalanobis_stats": mahalanobis_stats,
        "pca_artifact": pca_artifact,
        "output_dir": output_dir,
        "checkpoint_path": checkpoint_path,
        "yolo_r_path": yolo_r_path,
        "use_trt_vit": (trt_vit is not None and use_trt_vit),   # NEW
    }

def warmup_runtime(runtime):
    import tempfile

    device = runtime["device"]
    model = runtime["model"]
    seg_models = runtime.get("seg_models", {})

    if device != "cuda":
        return

    try:
        # Warm up ViT
        if hasattr(model, "extract"):
            # TRT warmup: run a dummy inference
            dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).cpu()
            _ = model.extract(dummy)
            print("[WARMUP] TRT ViT warmed up")
        else:
            with torch.inference_mode():
                dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=device)
                if device == "cuda":
                    dummy = dummy.half()
                _ = model(dummy)
            print("[WARMUP] PyTorch ViT warmed up")

        # Warm up YOLO segmentation models with REAL tyre patches
        if seg_models:
            warmup_patches = []
            created_temp_files = []
            temp_dir = tempfile.gettempdir()

            # Try to find actual patches from output directory
            calib_patches_dir = os.path.join(
                runtime.get("output_dir", ""),
                "calibration",
                "cropped"
            )

            if os.path.isdir(calib_patches_dir):
                for root, _, files in os.walk(calib_patches_dir):
                    for f in files:
                        if f.lower().endswith((".png", ".jpg", ".jpeg")):
                            warmup_patches.append(os.path.join(root, f))
                            if len(warmup_patches) >= 3:
                                break
                    if warmup_patches:
                        break

            # If no real patches found, create temporary warmup patches
            if not warmup_patches:
                for i in range(3):
                    texture = np.random.randint(
                        0, 255, (BIG_PATCH_H, BIG_PATCH_W, 3), dtype=np.uint8
                    )
                    texture = cv2.GaussianBlur(texture, (5, 5), 0)
                    temp_path = os.path.join(temp_dir, f"warmup_patch_{i}.png")
                    cv2.imwrite(temp_path, texture)
                    warmup_patches.append(temp_path)
                    created_temp_files.append(temp_path)

            print(f"[WARMUP] Using {len(warmup_patches)} patches for YOLO warmup")

            for model_key, seg_model in seg_models.items():
                if seg_model is None:
                    continue
                try:
                    _ = segment_patch_paths(
                        seg_model,
                        warmup_patches[:3],
                        conf_threshold=0.5,
                        max_batch_size=3,
                    )
                    print(f"[WARMUP] YOLO model '{model_key}' warmed up with real patches")
                except Exception as e:
                    print(f"[WARMUP][WARN] YOLO warmup failed for '{model_key}': {e}")

            # Clean up only the temp files we created
            for p in created_temp_files:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

        # Warm up alignment
        try:
            ref_pre_bgr = runtime.get("ref_pre_bgr")
            if ref_pre_bgr is not None:
                _ = align_crop_from_preprocessed(
                    pre_bgr=ref_pre_bgr.copy(),
                    ref_pre_bgr=ref_pre_bgr,
                    r_detector=None,
                    save_template_path=None,
                    ref_info=runtime.get("reference_band_info"),
                    use_incoming_r_detection=False,
                )
        except Exception:
            pass

        torch.cuda.synchronize()
        print("[WARMUP] done")

    except Exception as e:
        print(f"[WARMUP][WARN] {e}")

def calibrate_side(
    runtime,
    calib_good_dir_override=None,
    output_dir_override=None,
    ref_image_path_override=None,
    gpu_sem=None,
):
    calib_dir  = calib_good_dir_override or CALIB_GOOD_DIR
    output_dir = output_dir_override     or OUTPUT_DIR
    ref_path   = ref_image_path_override or REF_IMAGE_PATH

    return build_calibration_pipeline(
        runtime["model"],
        runtime["r_detector"],
        runtime["device"],
        gpu_sem=gpu_sem,
        calib_good_dir=calib_dir,
        output_dir=output_dir,
        ref_image_path=ref_path,
    )

def infer_patches_generic_from_arrays(
    model,
    patch_records,
    reference_bank,
    reference_bank_meta,
    thresholds_by_rc,
    mu_by_rc,
    sigma_by_rc,
    mahalanobis_stats,
    pca_artifact,
    save_dir,
    device,
    patch_transform=None,
):
    os.makedirs(save_dir, exist_ok=True)

    rows = []
    raw_compare_rows = []
    visual_records = []

    total = len(patch_records)
    paired = 0
    skipped_no_rc = 0
    skipped_no_ref = 0
    skipped_black_bg = 0
    skipped_no_threshold = 0

    if patch_transform is None:
        patch_transform = _build_transform()

    usable_patch_records = []
    for rec in patch_records:
        r = rec.get("r")
        c = rec.get("c")
        patch = rec.get("patch")

        if r is None or c is None:
            skipped_no_rc += 1
            continue

        if not is_nonblack_patch_array(patch, black_thresh=10, min_nonblack_ratio=0.25):
            skipped_black_bg += 1
            continue

        usable_patch_records.append(rec)

    for batch_recs in _batched(usable_patch_records, batch_size=BATCH_SIZE):
        emb, valid_recs = get_patch_embeddings_from_arrays(
            model=model,
            patch_records=batch_recs,
            device=device,
            tfm=patch_transform,
        )

        for i, rec in enumerate(valid_recs):
            r = rec.get("r")
            c = rec.get("c")
            patch = rec.get("patch")
            filename = rec.get("name", f"patch__r{r:03d}_c{c:03d}.png")

            if r is None or c is None:
                skipped_no_rc += 1
                continue

            if not is_nonblack_patch_array(patch, black_thresh=10, min_nonblack_ratio=0.25):
                skipped_black_bg += 1
                continue

            key = (int(r), int(c))

            has_ref = (
                (reference_bank is not None and key in reference_bank) or
                (mahalanobis_stats is not None and key in mahalanobis_stats)
            )
            if not has_ref:
                skipped_no_ref += 1
                continue

            query_vec = emb[i].clone().float()
            if DISTANCE_METRIC == "mahalanobis_pca" and pca_artifact is not None:
                query_vec = pca_transform_embedding(query_vec, pca_artifact)

            nearest_ref_name = None
            ang_dist = None
            mahal_stats = mahalanobis_stats.get(key) if mahalanobis_stats is not None and key in mahalanobis_stats else None

            if DISTANCE_METRIC in ["mahalanobis", "mahalanobis_pca"]:
                nearest_ref_name = None
                _, best_dist = nearest_distance_to_bank(
                    query_emb=query_vec,
                    bank_embs=None,
                    metric=DISTANCE_METRIC,
                    mahalanobis_stats=mahal_stats,
                )
            else:
                bank_embs = reference_bank.get(key, [])
                if not bank_embs:
                    skipped_no_ref += 1
                    continue

                nearest_ref_name = None
                _, best_dist = nearest_distance_to_bank(
                    query_emb=query_vec,
                    bank_embs=bank_embs,
                    metric=DISTANCE_METRIC,
                    mahalanobis_stats=None,
                )

            thr = thresholds_by_rc.get(key)
            mu = mu_by_rc.get(key)
            sigma_eff = sigma_by_rc.get(key)

            if thr is None or mu is None or sigma_eff is None:
                skipped_no_threshold += 1
                continue

            thr = float(thr)
            mu = float(mu)
            sigma_eff = max(float(sigma_eff), SIGMA_FLOOR)

            z_score = (float(best_dist) - mu) / sigma_eff
            is_defect = float(best_dist) > thr
            paired += 1

            row = {
                "filename": filename,
                "full_path": None,   # fill later only for defect patches
                "r": int(r),
                "c": int(c),
                "distance": float(best_dist),
                "nearest_ref": nearest_ref_name,
                "mahalanobis_mode": mahal_stats.get("mode") if mahal_stats is not None else None,
                "mahalanobis_num_samples": int(mahal_stats.get("num_samples", 0)) if mahal_stats is not None else None,
                "ang_dist_rad": float(ang_dist) if ang_dist is not None else None,
                "threshold_used": float(thr),
                "mu_used": float(mu),
                "sigma_used": float(sigma_eff),
                "z_score": float(z_score),
                "classification": "DEFECT" if is_defect else "GOOD",
            }
            rows.append(row)

            raw_compare_rows.append({
                "filename": filename,
                "r": int(r),
                "c": int(c),
                "distance": float(best_dist),
                "nearest_ref": nearest_ref_name,
            })

            visual_records.append({
                "r": int(r),
                "c": int(c),
                "patch": patch,
                "distance": float(best_dist),
                "classification": row["classification"],
                "filename": filename,
            })

    print(
        f"[{DISTANCE_METRIC.upper()}] total={total} | paired={paired} | "
        f"skip_no_rc={skipped_no_rc} | skip_no_ref={skipped_no_ref} | "
        f"skip_black_bg={skipped_black_bg} | skip_no_threshold={skipped_no_threshold}"
    )

    df = pd.DataFrame(rows)

    SAVE_RAW_COMPARE_CSV = False
    out_csv = os.path.join(save_dir, "patch_distance_results.csv")
    df.to_csv(out_csv, index=False)

    if SAVE_RAW_COMPARE_CSV:
        raw_compare_csv = os.path.join(save_dir, "patch_all_reference_distances.csv")
        pd.DataFrame(raw_compare_rows).to_csv(raw_compare_csv, index=False)

    stitched_template_path = None

    if visual_records:
        ph, pw = visual_records[0]["patch"].shape[:2]
        max_r = max(int(x["r"]) for x in visual_records)
        max_c = max(int(x["c"]) for x in visual_records)

        canvas = np.zeros(((max_r + 1) * ph, (max_c + 1) * pw, 3), dtype=np.uint8)

        for rec in visual_records:
            patch = rec["patch"]
            if patch.ndim == 2:
                patch_bgr = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
            else:
                patch_bgr = patch.copy()

            y0 = int(rec["r"]) * ph
            x0 = int(rec["c"]) * pw
            canvas[y0:y0 + ph, x0:x0 + pw] = patch_bgr

            if rec["classification"] == "DEFECT":
                color = (0, 0, 255)
                cv2.rectangle(canvas, (x0, y0), (x0 + pw, y0 + ph), color, 2)
                cv2.putText(
                    canvas,
                    f"{rec['distance']:.2f}",
                    (x0 + 5, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        stitched_template_path = os.path.join(save_dir, "template_stitched.png")
        cv2.imwrite(stitched_template_path, canvas)

    # save only defect patches for YOLO
    if not df.empty:
        defect_cache_dir = os.path.join(save_dir, "defect_patch_cache")
        os.makedirs(defect_cache_dir, exist_ok=True)

        filename_to_patch = {x["filename"]: x["patch"] for x in visual_records}

        for idx, row in df.iterrows():
            if row["classification"] != "DEFECT":
                continue

            patch = filename_to_patch.get(row["filename"])
            if patch is None:
                continue

            defect_path = os.path.join(defect_cache_dir, row["filename"])
            cv2.imwrite(defect_path, patch)
            df.at[idx, "full_path"] = defect_path

    return df, stitched_template_path

# =========================================================
# BATCH INFERENCE HELPERS 
# =========================================================

@torch.inference_mode()
def get_patch_embeddings_batched(model, patch_records_by_side, device, tfm=None):
    """
    Batch patches from multiple sides together.
    """
    if tfm is None:
        tfm = _build_transform()
    
    all_imgs = []
    all_metadata = []
    
    for side_name, patch_records in patch_records_by_side.items():
        for idx, rec in enumerate(patch_records):
            try:
                rgb = cv2.cvtColor(rec["patch"], cv2.COLOR_GRAY2RGB)
                pil = Image.fromarray(rgb)
                all_imgs.append(tfm(pil))
                all_metadata.append((side_name, idx, rec))
            except Exception:
                pass
    
    if not all_imgs:
        return {}, {}
    
    batch = torch.stack(all_imgs).to(device, non_blocking=True)
    if device == "cuda":
        batch = batch.half()
    
    if USE_INTERMEDIATE_BLOCKS:
        all_embs = extract_vit_features(
            model=model,
            batch=batch,
            target_block_indices=TARGET_BLOCK_INDICES,
            fusion=BLOCK_FUSION,
            normalize_each_block=NORMALIZE_EACH_BLOCK,
            normalize_final=NORMALIZE_EMBEDDINGS,
        )
    else:
        tokens = model.encoder.forward_features(batch)
        patch_tokens = tokens[:, 1:, :]
        all_embs = patch_tokens.mean(dim=1)
        if NORMALIZE_EMBEDDINGS:
            all_embs = F.normalize(all_embs, dim=1)
    
    all_embs = all_embs.detach().cpu()
    
    embeddings_by_side = defaultdict(list)
    valid_records_by_side = defaultdict(list)
    
    for emb, (side_name, orig_idx, rec) in zip(all_embs, all_metadata):
        embeddings_by_side[side_name].append(emb)
        valid_records_by_side[side_name].append(rec)
    
    for side_name in embeddings_by_side:
        embeddings_by_side[side_name] = torch.stack(embeddings_by_side[side_name])
    
    return embeddings_by_side, valid_records_by_side


def process_precomputed_embeddings(embeddings, valid_records, runtime, save_dir, defect_cache_dir=None):
    """
    Process pre-computed embeddings (distance, threshold, classification)
    WITHOUT running ViT again.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    rows = []
    visual_records = []
    
    reference_bank = runtime.get("reference_bank", {})
    mahalanobis_stats = runtime.get("mahalanobis_stats", {})
    thresholds_by_rc = runtime.get("thresholds_by_rc", {})
    mu_by_rc = runtime.get("mu_by_rc", {})
    sigma_by_rc = runtime.get("sigma_by_rc", {})
    pca_artifact = runtime.get("pca_artifact")
    
    for i, rec in enumerate(valid_records):
        r = rec.get("r")
        c = rec.get("c")
        patch = rec.get("patch")
        filename = rec.get("name", f"patch__r{r:03d}_c{c:03d}.png")
        
        if r is None or c is None:
            continue
        
        if not is_nonblack_patch_array(patch, black_thresh=10, min_nonblack_ratio=0.25):
            continue
        
        key = (int(r), int(c))
        query_vec = embeddings[i].clone().float()
        
        if DISTANCE_METRIC == "mahalanobis_pca" and pca_artifact is not None:
            query_vec = pca_transform_embedding(query_vec, pca_artifact)
        
        if DISTANCE_METRIC in ["mahalanobis", "mahalanobis_pca"]:
            mahal_stats = mahalanobis_stats.get(key)
            if mahal_stats is None:
                continue
            _, best_dist = nearest_distance_to_bank(
                query_emb=query_vec,
                bank_embs=None,
                metric=DISTANCE_METRIC,
                mahalanobis_stats=mahal_stats,
            )
        else:
            bank_embs = reference_bank.get(key)
            if bank_embs is None:
                continue
            _, best_dist = nearest_distance_to_bank(
                query_emb=query_vec,
                bank_embs=bank_embs,
                metric=DISTANCE_METRIC,
            )
        
        if best_dist is None:
            continue
        
        thr = thresholds_by_rc.get(key)
        mu = mu_by_rc.get(key)
        sigma_eff = sigma_by_rc.get(key)
        
        if thr is None or mu is None or sigma_eff is None:
            continue
        
        thr = float(thr)
        mu = float(mu)
        sigma_eff = max(float(sigma_eff), SIGMA_FLOOR)
        
        z_score = (float(best_dist) - mu) / sigma_eff
        is_defect = float(best_dist) > thr
        
        row = {
            "filename": filename,
            "full_path": None,
            "r": int(r),
            "c": int(c),
            "distance": float(best_dist),
            "threshold_used": thr,
            "mu_used": mu,
            "sigma_used": sigma_eff,
            "z_score": float(z_score),
            "classification": "DEFECT" if is_defect else "GOOD",
        }
        rows.append(row)
        
        visual_records.append({
            "r": int(r),
            "c": int(c),
            "patch": patch,
            "distance": float(best_dist),
            "classification": row["classification"],
            "filename": filename,
        })
    
    df = pd.DataFrame(rows)
    
    stitched_path = None
    if visual_records:
        ph, pw = visual_records[0]["patch"].shape[:2]
        max_r = max(x["r"] for x in visual_records)
        max_c = max(x["c"] for x in visual_records)
        
        canvas = np.zeros(((max_r + 1) * ph, (max_c + 1) * pw, 3), dtype=np.uint8)
        
        for rec in visual_records:
            patch = rec["patch"]
            if patch.ndim == 2:
                patch_bgr = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
            else:
                patch_bgr = patch.copy()
            
            y0 = int(rec["r"]) * ph
            x0 = int(rec["c"]) * pw
            canvas[y0:y0 + ph, x0:x0 + pw] = patch_bgr
            
            if rec["classification"] == "DEFECT":
                cv2.rectangle(canvas, (x0, y0), (x0 + pw, y0 + ph), (0, 0, 255), 2)
                cv2.putText(canvas, f"{rec['distance']:.2f}", (x0 + 5, y0 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
        
        stitched_path = os.path.join(save_dir, "template_stitched.png")
        cv2.imwrite(stitched_path, canvas)
    
    if not df.empty:
        if defect_cache_dir is None:
            defect_cache_dir = os.path.join(save_dir, "defect_patch_cache")

        os.makedirs(defect_cache_dir, exist_ok=True)

        filename_to_patch = {x["filename"]: x["patch"] for x in visual_records}

        if "full_path" not in df.columns:
            df["full_path"] = None

        for idx, row in df.iterrows():
            if row["classification"] != "DEFECT":
                continue

            patch = filename_to_patch.get(row["filename"])
            if patch is None:
                continue

            defect_path = os.path.join(defect_cache_dir, row["filename"])
            cv2.imwrite(defect_path, patch)
            df.at[idx, "full_path"] = defect_path
    
    return df, stitched_path



