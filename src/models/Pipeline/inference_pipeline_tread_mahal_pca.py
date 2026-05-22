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

from src.COMMON.common import tread_dimensions
from src.models.defect_dimension import area_defect_tread, cor_tread

from src.models.Pipeline.polarizer import polarizer_optimized

from src.models.Pipeline.patchs import patchify_index_grouped
from src.models.Pipeline.vit_autoencoder import ViTEncoderDecoder
from src.models.Pipeline.yolo_patch_classifier import load_yolo_seg, segment_patch_paths

try:
    from src.models.Pipeline.vit_trt_inference import TRTViTFeatureExtractor
except Exception:
    TRTViTFeatureExtractor = None

try:
    from src.models.Pipeline.checkpoint import load_checkpoint
except Exception:
    from src.models.Pipeline.checkpoint import load_checkpoint

from src.models.Pipeline.R_inner_mapping_alignment import (
    crop_resize_xalign_non_r_side,
)

# =========================================================
# CONFIG
# =========================================================
MODE = "infer"   # "calibrate" or "infer"
DEBUG_SAVE_INTERMEDIATE = False

# calibration good raw images
CALIB_GOOD_DIR = r"C:\Users\DELL\Downloads\sidewall_qutrac\sidewall_def\calib"
CALIBRATION_DIR_NAME = "calibration_tread"

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

# =========================================================
# INFERENCE DEFECT DECISION FILTERS
# =========================================================
USE_Z_SCORE_FILTER = True
USE_SCORE_RATIO_FILTER = True

# distance must be this much higher than threshold
# 1.00 means just above threshold
# 1.05 means 5% above threshold
# 1.10 means 10% above threshold
SCORE_RATIO_THRESHOLD = 1.30
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
USE_DEFECT_CALIB_IMAGES = False
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

SIDE_NAME = "tread"
SIDE_LABEL_PREFIX = "TREAD"
ENABLE_YOLO_DIMENSIONS = False

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

    # TRT path
    if hasattr(model, "extract"):
        batch = torch.stack(imgs).cpu()   # TRT extractor expects CPU tensor input
        embeddings = model.extract(batch) # returns torch tensor
        return embeddings, valid_paths

    # Original PyTorch path
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
            label_prefix=SIDE_LABEL_PREFIX,
        )

        for _, row in defect_df.iterrows():
            path = row["full_path"]
            if path not in seg_results:
                continue

            info = seg_results[path]

            filter_names = info.get("cls_names_raw", info["cls_names"])

            if KEEP_SEG_CLASSES is not None:
                if not any(name in KEEP_SEG_CLASSES for name in filter_names):
                    continue

            # Combine overlays (first model sets base, others blend)
            if path not in combined_overlay_cache:
                combined_overlay_cache[path] = info["overlay"].copy()
            else:
                cv2.addWeighted(combined_overlay_cache[path], 0.5, info["overlay"], 0.5, 0)

            cls_names_raw = info.get("cls_names_raw", info["cls_names"])
            cls_names_prefixed = info["cls_names"]

            for box_xyxy, cid, cname_raw, cname_prefixed, conf in zip(
                info.get("boxes_xyxy", []),
                info["cls_ids"],
                cls_names_raw,
                cls_names_prefixed,
                info["confs"],
            ):
                x1, y1, x2, y2 = box_xyxy
                all_seg_rows.append({
                    "filename": row["filename"],
                    "full_path": path,
                    "r": int(row["r"]),
                    "c": int(row["c"]),
                    "distance": float(row["distance"]),
                    "side": SIDE_NAME,
                    "cls_id": int(cid),
                    "cls_name_raw": cname_raw,
                    "cls_name": cname_prefixed,
                    "cls_conf": float(conf),
                    "bbox_x1_px": float(x1),
                    "bbox_y1_px": float(y1),
                    "bbox_x2_px": float(x2),
                    "bbox_y2_px": float(y2),
                    "model_key": model_key,
                })

    seg_df = pd.DataFrame(all_seg_rows)

    if not seg_df.empty:
        print(f"[DIM] Skipping dimension calculation for {SIDE_NAME}; laser measurement will be used.")

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
        cv2.imwrite(stitched_path, canvas)
        print(f"[SAVE] {stitched_path}")

    return seg_df, stitched_path, dim_summary

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
    sidewall_r_anchor,
    offset_ratio,
    x_align_artifacts_dir,
    create_x_reference_if_missing=True,
    x_align_debug_path=None,
    return_meta=False,
):
    """
    TREAD flow:

        polarized tread image
        -> crop using current sidewall R1/R2 + tread offset_ratio
        -> resize to RESIZE_CROP_TO
        -> tread profile X alignment
        -> return final crop

    No tread R detection.
    No alignment_reference_polarized.
    No reference_r.
    No reference_band_info.
    """

    if sidewall_r_anchor is None:
        raise RuntimeError(
            f"[{SIDE_NAME}] sidewall_r_anchor is required. "
            f"Tread must use current sidewall R1/R2 from Maincycle."
        )

    if offset_ratio is None:
        raise RuntimeError(f"[{SIDE_NAME}] offset_ratio is required.")

    if x_align_artifacts_dir is None:
        raise RuntimeError(f"[{SIDE_NAME}] x_align_artifacts_dir is required.")

    crop_bgr, meta = crop_resize_xalign_non_r_side(
        pre_bgr=pre_bgr,
        side_name=SIDE_NAME,
        sidewall_r_anchor=sidewall_r_anchor,
        offset_ratio=float(offset_ratio),
        target_size=RESIZE_CROP_TO,
        artifacts_dir=x_align_artifacts_dir,
        create_x_reference_if_missing=create_x_reference_if_missing,
        debug_save_path=x_align_debug_path,
    )

    if return_meta:
        return crop_bgr, meta

    return crop_bgr


def build_calibration_pipeline(
    model,
    r_detector,
    device,
    gpu_sem=None,
    calib_good_dir=None,
    output_dir=None,
    ref_image_path=None,

    # New calibration inputs from Maincycle
    sidewall_anchor_records=None,
    offset_ratio=None,
    x_align_artifacts_dir=None,
):
    calib_good_dir = calib_good_dir or CALIB_GOOD_DIR
    output_dir = output_dir or OUTPUT_DIR

    calib_root = os.path.join(output_dir, CALIBRATION_DIR_NAME)
    template_dir = os.path.join(calib_root, "template_result")
    crop_dir = os.path.join(calib_root, "cropped")
    art_dir = os.path.join(calib_root, "artifacts")
    summary_dir = os.path.join(calib_root, "summary")

    for d in [template_dir, crop_dir, art_dir, summary_dir]:
        os.makedirs(d, exist_ok=True)

    all_calib_paths = _list_images(calib_good_dir)

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

    if sidewall_anchor_records is None:
        raise RuntimeError(
            f"[{SIDE_NAME} CALIB] sidewall_anchor_records is required. "
            f"Run sidewall calibration first and pass its R anchors."
        )

    if offset_ratio is None:
        raise RuntimeError(f"[{SIDE_NAME} CALIB] offset_ratio is required.")

    if len(sidewall_anchor_records) != len(processing_items):
        raise RuntimeError(
            f"[{SIDE_NAME} CALIB] sidewall anchor count mismatch. "
            f"sidewall_anchor_records={len(sidewall_anchor_records)}, "
            f"processing_items={len(processing_items)}. "
            f"Make sure sidewall and {SIDE_NAME} calibration folders contain matching images in sorted order."
        )

    x_align_artifacts_dir = x_align_artifacts_dir or art_dir

    x_align_artifacts_dir = x_align_artifacts_dir or art_dir

    old_ref_files = [
        # old edge-based reference
        os.path.join(x_align_artifacts_dir, f"{SIDE_NAME}_x_ref_edges.json"),
        os.path.join(x_align_artifacts_dir, f"{SIDE_NAME}_x_reference_bbox.png"),

        # new tread profile reference
        os.path.join(x_align_artifacts_dir, f"{SIDE_NAME}_x_reference_signature.npy"),
        os.path.join(x_align_artifacts_dir, f"{SIDE_NAME}_x_reference_signature_meta.json"),
        os.path.join(x_align_artifacts_dir, f"{SIDE_NAME}_x_reference_crop.png"),
    ]

    for p in old_ref_files:
        if os.path.exists(p):
            os.remove(p)
            print(f"[{SIDE_NAME.upper()} XALIGN] Removed old reference: {p}")

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

        _raw_bgr, pre_bgr = read_and_polarize(raw_path)

        anchor_record = sidewall_anchor_records[idx]
        sidewall_r_anchor = anchor_record["sidewall_r_anchor"]

        x_align_debug_path = os.path.join(
            single_template_dir,
            f"{name}_{SIDE_NAME}_xalign_debug.png"
        )

        crop_bgr, crop_meta = align_crop_from_preprocessed(
            pre_bgr=pre_bgr,
            sidewall_r_anchor=sidewall_r_anchor,
            offset_ratio=float(offset_ratio),
            x_align_artifacts_dir=x_align_artifacts_dir,
            create_x_reference_if_missing=True,
            x_align_debug_path=x_align_debug_path,
            return_meta=True,
        )

        # Save aligned crop/debug output for review
        cv2.imwrite(template_path, crop_bgr)

        meta_path = os.path.join(single_template_dir, f"{name}_offset_xalign_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "side": SIDE_NAME,
                "raw_path": raw_path,
                "role": role,
                "sidewall_anchor_record": anchor_record,
                "non_r_crop_meta": crop_meta,
            }, f, indent=2, default=str)

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
def load_calibration_artifacts_from_dir(
    output_dir=None,
    ref_image_path_override=None,
    calibration_artifact_dir_override=None,
):
    if calibration_artifact_dir_override:
        art_dir = calibration_artifact_dir_override
    elif ref_image_path_override:
        art_dir = os.path.dirname(ref_image_path_override)
    else:
        output_dir = output_dir or OUTPUT_DIR
        calib_root = os.path.join(output_dir, CALIBRATION_DIR_NAME)
        art_dir = os.path.join(calib_root, "artifacts")

    bank_path = os.path.join(art_dir, "embedding_bank.pt")
    meta_path = os.path.join(art_dir, "embedding_bank_meta.pt")
    thr_path = os.path.join(art_dir, "thresholds_by_rc.pt")
    mahal_path = os.path.join(art_dir, "mahalanobis_stats.pt")
    pca_path = os.path.join(art_dir, "pca_artifact.pt")

    if not os.path.isfile(thr_path):
        raise RuntimeError(f"[{SIDE_NAME}] Missing thresholds: {thr_path}")

    if DISTANCE_METRIC in ["mahalanobis", "mahalanobis_pca"]:
        if not os.path.isfile(mahal_path):
            raise RuntimeError(f"[{SIDE_NAME}] Missing mahalanobis stats: {mahal_path}")

    if DISTANCE_METRIC == "mahalanobis_pca":
        if not os.path.isfile(pca_path):
            raise RuntimeError(f"[{SIDE_NAME}] Missing PCA artifact: {pca_path}")

    reference_bank = torch.load(bank_path, map_location="cpu") if os.path.isfile(bank_path) else {}
    reference_bank_meta = torch.load(meta_path, map_location="cpu") if os.path.isfile(meta_path) else {}
    thr_obj = torch.load(thr_path, map_location="cpu")
    mahalanobis_stats = torch.load(mahal_path, map_location="cpu") if os.path.isfile(mahal_path) else None
    pca_artifact = torch.load(pca_path, map_location="cpu") if os.path.isfile(pca_path) else None

    thresholds_by_rc = thr_obj["thresholds_by_rc"]
    mu_by_rc = thr_obj["mu_by_rc"]
    sigma_by_rc = thr_obj["sigma_by_rc"]

    print(f"[ARTIFACT] {SIDE_NAME}: loaded artifacts from: {art_dir}")

    return (
        reference_bank,
        reference_bank_meta,
        thresholds_by_rc,
        mu_by_rc,
        sigma_by_rc,
        mahalanobis_stats,
        pca_artifact,
        art_dir,
    )


def load_calibration_artifacts(calibration_artifact_dir_override=None):
    return load_calibration_artifacts_from_dir(
        OUTPUT_DIR,
        calibration_artifact_dir_override=calibration_artifact_dir_override,
    )

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
    calibration_artifact_dir_override=None,
):
    output_dir = output_dir_override or OUTPUT_DIR
    checkpoint_path = checkpoint_path_override or CHECKPOINT_PATH
    yolo_r_path = yolo_r_path_override or YOLO_R_PATH
    calibration_artifact_dir = calibration_artifact_dir_override

    os.makedirs(output_dir, exist_ok=True)

    if device is None:
        device = DEVICE

    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, using CPU")
        device = "cpu"

    # Tread does not need R detection.
    # It uses sidewall_r_anchor from sidewall1/sidewall2.
    r_detector = r_detector_override

    if r_detector is not None:
        print("[RUNTIME] tread received shared R-detector, but tread will not use it")
    else:
        print("[RUNTIME] tread does not require R-detector")

    if use_trt_vit and trt_vit is not None:
        model = trt_vit
        print("[RUNTIME] using TensorRT ViT engine")

    else:
        if checkpoint_path and str(checkpoint_path).lower().endswith(".engine"):
            raise RuntimeError(
                "[RUNTIME] checkpoint_path is a TensorRT .engine file, "
                "but trt_vit was not passed. Check cycle_engine.py TRT loading."
            )

        model = make_model().to(device).eval()

        if device == "cuda":
            model = model.half()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-4,
        )

        load_checkpoint(model, optimizer, checkpoint_path)
        print(f"[RUNTIME] PyTorch ViT checkpoint loaded: {checkpoint_path}")

    patch_transform = _build_transform()

    use_yolo_seg = (
        USE_YOLO_SEG
        if use_yolo_seg_override is None
        else bool(use_yolo_seg_override)
    )

    if seg_models is not None:
        runtime_seg_models = seg_models

    elif seg_model_override is not None:
        runtime_seg_models = {"default": seg_model_override}

    else:
        runtime_seg_models = {}

        if use_yolo_seg:
            try:
                runtime_seg_models["default"] = load_yolo_seg(
                    YOLO_SEG_MODEL_PATH,
                    device=device,
                )
                print("[YOLO] segmentation model loaded")
            except Exception as e:
                print(f"[YOLO][WARN] failed to load model: {e}")

    if load_artifacts:
        (
            reference_bank,
            reference_bank_meta,
            thresholds_by_rc,
            mu_by_rc,
            sigma_by_rc,
            mahalanobis_stats,
            pca_artifact,
            calibration_artifact_dir,
        ) = load_calibration_artifacts_from_dir(
            output_dir=output_dir,
            ref_image_path_override=ref_image_path_override,
            calibration_artifact_dir_override=calibration_artifact_dir_override,
        )
    else:
        reference_bank = {}
        reference_bank_meta = {}
        thresholds_by_rc = {}
        mu_by_rc = {}
        sigma_by_rc = {}
        mahalanobis_stats = None
        pca_artifact = None
        calibration_artifact_dir = calibration_artifact_dir_override

    return {
        "device": device,
        "model": model,
        "patch_transform": patch_transform,
        "r_detector": r_detector,
        "seg_models": runtime_seg_models,
        "use_yolo_seg": use_yolo_seg,

        "reference_bank": reference_bank,
        "reference_bank_meta": reference_bank_meta,
        "thresholds_by_rc": thresholds_by_rc,
        "mu_by_rc": mu_by_rc,
        "sigma_by_rc": sigma_by_rc,
        "mahalanobis_stats": mahalanobis_stats,
        "pca_artifact": pca_artifact,

        "output_dir": output_dir,
        "checkpoint_path": checkpoint_path,
        "yolo_r_path": yolo_r_path,
        "ref_image_path_override": ref_image_path_override,
        "tyre_name": tyre_name_override,

        "calibration_artifact_dir": calibration_artifact_dir,
        "x_align_artifacts_dir": calibration_artifact_dir,

        "use_trt_vit": bool(trt_vit is not None and use_trt_vit),
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
    sidewall_anchor_records=None,
    offset_ratio=None,
    x_align_artifacts_dir=None,
):
    calib_dir = calib_good_dir_override or CALIB_GOOD_DIR
    output_dir = output_dir_override or OUTPUT_DIR

    if x_align_artifacts_dir is None:
        x_align_artifacts_dir = os.path.join(
            output_dir,
            CALIBRATION_DIR_NAME,
            "artifacts",
        )

    return build_calibration_pipeline(
        runtime["model"],
        runtime.get("r_detector"),
        runtime["device"],
        gpu_sem=gpu_sem,
        calib_good_dir=calib_dir,
        output_dir=output_dir,
        ref_image_path=ref_image_path_override,
        sidewall_anchor_records=sidewall_anchor_records,
        offset_ratio=offset_ratio,
        x_align_artifacts_dir=x_align_artifacts_dir,
    )

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

        distance = float(best_dist)

        # =========================================================
        # EXTRA DEFECT DECISION SCORES
        # =========================================================
        z_score = (distance - mu) / sigma_eff

        if thr > 1e-9:
            score_ratio = distance / thr
        else:
            score_ratio = 0.0

        distance_pass = distance > thr
        z_score_pass = z_score >= Z_SCORE_THRESHOLD
        score_ratio_pass = score_ratio >= SCORE_RATIO_THRESHOLD

        is_defect = distance_pass

        if USE_Z_SCORE_FILTER:
            is_defect = is_defect and z_score_pass

        if USE_SCORE_RATIO_FILTER:
            is_defect = is_defect and score_ratio_pass

        row = {
            "filename": filename,
            "full_path": None,
            "r": int(r),
            "c": int(c),

            "distance": float(distance),
            "threshold_used": float(thr),
            "mu_used": float(mu),
            "sigma_used": float(sigma_eff),

            "z_score": float(z_score),
            "score_ratio": float(score_ratio),

            "distance_pass": bool(distance_pass),
            "z_score_pass": bool(z_score_pass),
            "score_ratio_pass": bool(score_ratio_pass),

            "z_score_threshold": float(Z_SCORE_THRESHOLD),
            "score_ratio_threshold": float(SCORE_RATIO_THRESHOLD),

            "classification": "DEFECT" if is_defect else "GOOD",
        }
        rows.append(row)
        
        visual_records.append({
            "r": int(r),
            "c": int(c),
            "patch": patch,
            "distance": float(distance),
            "threshold_used": float(thr),
            "z_score": float(z_score),
            "score_ratio": float(score_ratio),
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

        out_csv = os.path.join(save_dir, "patch_distance_results.csv")
        df.to_csv(out_csv, index=False)
        print(f"[SAVE] {out_csv}")

        raw_compare_csv = os.path.join(save_dir, "patch_all_reference_distances.csv")
        pd.DataFrame(rows).to_csv(raw_compare_csv, index=False)
        print(f"[SAVE] {raw_compare_csv}")
    
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