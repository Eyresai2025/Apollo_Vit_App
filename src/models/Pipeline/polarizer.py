import cv2
import numpy as np
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_KERNEL1D_CACHE = {}
_LUT_CACHE = {}


def _get_lut(gamma: float):
    key = float(gamma)
    if key not in _LUT_CACHE:
        inv_gamma = 1.0 / gamma
        _LUT_CACHE[key] = torch.tensor(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
            dtype=torch.uint8,
            device=device
        )
    return _LUT_CACHE[key]


def _get_gaussian_kernels_1d(sigma: float, dtype=torch.float16):
    key = (float(sigma), str(dtype))
    if key not in _KERNEL1D_CACHE:
        k = int(6 * sigma + 1)
        if k % 2 == 0:
            k += 1

        x = torch.arange(k, device=device, dtype=torch.float32) - k // 2
        g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        g = g / g.sum()

        g = g.to(dtype)

        kernel_h = g.view(1, 1, 1, k)   # horizontal
        kernel_v = g.view(1, 1, k, 1)   # vertical
        pad = k // 2

        _KERNEL1D_CACHE[key] = (kernel_h, kernel_v, pad)

    return _KERNEL1D_CACHE[key]


@torch.no_grad()
def polarizer_optimized(img, gamma: float = 1.5, sigma: float = 35, downsample: int = 4):
    if img is None:
        raise ValueError("Input image is None")

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    img = np.ascontiguousarray(img)
    H, W = img.shape[:2]

    # keep original in float32 for stable division
    img_t = torch.from_numpy(img).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # low-frequency illumination can be estimated on smaller image
    if downsample > 1:
        small_h = max(1, H // downsample)
        small_w = max(1, W // downsample)
        img_small = F.interpolate(img_t, size=(small_h, small_w), mode="bilinear", align_corners=False)
    else:
        img_small = img_t

    # use fp16 for blur stage
    blur_in = img_small.to(torch.float16)
    kernel_h, kernel_v, pad = _get_gaussian_kernels_1d(sigma / downsample if downsample > 1 else sigma, dtype=torch.float16)

    # separable Gaussian blur
    x = F.pad(blur_in, (pad, pad, 0, 0), mode="reflect")
    x = F.conv2d(x, kernel_h)

    x = F.pad(x, (0, 0, pad, pad), mode="reflect")
    illumination_small = F.conv2d(x, kernel_v)

    illumination_small = illumination_small.to(torch.float32)

    if downsample > 1:
        illumination = F.interpolate(illumination_small, size=(H, W), mode="bilinear", align_corners=False)
    else:
        illumination = illumination_small

    illumination = illumination + 1.0

    normalized = img_t / illumination

    min_val = normalized.amin()
    max_val = normalized.amax()
    normalized = (normalized - min_val) / (max_val - min_val + 1e-6)
    normalized = normalized * 255.0

    normalized_uint8 = normalized.clamp(0, 255).to(torch.uint8)
    lut = _get_lut(gamma)
    result = lut[normalized_uint8.long()]

    return result.squeeze().cpu().numpy()