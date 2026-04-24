import cv2
import numpy as np
import os
from glob import glob

def polarizer(img, gamma: float = 1.5):
    img = img.astype(np.float32)
    illumination = cv2.GaussianBlur(img, (0, 0), sigmaX=35, sigmaY=35)
    illumination += 1
    normalized = img / illumination
    normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    inv_gamma = 1.0 / gamma
    table = (np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)])).astype(
        np.uint8
    )
    result = cv2.LUT(normalized, table)
    return result

 