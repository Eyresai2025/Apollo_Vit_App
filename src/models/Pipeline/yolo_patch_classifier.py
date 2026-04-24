from ultralytics import YOLO
import cv2
import numpy as np

# =========================================================
# LOAD MODEL (Supports .pt, .onnx, .engine)
# =========================================================
def load_yolo_seg(model_path: str, device: str = "cuda", imgsz: int = 224):
    """
    Load YOLO segmentation model.
    Supports: .pt (PyTorch), .onnx (ONNX Runtime), .engine (TensorRT)
    """
    model = YOLO(model_path)
    
    # Store imgsz for later use (Ultralytics uses it automatically)
    model.imgsz = imgsz
    
    print(f"[YOLO SEG] loaded: {model_path}")
    print(f"[YOLO SEG] device: {device}")
    print(f"[YOLO SEG] imgsz: {imgsz}")
    
    return model


# =========================================================
# MAIN INFERENCE
# =========================================================
def segment_patch_paths(
    model,
    img_paths,
    conf_threshold: float = 0.5,
    max_batch_size: int = 32,
    iou_threshold: float = 0.45,
):
    """
    Run segmentation on a list of image paths.
    
    Returns dict:
        {
            path: {
                "cls_ids": [...],
                "cls_names": [...],
                "confs": [...],
                "overlay": overlay_bgr,
                "boxes_xyxy": [...],
            }
        }
    """
    if not img_paths:
        return {}

    out = {}

    # Process in batches
    for start in range(0, len(img_paths), max_batch_size):
        batch_paths = img_paths[start:start + max_batch_size]

        # Ultralytics automatically uses TensorRT/ONNX with half=True if engine/FP16
        results = model(batch_paths, verbose=False, conf=conf_threshold, iou=iou_threshold)

        for path, res in zip(batch_paths, results):
            boxes = getattr(res, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue

            img = cv2.imread(path)
            if img is None:
                continue

            cls_ids = boxes.cls.int().cpu().numpy().tolist()
            conf_vals = boxes.conf.float().cpu().numpy().tolist()
            cls_names = [model.names[int(cid)] for cid in cls_ids]
            boxes_xyxy = boxes.xyxy.cpu().numpy().tolist()

            # Create overlay
            overlay_bgr = img.copy()
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]

            for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
                color = colors[int(cls_ids[i]) % len(colors)]
                cv2.rectangle(overlay_bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"{cls_names[i]} {conf_vals[i]:.2f}"
                cv2.putText(overlay_bgr, label, (int(x1), max(int(y1) - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

                # If masks are available, overlay them
                if hasattr(res, "masks") and res.masks is not None:
                    mask = res.masks.data[i].cpu().numpy().squeeze()
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
                    mask_bin = (mask > 0.5).astype(np.uint8)
                    mask_color = np.zeros_like(overlay_bgr)
                    mask_color[mask_bin > 0] = color
                    overlay_bgr = cv2.addWeighted(overlay_bgr, 1.0, mask_color, 0.35, 0)

            out[path] = {
                "cls_ids": cls_ids,
                "cls_names": cls_names,
                "confs": conf_vals,
                "overlay": overlay_bgr,
                "boxes_xyxy": boxes_xyxy,
            }

    return out