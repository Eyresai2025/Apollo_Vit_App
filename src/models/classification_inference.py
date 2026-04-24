from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2
import warnings
import numpy as np
import threading

warnings.filterwarnings("ignore")


class CrackDetector:
    def __init__(
        self,
        model_path,
        device="cuda",
        conf=0.5,
        slice_height=1630,
        slice_width=4096,
        overlap_height_ratio=0.1,
        overlap_width_ratio=0.05,
    ):
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio

        self.model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=model_path,
            confidence_threshold=conf,
            device=device,
        )

        # important for shared model access in multi-thread live loop
        self.lock = threading.Lock()

    def _prepare_image(self, image):
        if image is None:
            raise ValueError("Input image is None")

        # uint16 -> uint8
        if image.dtype == np.uint16:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # grayscale -> BGR
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        return image

    def infer(self, image):
        image = self._prepare_image(image)

        # safest when one model is shared across threads
        with self.lock:
            result = get_sliced_prediction(
                image,
                self.model,
                slice_height=self.slice_height,
                slice_width=self.slice_width,
                overlap_height_ratio=self.overlap_height_ratio,
                overlap_width_ratio=self.overlap_width_ratio,
            )

        print(f"📦 Detections: {len(result.object_prediction_list)}")

        result_image = image.copy()
        detections = []

        for obj in result.object_prediction_list:
            bbox = obj.bbox.to_xyxy()
            score = obj.score.value
            category = obj.category.name

            x1, y1, x2, y2 = map(int, bbox)

            detections.append({
                "label": category,
                "score": score,
                "bbox": [x1, y1, x2, y2]
            })

            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 0, 255), 3)

            label = f"{category}: {score:.2f}"
            cv2.putText(
                result_image,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        return result_image, detections