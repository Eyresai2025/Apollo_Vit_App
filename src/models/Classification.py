import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

def detect_and_annotate_image(img_path, model_path_class, slice_height=1630, slice_width=1024):
 
    # Initialize detection model inside the function
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_path_class,
        confidence_threshold=0.2,
        device="cuda"
    )

    result = get_sliced_prediction(
        image=img_path,
        detection_model=detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=0,
        overlap_width_ratio=0
    )

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    labels = []
    for prediction in result.object_prediction_list:
        bbox = prediction.bbox.to_voc_bbox()  # x1, y1, x2, y2
        category_name = prediction.category.name
        score = prediction.score.value
        x1, y1, x2, y2 = map(int, bbox)
        labels.append((category_name, score, (x1, y1, x2, y2)))

        # Draw bounding box and label
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{category_name} ({score:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return img, labels