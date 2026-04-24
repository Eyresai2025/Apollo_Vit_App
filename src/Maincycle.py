import os
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.COMMON.db import save_cycle_metadata

from src.COMMON.cycle_engine import (
    DEVICE,
    CAMERA_CAPTURE_ENABLED,

    R_ALIGN_GPU_CONCURRENCY,
    VIT_GPU_CONCURRENCY,
    YOLO_GPU_CONCURRENCY,

    _normalize_device,
    _resolve_sides,
    _required_file,
    _get_sku_calibration_dir,
    _get_sku_artifacts_dir,
    _get_today_capture_root,

    build_cycle_capture_dir,
    capture_and_save_images,
    build_image_map_from_capture_dir,

    build_all_runtimes,
    _apply_tyre_name_to_runtimes,
    _maybe_warmup_runtimes,

    run_cycle,
    preload_live_runtimes,
)


def resolve_cycle_capture_dir(
    media_root: str,
    cycle_id: Optional[str],
    demo_capture_root: Optional[str],
) -> tuple[str, str]:
    """
    Decides which capture folder to use.

    Camera mode:
        creates new media/capture/<date>/Cycle_N/

    Demo/local mode:
        uses demo_capture_root if given,
        otherwise uses latest Cycle_N from today's capture folder.
    """

    if CAMERA_CAPTURE_ENABLED:
        cycle_capture_dir, cycle_id = build_cycle_capture_dir(media_root)
        return cycle_capture_dir, cycle_id

    if demo_capture_root:
        cycle_capture_dir = os.path.abspath(demo_capture_root)

        if cycle_id is None:
            cycle_id = f"Cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return cycle_capture_dir, cycle_id

    today_root = _get_today_capture_root(media_root)

    existing = [
        d for d in os.listdir(today_root)
        if os.path.isdir(os.path.join(today_root, d)) and d.startswith("Cycle_")
    ]

    if not existing:
        raise FileNotFoundError(
            f"No Cycle_N folders found under {today_root}. "
            "Place images there or enable CAMERA_CAPTURE_ENABLED."
        )

    existing.sort(key=lambda d: int(d.split("_", 1)[1]))
    latest_cycle = existing[-1]

    cycle_capture_dir = os.path.join(today_root, latest_cycle)
    cycle_id = cycle_id or latest_cycle

    return cycle_capture_dir, cycle_id


def build_cycle_image_map(
    cycle_capture_dir: str,
    sides_to_run: List[str],
    multi_camera_manager=None,
) -> Dict[str, str]:
    """
    Creates image_map for the cycle.

    Output:
        {
            "sidewall1": "...image path...",
            "sidewall2": "...image path...",
            "innerwall": "...image path...",
            "tread": "...image path...",
            "bead": "...image path...",
        }
    """

    if CAMERA_CAPTURE_ENABLED:
        if multi_camera_manager is None:
            raise ValueError(
                "CAMERA_CAPTURE_ENABLED=True but multi_camera_manager was not passed."
            )

        return capture_and_save_images(
            multi_camera_manager=multi_camera_manager,
            cycle_capture_dir=cycle_capture_dir,
            sides_to_run=sides_to_run,
        )

    return build_image_map_from_capture_dir(
        cycle_capture_dir=cycle_capture_dir,
        sides_to_run=sides_to_run,
    )


def prepare_runtimes_for_cycle(
    sku_name: str,
    media_root: str,
    cycle_capture_dir: str,
    device: str,
    seg_model_a_path: str,
    seg_model_b_path: str,
    vit_checkpoint_path: str,
    r_detector_path: str,
    tyre_name: str,
    side_configs: Optional[Dict[str, Dict[str, Any]]],
    sides_to_run: List[str],
):
    """
    Builds or reuses loaded models/runtimes.
    Warmup also happens here.
    """

    runtimes = build_all_runtimes(
        sku_name=sku_name,
        media_root=media_root,
        seg_model_a_path=seg_model_a_path,
        seg_model_b_path=seg_model_b_path,
        vit_checkpoint_path=vit_checkpoint_path,
        r_detector_path=r_detector_path,
        device=device,
        capture_root=cycle_capture_dir,
        tyre_name=tyre_name,
        side_configs=side_configs,
        sides_to_run=sides_to_run,
    )

    _apply_tyre_name_to_runtimes(runtimes, tyre_name)

    _maybe_warmup_runtimes(
        runtimes=runtimes,
        sku_name=sku_name,
        device=device,
        capture_root=cycle_capture_dir,
        seg_model_a_path=seg_model_a_path,
        seg_model_b_path=seg_model_b_path,
        vit_checkpoint_path=vit_checkpoint_path,
        r_detector_path=r_detector_path,
        tyre_name=tyre_name,
        media_root=media_root,
        sides_to_run=sides_to_run,
    )

    return runtimes


def build_gpu_semaphores():
    """
    Controls GPU concurrency for R alignment, ViT, and YOLO.
    """

    r_gpu_sem = threading.Semaphore(R_ALIGN_GPU_CONCURRENCY)
    vit_gpu_sem = threading.Semaphore(VIT_GPU_CONCURRENCY)
    yolo_gpu_sem = threading.Semaphore(YOLO_GPU_CONCURRENCY)

    return r_gpu_sem, vit_gpu_sem, yolo_gpu_sem


def print_cycle_inputs(
    sku_name: str,
    tyre_name: str,
    sku_calibration_dir: str,
    shared_artifacts_dir: str,
    cycle_capture_dir: str,
    cycle_id: str,
    image_map: Dict[str, str],
    sides_to_run: List[str],
):
    print(f"[MAIN] selected sku_name     : {sku_name}")
    print(f"[MAIN] selected tyre_name    : {tyre_name}")
    print(f"[MAIN] sku_calibration_dir   : {sku_calibration_dir}")
    print(f"[MAIN] sku_artifacts_dir     : {shared_artifacts_dir}")
    print(f"[MAIN] cycle_capture_dir     : {cycle_capture_dir}")
    print(f"[MAIN] cycle_id              : {cycle_id}")

    print("[MAIN] image_map:")
    for side_name in sides_to_run:
        print(f"    {side_name}: {image_map.get(side_name, 'MISSING')}")


def run_capture_folder_cycle(
    media_root: str,
    sku_name: str = "SKU_001",
    cycle_id: Optional[str] = None,
    device: str = DEVICE,
    seg_model_a_path: Optional[str] = None,
    seg_model_b_path: Optional[str] = None,
    vit_checkpoint_path: Optional[str] = None,
    r_detector_path: Optional[str] = None,
    tyre_name: str = "195_65_R15",
    side_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    sides_to_run: Optional[List[str]] = None,
    multi_camera_manager=None,
    demo_capture_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Clean cycle flow only.

    Flow:
        1. Resolve sides
        2. Validate model paths
        3. Validate SKU folders
        4. Resolve capture folder
        5. Build image map
        6. Load/reuse runtimes
        7. Build GPU semaphores
        8. Run inference cycle
        9. Save DB metadata
        10. Return result
    """

    # -------------------------------------------------
    # 1. Basic setup
    # -------------------------------------------------
    sides_to_run = _resolve_sides(sides_to_run)
    media_root = os.path.abspath(media_root)
    device = _normalize_device(device)

    os.makedirs(media_root, exist_ok=True)

    # -------------------------------------------------
    # 2. Validate model paths
    # -------------------------------------------------
    seg_model_a_path = _required_file(seg_model_a_path, "seg_model_a_path")
    seg_model_b_path = _required_file(seg_model_b_path, "seg_model_b_path")
    vit_checkpoint_path = _required_file(vit_checkpoint_path, "vit_checkpoint_path")
    r_detector_path = _required_file(r_detector_path, "r_detector_path")

    # -------------------------------------------------
    # 3. Validate SKU folders
    # -------------------------------------------------
    sku_calibration_dir = _get_sku_calibration_dir(media_root, sku_name)
    shared_artifacts_dir = _get_sku_artifacts_dir(media_root, sku_name)

    # -------------------------------------------------
    # 4. Resolve capture folder
    # -------------------------------------------------
    cycle_capture_dir, cycle_id = resolve_cycle_capture_dir(
        media_root=media_root,
        cycle_id=cycle_id,
        demo_capture_root=demo_capture_root,
    )

    # -------------------------------------------------
    # 5. Build image map
    # -------------------------------------------------
    image_map = build_cycle_image_map(
        cycle_capture_dir=cycle_capture_dir,
        sides_to_run=sides_to_run,
        multi_camera_manager=multi_camera_manager,
    )

    print_cycle_inputs(
        sku_name=sku_name,
        tyre_name=tyre_name,
        sku_calibration_dir=sku_calibration_dir,
        shared_artifacts_dir=shared_artifacts_dir,
        cycle_capture_dir=cycle_capture_dir,
        cycle_id=cycle_id,
        image_map=image_map,
        sides_to_run=sides_to_run,
    )

    # -------------------------------------------------
    # 6. Load / reuse runtimes
    # -------------------------------------------------
    runtimes = prepare_runtimes_for_cycle(
        sku_name=sku_name,
        media_root=media_root,
        cycle_capture_dir=cycle_capture_dir,
        device=device,
        seg_model_a_path=seg_model_a_path,
        seg_model_b_path=seg_model_b_path,
        vit_checkpoint_path=vit_checkpoint_path,
        r_detector_path=r_detector_path,
        tyre_name=tyre_name,
        side_configs=side_configs,
        sides_to_run=sides_to_run,
    )

    # -------------------------------------------------
    # 7. GPU semaphores
    # -------------------------------------------------
    r_gpu_sem, vit_gpu_sem, yolo_gpu_sem = build_gpu_semaphores()

    # -------------------------------------------------
    # 8. Run actual inference cycle
    # -------------------------------------------------
    output_root = os.path.join(media_root, "output")
    os.makedirs(output_root, exist_ok=True)

    result = run_cycle(
        image_map=image_map,
        runtimes=runtimes,
        output_root=output_root,
        cycle_id=cycle_id,
        sides_to_run=sides_to_run,
        r_gpu_sem=r_gpu_sem,
        vit_gpu_sem=vit_gpu_sem,
        yolo_gpu_sem=yolo_gpu_sem,
        sku_name=sku_name,
        tyre_name=tyre_name,
    )

    # -------------------------------------------------
    # 9. Save metadata to DB
    # -------------------------------------------------
    try:
        save_cycle_metadata(result)
        print(f"[DB] saved cycle metadata | cycle_id={result.get('cycle_id')}")
    except Exception as e:
        print(f"[DB][ERROR] save failed | error={e}")

    return result

# Keep this alias if your GUI imports run_cycle_for_gui
run_cycle_for_gui = run_capture_folder_cycle