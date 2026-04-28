## Deployment Modes / Capture Flow

| Mode | DEPLOYMENT (.env) | CAMERA_CAPTURE_ENABLED (`cycle_engine.py`) | TRIGGER_MODE (`HARDWARE_TRIGGER.py`) | Behavior |
|--------|-------------------|--------------------------------------------|--------------------------------------|-----------|
| **1. Production - PLC Trigger** | `True` | `True` | `software` | Cameras run in continuous stream mode. PLC tag (`DB100.DBX0.0`) triggers image capture. Once capture is completed, the AI pipeline starts processing. |
| **2. Production - Hardware Trigger** | `True` | `True` | `hardware` | Cameras wait for a physical trigger signal on `Line1`. Images are captured only when the hardware trigger is received, then the AI pipeline runs. |
| **3. Production - Image Folder Mode** | `True` | `False` | `N/A` | No live camera usage. System picks pre-captured images from a configured folder and runs the AI pipeline. Useful when cameras are unavailable but production flow is required. |
| **4. Demo / Local Testing Mode** | `False` | `False` | `N/A` | No PLC connection and no cameras. System loads images from `LOCAL_MULTI_SIDE_TEST_FOLDER` for local testing/demo purposes. |


Flow for Each Mode:

Mode 1 & 2: Production with Cameras (deployment="True", CAMERA_CAPTURE_ENABLED=True):
User clicks "Live" button
         │
         ▼
GUI.py: open_live_selection_dialog()
  → User selects SKU + Tyre name
  → Calls begin_live_flow(sku, tyre)
         │
         ▼
GUI.py: begin_live_flow()
  → Checks: deployment=="True" AND CAMERA_CAPTURE_ENABLED==True? YES
  → Calls start_continuous_inspection()
         │
         ▼
GUI.py: start_continuous_inspection()
  → Creates ContinuousCycleWorker via Main_cam.py
  → Connects signals (status_update, processing_completed, etc.)
  → Starts worker in QThread via thread_manager
         │
         ▼
Main_cam.py: ContinuousCycleWorker.run()
  → Preloads AI models
  → Starts camera streams via HARDWARE_TRIGGER.py
  → Enters MAIN LOOP:
         │
         ├── IF TRIGGER_MODE=="software":
         │     → Reads PLC tag DB100.DBX0.0 every 10ms
         │     → Detects rising edge (LOW→HIGH)
         │     → Triggers capture
         │
         └── IF TRIGGER_MODE=="hardware":
               → Calls capture_all() immediately
               → Cameras internally wait for hardware trigger on Line1
               → Each camera captures when signal arrives
         │
         ▼
HARDWARE_TRIGGER.py: MultiCameraManager.capture_all()
  → 5 cameras capture in parallel (ThreadPoolExecutor)
  → Each camera stitches 42000 rows
  → Returns {serial: numpy_array}
         │
         ▼
Main_cam.py: _execute_capture()
  → Saves images to cycle directory (media/capture/date/Cycle_N/)
  → Maps camera serials to side names
  → Calls _run_ai_pipeline()
         │
         ▼
cycle_engine.py: run_cycle()
  → Processes 5 sides in parallel
  → Stage 1: Alignment (R-detector)
  → Stage 2: ViT/Template matching
  → Stage 3: YOLO on defect patches
  → Returns result {final_label, side_results, etc.}
         │
         ▼
GUI.py: _on_continuous_completed()
  → Updates status bar: "✅ Cycle_X | Result: OK/DEFECT"
  → Updates tyre count
  → Refreshes image panels
  → Worker goes back to waiting for next trigger




Mode 3: Production without Cameras (deployment="True", CAMERA_CAPTURE_ENABLED=False):

User clicks "Live"
         │
         ▼
GUI.py: begin_live_flow()
  → Checks: deployment=="True" AND CAMERA_CAPTURE_ENABLED==True? NO
  → Falls to DEMO MODE path (same as Mode 4)
         │
         ▼
Uses LiveInspectionWorker (single cycle)
  → Reads images from folder
  → Runs AI pipeline once
  → Shows results


Mode 4: Demo/Local (deployment="False", CAMERA_CAPTURE_ENABLED=False):

User clicks "Live"
         │
         ▼
GUI.py: begin_live_flow()
  → Checks: deployment=="True"? NO
  → Falls to DEMO MODE path
         │
         ▼
start_live_inspection()
  → CAMERA_CAPTURE_ENABLED? False
  → Uses demo_capture_root = LOCAL_MULTI_SIDE_TEST_FOLDER
  → Creates LiveInspectionWorker
         │
         ▼
Main_cam.py: run_capture_folder_cycle()
  → demo_capture_root is not None
  → Reads images from:
      C:\Users\...\DEMO_CYCLE_FOLDER\
        ├── serial_254701283/  → bead
        ├── serial_254701292/  → innerwall
        ├── serial_254901428/  → sidewall1
        ├── serial_254901430/  → tread
        └── serial_254901432/  → sidewall2
  → Runs AI pipeline
  → Shows results