####CYCLE FLOW####
User clicks "Live"
        │
        ▼
start_live_inspection()
        │
        ├─ CAMERA mode ──→ multi_camera_manager.capture_all()
        │                        │
        │                   5 cameras in parallel
        │                        │
        │                   Save to Cycle_N/serial_XXXXX/image.png
        │
        ├─ DEMO mode ────→ Read images from existing folder
        │
        ▼
build_all_runtimes()  ──→  [CACHE HIT after first cycle]
        │
        ▼
_maybe_warmup_runtimes()  ──→  [SKIP after first cycle]
        │
        ▼
run_cycle()
        │
   For each side:
   infer_single_image()
        │
        ▼
combine_tire_decision()
        │
        ▼
Save CSV + JSON
        │
        ▼
Return result to GUI
        │
        ▼
Refresh image panels on screen




#####CAMERA_CAPTURE_ENABLED = False (Local folder / Demo mode)####

User clicks "Live"
        │
        ▼
start_live_inspection()
        │
        ├── cam_mgr = None
        └── demo_capture_root = LOCAL_MULTI_SIDE_TEST_FOLDER
                │
                ▼
        Reads images from existing folder:
        LOCAL_MULTI_SIDE_TEST_FOLDER/
        ├── serial_254901428/image.png   ← sidewall1
        ├── serial_254901430/image.png   ← tread
        ├── serial_254701292/image.png   ← innerwall
        ├── serial_254901432/image.png   ← sidewall2
        └── serial_254701283/image.png   ← bead
                │
                ▼
        Passes images to AI inference
        (no camera involved at all)



########CAMERA_CAPTURE_ENABLED = True (Real camera mode)#####
User clicks "Live"
        │
        ▼
start_live_inspection()
        │
        ├── cam_mgr = self.multi_cam
        └── demo_capture_root = None
                │
                ▼
        Captures from 5 real cameras
        Saves to media/capture/23-04-2026/Cycle_N/
                │
                ▼
        Passes saved images to AI inference

#####So the safe combinations are:
deployment  CAMERA_CAPTURE_ENABLED            Result
False            False                      Works — uses local folder
True             True                       Works — uses real cameras
False            True                       Crashes — no cameras but trying to capture
True             False                      Works — cameras connected but uses local folde



#########Full flow diagram:
APP START
   |
   v
GUI loads
   |
   +--> Load UI, sidebar, dashboard, image panels
   |
   +--> Load startup placeholder images
   |
   +--> Load global env / paths / device info
   |
   +--> Load common model path references
   |
   +--> No SKU warmup yet
   |
   v
User clicks LIVE button
   |
   v
Popup opens
   |
   +--> User selects SKU
   |
   +--> User enters Tyre Number
   |
   v
GUI updates dashboard cards
   |
   +--> Selected SKU card updated
   |
   +--> Tyre Number card updated
   |
   v
Check:
"Is this SKU already preloaded and warmed?"
   |
   +--> YES
   |      |
   |      v
   |   Start live inspection immediately
   |
   +--> NO
          |
          v
      Start PRELOAD thread
          |
          v
      PRELOAD FLOW
          |
          +--> Build / reuse segmentation models
          |
          +--> Build / reuse shared R detector
          |
          +--> Load selected SKU artifacts
          |       - alignment_reference_polarized.png
          |       - thresholds
          |       - embedding bank
          |       - mahalanobis stats
          |       - PCA artifact
          |       - reference_r (where needed)
          |
          +--> Build runtimes for:
          |       - innerwall
          |       - sidewall1
          |       - sidewall2
          |       - tread
          |       - bead
          |
          +--> Warmup all runtimes once
          |
          v
      Mark SKU as READY
          |
          v
      Start live inspection automatically


####Live inspection flow:
LIVE INSPECTION START
   |
   v
Create LiveInspectionWorker thread
   |
   v
run_capture_folder_cycle(...)
   |
   v
Resolve selected SKU + tyre number
   |
   +--> Apply tyre number into runtimes
   |       (without rebuilding runtimes)
   |
   v
Check mode
   |
   +--> CAMERA_CAPTURE_ENABLED = True
   |       |
   |       v
   |   Capture fresh images from cameras
   |
   +--> CAMERA_CAPTURE_ENABLED = False
           |
           v
       Use demo/local images from capture folder


########Per-cycle processing flow:
Build image_map for all sides
   |
   v
For each side:
   - innerwall
   - sidewall1
   - sidewall2
   - tread
   - bead
   |
   v
infer_single_image(...)
   |
   +--> Read image
   +--> Polarizer preprocessing
   +--> R detect / align / crop
   +--> Patchify
   +--> ViT embedding comparison
   +--> Threshold decision
   +--> YOLO on selected defect patches
   +--> Save outputs
   |
   v
Side result generated


##########After all sides finish:
All side results collected
   |
   v
Combine final tyre decision
   |
   +--> If any side DEFECT --> final tyre DEFECT
   +--> Else if SUSPECT --> SUSPECT
   +--> Else --> OK
   |
   v
Save cycle outputs
   |
   +--> output/Cycle_N/
   +--> sidewise final images
   +--> CSV / JSON summary
   +--> SKU name
   +--> tyre number
   +--> final decision
   |
   v
Return result to GUI


#############GUI update after cycle:
Live worker finished
   |
   v
GUI status bar updated
   |
   v
Refresh latest cycle images
   |
   +--> Read newest final_stitched.png for:
   |       - sidewall1
   |       - sidewall2
   |       - innerwall
   |       - tread
   |
   v
Dashboard panels updated dynamically


######Important reuse behavior
#########Same SKU again:

User clicks LIVE
   |
   v
Select same SKU as before
   |
   v
SKU already warmed
   |
   v
Skip preload
   |
   v
Start inspection directly

########Different SKU:

User clicks LIVE
   |
   v
Select new SKU
   |
   v
SKU not warmed yet
   |
   v
Run preload + warmup once for that SKU
   |
   v
Start inspection



###########Different tyre number, same SKU:
User changes only tyre number
   |
   v
No new warmup
   |
   v
Reuse same SKU runtimes
   |
   v
Only tyre metadata/dimension input changes


###########Final architecture in one compact diagram:

APP START
   |
   v
GUI READY
   |
   v
LIVE CLICK
   |
   v
SKU + TYRE POPUP
   |
   v
Check SKU warmed?
   |
   +--> Yes --> Start Cycle
   |
   +--> No  --> Preload SKU --> Warmup --> Start Cycle
                                   |
                                   v
                              Cache SKU runtimes

##Overall file responsibility

GUI.py
  |
  | calls
  v
maincycle.py
  |
  | controls only the cycle flow
  v
src/COMMON/cycle_engine.py
  |
  | does heavy work:
  | capture helpers
  | image map
  | runtime/model loading
  | warmup
  | per-side inference
  | final result saving

##maincycle.py flow:

run_capture_folder_cycle()
        |
        v
1. Resolve selected sides
        |
        v
2. Validate model paths
        |
        v
3. Validate SKU calibration/artifacts folder
        |
        v
4. Resolve cycle capture folder
        |
        v
5. Build image_map
        |
        v
6. Load or reuse runtimes
        |
        v
7. Create GPU semaphores
        |
        v
8. Run inference cycle
        |
        v
9. Save cycle metadata to MongoDB
        |
        v
10. Return result to GUI

####cycle_engine.py flow:

run_cycle()
        |
        v
create output cycle folder
        |
        v
run all selected sides
        |
        v
for each side:
    alignment
    crop save
    ViT/template matching
    YOLO on defect patches
    final result
        |
        v
combine tire decision
        |
        v
save side_results.csv
        |
        v
save tire_summary.json
        |
        v
return result


####Per-side cycle flow inside cycle_engine.py:
Per-side detailed flow:

_run_one_side_infer()
        |
        v
Check if side module supports stage pipeline
        |
        v
run_side_pipeline()


####run_side_pipeline() flow:


run_side_pipeline(side_name, image_path, runtime, cycle_dir, ...)
        |
        v
1. Create side output folders
        |
        v
2. Read and polarize image
        |
        v
3. Align and crop
        |
        v
4. Save crop image
        |
        v
5. Patchify crop
        |
        v
6. Extract ViT embeddings
        |
        v
7. Template matching / ViT classification
        |
        v
8. Save template_stitched image
        |
        v
9. If ViT defects found, run YOLO
        |
        v
10. Save final_stitched image
        |
        v
11. Return side result

###3Stage 1: Alignment and crop:

raw image
   |
   v
read_and_polarize()
   |
   v
align_crop_from_preprocessed()
   |
   v
crop.png

For these sides:

innerwall
tread
bead


####3Stage 2: ViT/template matching:
crop.png
   |
   v
patchify_array_indexed()
   |
   v
get_patch_embeddings_from_arrays()
   |
   v
process_precomputed_embeddings()
   |
   v
template_stitched.png


Output:
   GOOD patches
   DEFECT patches
   template_stitched image

###Stage 3: YOLO on only ViT defect patches:
ViT DEFECT patches only
   |
   v
run_yolo_on_vit_defect_patches()
   |
   v
final_stitched.png

Decision logic:

If YOLO detections > 0
    side result = DEFECT

If ViT defect patches > 0 but YOLO detections = 0
    side result = SUSPECT

If ViT defect patches = 0
    side result = OK

###Final tire decision::
If any side = DEFECT
    tire = DEFECT

Else if any side = SUSPECT
    tire = SUSPECT

Else if any side = INVALID / FAILED
    tire = INVALID

Else
    tire = OK



##########Full combined flow##############
GUI.py
  |
  v
maincycle.run_capture_folder_cycle()
  |
  v
Resolve sides
  |
  v
Validate model paths
  |
  v
Validate SKU calibration/artifacts
  |
  v
Resolve capture folder
  |
  v
Build image_map
  |
  v
Load/reuse runtimes
  |
  v
Warmup if not already warmed
  |
  v
Create GPU semaphores
  |
  v
cycle_engine.run_cycle()
  |
  v
Run sides in parallel
  |
  +--> sidewall1: align -> crop -> ViT -> YOLO -> result
  |
  +--> sidewall2: align -> crop -> ViT -> YOLO -> result
  |
  +--> innerwall: align -> crop -> ViT -> YOLO -> result
  |
  +--> tread: align -> crop -> ViT -> YOLO -> result
  |
  +--> bead: align -> crop -> ViT -> YOLO -> result
  |
  v
Combine side results
  |
  v
Save side_results.csv
  |
  v
Save tire_summary.json
  |
  v
Return result to maincycle.py
  |
  v
Save cycle metadata to MongoDB
  |
  v
Return result to GUI
  |
  v
GUI displays latest final_stitched images