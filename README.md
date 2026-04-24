<div>
  <img src="media/img/Eyres.jpeg" alt="Eyres Logo" width="200" align="left"/>
  <img src="media/img/Apollo.png" alt="Apollo Logo" width="300" align="right"/>
</div>

<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>

# EyresQC+ Apollo VIT App

## Overview

EyresQC+ Apollo VIT App is a PyQt-based tire inspection application developed for Apollo tire quality inspection.

The application provides a graphical interface to run the SmartQC+ inspection cycle, select SKU / tyre nomenclature, capture or load tyre images, run the AI inspection pipeline, and display the final inspection result for each tyre side.

The current project includes:

- Apollo GUI application
- SmartQC+ live inspection cycle
- Multi-side tyre inspection pipeline
- ViT/template-matching based patch inspection
- YOLO segmentation based defect confirmation
- R-detection based alignment and crop flow
- MongoDB cycle metadata saving
- Dynamic output folder generation
- Side-wise result visualization in GUI

---

## Main Inspection Sides

The inspection flow currently supports these tyre sides:

~~~text
innerwall
sidewall1
sidewall2
tread
bead
~~~

Each side runs through its own AI pipeline and generates side-wise output.

---

## Project Flow

~~~text
GUI.py
  |
  v
src/Maincycle.py
  |
  v
src/COMMON/cycle_engine.py
  |
  v
Capture / Load images
  |
  v
Build image map
  |
  v
Load cached runtimes
  |
  v
Run side-wise inspection
  |
  v
Generate final stitched outputs
  |
  v
Save cycle result
  |
  v
Save metadata to MongoDB
  |
  v
Display result in GUI
~~~

---

## AI Inspection Pipeline

For each tyre side, the pipeline follows this process:

~~~text
Input image
  |
  v
Preprocessing / Polarizer
  |
  v
R detection / alignment
  |
  v
Crop generation
  |
  v
Patch generation
  |
  v
ViT feature extraction
  |
  v
Template matching / patch classification
  |
  v
YOLO segmentation on selected defect patches
  |
  v
Final stitched result
  |
  v
Side decision
~~~

---

## Output Folder Structure

For every inspection cycle, output is saved in:

~~~text
media/output/Cycle_N/
~~~

Example:

~~~text
media/output/Cycle_1/
  side_results.csv
  tire_summary.json

  innerwall/
    crop/
      crop.png
    final/
      template_stitched.png
      final_stitched.png

  sidewall1/
    crop/
      crop.png
    final/
      template_stitched.png
      final_stitched.png

  sidewall2/
    crop/
      crop.png
    final/
      template_stitched.png
      final_stitched.png

  tread/
    crop/
      crop.png
    final/
      template_stitched.png
      final_stitched.png

  bead/
    crop/
      crop.png
    final/
      template_stitched.png
      final_stitched.png
~~~

---

## Important Project Files

### `GUI.py`

Main PyQt GUI file used to launch the Apollo SmartQC+ application.

### `src/Maincycle.py`

Main cycle controller file.

This file keeps the high-level cycle flow:

- Resolve sides
- Validate model paths
- Validate SKU folders
- Resolve capture folder
- Build image map
- Load/reuse runtimes
- Run inference cycle
- Save metadata to database
- Return result to GUI

### `src/COMMON/cycle_engine.py`

Heavy cycle engine file.

This file contains:

- Runtime loading
- Model cache
- Warmup logic
- Image map helpers
- Camera/capture helpers
- Per-side inference
- ViT/template matching flow
- YOLO defect confirmation flow
- CSV and JSON result saving

### `src/COMMON/db.py`

MongoDB helper file used to save cycle metadata.

### `src/COMMON/common.py`

Common utility functions used by the application.

### `src/models/Pipeline/`

Contains side-wise AI inference pipeline files:

~~~text
inference_pipeline_innerwall_mahal_pca.py
inference_pipeline_sidewall1_mahal_pca.py
inference_pipeline_sidewall2_mahal_pca.py
inference_pipeline_tread_mahal_pca.py
inference_pipeline_bead_mahal_pca.py
~~~

### `src/camera/`

Contains camera connection and capture-related files.

### `src/Pages/`

Contains PyQt GUI page files.

### `docs/CYCLE_FLOW.md`

Contains inspection cycle flow documentation.

---

## Features

- PyQt-based GUI application
- Login and dashboard pages
- SmartQC+ inspection cycle execution
- SKU / tyre nomenclature selection
- Side-wise AI inspection
- ViT-based template matching
- YOLO-based defect segmentation
- R-detection based alignment
- Runtime caching to avoid model reload every cycle
- Warmup support before live inspection
- Dynamic output folder creation
- Side-wise `crop` and `final` output folders
- Final stitched result visualization
- MongoDB cycle metadata saving
- CSV and JSON cycle summary generation

---

## Installation

### 1. Clone the repository

~~~bash
git clone https://github.com/Eyresai2025/Apollo_Vit_App.git
~~~

### 2. Go to the project folder

~~~bash
cd Apollo_Vit_App
~~~

### 3. Create Conda environment

~~~bash
conda create -n Apollo python=3.10 -y
~~~

### 4. Activate environment

~~~bash
conda activate Apollo
~~~

### 5. Install required packages

If `requirements.txt` is available:

~~~bash
pip install -r requirements.txt
~~~

Otherwise install commonly required packages:

~~~bash
pip install numpy pandas pillow opencv-python pymongo python-dotenv matplotlib scipy requests PyYAML
pip install torch torchvision
pip install ultralytics onnxruntime
~~~

---

## Environment File

Create a `.env` file in the project root directory.

Example:

~~~env
DATABASE_URL=mongodb://localhost:27017/
DATABASE_NAME=EyresQC_Apollo

DEPLOYMENT=False
PLC_IP=192.168.10.1

WEIGHT_FILE_Apollo=classification.pt
WEIGHT_FILE_RE=best_R_CEAT_DEMO.onnx
WEIGHT_FILE_CLASS=classification.pt
VIT_CHECKPOINT=vit_checkpoint.pth
~~~

Update the model names and paths according to your local setup.

---

## Running the Application

Activate the environment:

~~~bash
conda activate Apollo
~~~

Go to the project directory:

~~~bash
cd "C:\Users\PrajwalSridhar\Desktop\Apollo_Application\New folder"
~~~

Run the GUI:

~~~bash
python GUI.py
~~~

---

## Running Using BAT File

You can also create a Windows `.bat` file.

Create:

~~~text
run_apollo.bat
~~~

Paste this:

~~~bat
@echo off

call "C:\Users\PrajwalSridhar\anaconda3\Scripts\activate.bat" Apollo

cd /d "C:\Users\PrajwalSridhar\Desktop\Apollo_Application\New folder"

python GUI.py

pause
~~~

Double-click `run_apollo.bat` to launch the application.

---

## Git Ignore Notes

Large model files, output images, captured images, and local configuration files should not be pushed to GitHub.

Recommended ignored files/folders:

~~~text
*.pt
*.pth
*.onnx
*.engine
media/capture/
media/output/
.env
__pycache__/
*.pyc
media/img/login_ai.gif
media/img/login_ai1.gif
~~~

---

## Repository

~~~text
https://github.com/Eyresai2025/Apollo_Vit_App.git
~~~

---

## Contributors
- Eyres AI Team
- Apollo SmartQC+ Development Team
- [Yerriswamy Chakala](https://github.com/Yerriswamy2001)