<div>
  <img src="media/img/Eyres.jpeg" alt="Logo" width="200" align="left"/>
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
<br/>
<br/>
<br/>

# EyresQC+ Apollo GUI

## Overview

EyresQC+ Apollo GUI is a graphical user interface application designed for letter reading and conducting quality control checks on Apollo tires. The application provides an intuitive interface for running quality control cycles, selecting tire parameters, and visualizing the results.

## Features

- **Main Cycle Execution**: Clicking the "Run SmartQC+" button initiates the main cycle for quality control checks.
- **Tyre Parameter Selection**: Users can select the tire nomenclature before starting the quality control cycle.
- **Result Visualization**: After completing the cycle, the results are displayed.Reading letter tyre will be shown in GUI and detected letters and words file also.

## Usage

To use EyresQC+ Apollo GUI:

1. Download the application from the provided source.
2. Run the application executable file.
3. Click on the "Run SmartQC+" button to start the quality control cycle.
4. Select the appropriate tire nomenclature culture.
5. After completion, review the results displayed on the interface.

## Installation

1. Clone the project:
    ```bash
    git clone https://github.com/Eyresai2025/Apollo.git
    ```

2. Create a virtual environment:
    ```bash
    python -m venv Apollo_env
    ```

3. Activate the virtual environment:
    - For Windows:
        ```bash
        Apollo_env\Scripts\activate
        ```
    - For Linux/Mac:
        ```bash
        source Apollo_env/bin/activate
        ```

4. Navigate to the project directory:
    ```bash
    cd Apollo_GUI
    ```

5. Install the required packages using `requirements.txt`:
    ```bash
    pip install numpy pillow==10.4.0 pymongo==4.10.1 pandas ultralytics==8.3.39 pymodbus==3.6.9 sahi==0.11.19 torch==2.4.1 torchvision==0.19.1 opencv-python==4.9.0.80 matplotlib==3.7.5 seaborn==0.13.2 scipy==1.10.1 requests==2.32.3 PyYAML==6.0.2
    ```
## Usage

To run the EyresQC+ Apollo GUI:

1. Ensure that you have activated the virtual environment.
2. Navigate to the project directory if you haven't already.
3. Create .env file in that directory.
   ![env sc](media/Screenshot/Main6.png)
4. Run the GUI file:

    ```bash
    python Apollo_GUI.py
    ```

## Contributors
- [Yerriswamy Chakala](https://github.com/Yerriswamy2001)