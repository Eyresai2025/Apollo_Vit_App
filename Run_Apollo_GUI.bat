@echo off
title Apollo GUI Application
echo ==========================================
echo Starting Apollo GUI Application
echo ==========================================

set "PROJECT_DIR=C:\Users\YerriswamyChakala\Desktop\Apollo_Vit_App"
set "ENV_NAME=Apollo"

echo.
echo Activating Conda environment: %ENV_NAME%

if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\anaconda3\Scripts\activate.bat" %ENV_NAME%
) else if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\miniconda3\Scripts\activate.bat" %ENV_NAME%
) else if exist "C:\ProgramData\anaconda3\Scripts\activate.bat" (
    call "C:\ProgramData\anaconda3\Scripts\activate.bat" %ENV_NAME%
) else (
    echo.
    echo ERROR: Could not find Anaconda/Miniconda activate.bat
    echo Please check your Anaconda installation path.
    echo.
    pause
    exit /b 1
)

echo.
echo Changing directory to project:
echo %PROJECT_DIR%

cd /d "%PROJECT_DIR%"

echo.
echo Running GUI.py...
echo ==========================================

python GUI.py

echo.
echo ==========================================
echo GUI closed or error occurred.
echo Console kept open for logs.
echo ==========================================
pause