@echo off
REM ===== Double-click this on a WINDOWS PC to build the app =====
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python is not installed on THIS PC ^(the builder^). 
  echo Install Python 3.12 from https://python.org ^(tick "Add Python to PATH"^),
  echo then double-click this file again.
  echo Your friend does NOT need Python - only this computer does, once.
  pause & exit /b
)
echo Installing build tools...
python -m pip install -r requirements.txt
echo Building Singularity.exe ...
python -m PyInstaller --clean -y singularity.spec
echo.
echo ============================================================
echo  DONE. The app your friend clicks is here:
echo     dist\Singularity.exe
echo  Send that ONE file. He just double-clicks it.
echo ============================================================
pause
