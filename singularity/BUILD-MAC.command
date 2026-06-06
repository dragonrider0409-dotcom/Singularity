#!/bin/bash
# ===== Double-click this on a MAC to build the app =====
cd "$(dirname "$0")"
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is not installed on THIS Mac (the builder)."
  echo "Install it from https://python.org, then double-click this again."
  echo "Your friend does NOT need Python - only this computer does, once."
  read -n1 -r -p "Press any key to close..."; exit 1
fi
echo "Installing build tools..."
python3 -m pip install -r requirements.txt
echo "Building the Singularity app ..."
python3 -m PyInstaller --clean -y singularity.spec
echo ""
echo "============================================================"
echo " DONE. The app your friend clicks is in the dist/ folder."
echo " Send that. He just double-clicks it."
echo "============================================================"
read -n1 -r -p "Press any key to close..."
