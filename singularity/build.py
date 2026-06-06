"""
Build a standalone Singularity app for whatever OS you run this on.

    python build.py

Output lands in dist/Singularity (Windows: dist/Singularity.exe).
Run this once on a Windows machine to make a .exe for a Windows friend,
or on a Mac to make a Mac app for a Mac friend. You cannot cross-build.
"""
import subprocess, sys, platform

def main():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    subprocess.check_call([sys.executable, "-m", "PyInstaller", "--clean", "-y", "singularity.spec"])
    print(f"\nBuilt for {platform.system()}. See the dist/ folder.")

if __name__ == "__main__":
    main()
