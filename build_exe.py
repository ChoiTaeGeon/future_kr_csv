"""
Automated PyInstaller Build Script with Extreme Size Optimization
- Packages the full tick data pipeline and backtesting engine into a standalone .exe
- Strips unused heavy dependencies and excludes unnecessary modules
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ENTRY_POINT = BASE_DIR / "main.py"
EXE_NAME = "QuantTerminal"

# Modules to exclude for size optimization (heavy unused dev/gui toolkits)
EXCLUDES = [
    # Heavy unused GUI Toolkits
    "PySide6", "shiboken6", "PyQt5", "PyQt6", "PySide2", "wx", "gtk", "curses",
    # Dev & Doc tools & Notebooks
    "IPython", "jupyter", "notebook", "sphinx", "pytest", "pydoc", "pdb",
    "tornado", "zmq", "jedi", "lib2to3", "setuptools", "wheel", "pip",
    # Heavy unused scipy subpackages (we only use basic array/stats)
    "scipy.spatial", "scipy.optimize", "scipy.integrate", "scipy.interpolate", "scipy.cluster",
    "scipy.ndimage", "scipy.odr", "scipy.signal.windows", "scipy.io", "scipy.linalg.lapack",
    # Matplotlib tests and unused backends
    "matplotlib.tests", "matplotlib.testing", "mpl_toolkits.mplot3d",
    # Misc
    "test", "email.test"
]

def install_pyinstaller():
    print("[*] Checking / Installing PyInstaller...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

def build():
    install_pyinstaller()
    
    print(f"\n[*] Starting optimized PyInstaller build for {EXE_NAME}...")
    
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", EXE_NAME,
        "--onefile",
        "--clean",
        "--noconfirm",
        "--console",  # CLI console output
    ]

    # Add excludes
    for exc in EXCLUDES:
        cmd.extend(["--exclude-module", exc])

    # Include project packages and UI assets
    cmd.extend(["--paths", str(BASE_DIR)])
    cmd.extend(["--collect-submodules", "engine"])
    
    # Explicitly include all engine packages & submodules
    hidden_imports = [
        "engine",
        "engine.server",
        "engine.data_engine",
        "engine.resampler",
        "engine.strategy",
        "engine.backtester",
        "engine.reporter",
        "engine.tracker",
        "engine.indicators",
        "engine.cdt_converter",
        "engine.overseas_engine",
        "engine.kiwoom_manager",
        "engine.native_dialog",
        "engine.logger",
        "deep_learning",
        "deep_learning.config_loader",
        "tkinter",
        "tkinter.filedialog",
        "config"
    ]
    for hi in hidden_imports:
        cmd.extend(["--hidden-import", hi])

    ui_dir = BASE_DIR / "ui"
    if ui_dir.exists():
        cmd.extend(["--add-data", f"{ui_dir};ui"])
    engine_dir = BASE_DIR / "engine"
    if engine_dir.exists():
        cmd.extend(["--add-data", f"{engine_dir};engine"])
    configs_dir = BASE_DIR / "configs"
    if configs_dir.exists():
        cmd.extend(["--add-data", f"{configs_dir};configs"])

    cmd.append(str(ENTRY_POINT))

    print(f"[+] Running command: {' '.join(cmd)}\n")
    ret = subprocess.call(cmd)
    
    if ret != 0:
        print("\n[-] Build failed with exit code:", ret)
        sys.exit(ret)

    dist_data_dir = BASE_DIR / "dist" / "data"
    dist_data_dir.mkdir(parents=True, exist_ok=True)
    dist_cache_dir = dist_data_dir / "cache" / "resampled"
    dist_cache_dir.mkdir(parents=True, exist_ok=True)
    clean_db = BASE_DIR / "data" / "market_data.duckdb"
    if clean_db.exists():
        dist_db = dist_data_dir / "market_data.duckdb"
        shutil.copy2(clean_db, dist_db)
        print(f"[+] Synced clean database to {dist_db}")

    settings_src = BASE_DIR / "data" / "settings.json"
    if settings_src.exists():
        shutil.copy2(settings_src, dist_data_dir / "settings.json")

    dist_configs_dir = BASE_DIR / "dist" / "configs"
    dist_configs_dir.mkdir(parents=True, exist_ok=True)
    config_src = BASE_DIR / "configs" / "config.yaml"
    if config_src.exists():
        shutil.copy2(config_src, dist_configs_dir / "config.yaml")

    exe_path = BASE_DIR / "dist" / f"{EXE_NAME}.exe"
    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print("\n" + "=" * 70)
        print(f" [SUCCESS] Executable built successfully!")
        print(f" [Location] {exe_path}")
        print(f" [Optimized Size] {size_mb:.2f} MB")
        print("=" * 70)
    else:
        print("[-] Executable not found in dist folder.")

if __name__ == "__main__":
    build()
