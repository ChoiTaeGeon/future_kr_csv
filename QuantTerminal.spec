# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = ['engine', 'engine.server', 'engine.data_engine', 'engine.resampler', 'engine.strategy', 'engine.backtester', 'engine.reporter', 'engine.tracker', 'engine.indicators', 'engine.cdt_converter', 'engine.overseas_engine', 'engine.kiwoom_manager', 'engine.native_dialog', 'engine.logger', 'deep_learning', 'deep_learning.config_loader', 'tkinter', 'tkinter.filedialog', 'config']
hiddenimports += collect_submodules('engine')


a = Analysis(
    ['D:/coding/future_kr_csv/main.py'],
    pathex=['D:/coding/future_kr_csv'],
    binaries=[],
    datas=[('D:/coding/future_kr_csv/ui', 'ui'), ('D:/coding/future_kr_csv/engine', 'engine'), ('D:/coding/future_kr_csv/configs', 'configs')],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PySide6', 'shiboken6', 'PyQt5', 'PyQt6', 'PySide2', 'wx', 'gtk', 'curses', 'IPython', 'jupyter', 'notebook', 'sphinx', 'pytest', 'pydoc', 'pdb', 'tornado', 'zmq', 'jedi', 'lib2to3', 'setuptools', 'wheel', 'pip', 'scipy.spatial', 'scipy.optimize', 'scipy.integrate', 'scipy.interpolate', 'scipy.cluster', 'scipy.ndimage', 'scipy.odr', 'scipy.signal.windows', 'scipy.io', 'scipy.linalg.lapack', 'matplotlib.tests', 'matplotlib.testing', 'mpl_toolkits.mplot3d', 'test', 'email.test'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='QuantTerminal',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
