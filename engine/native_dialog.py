"""
Native Windows Explorer File and Folder Dialog Handler
Guaranteed to appear in foreground (TopMost) across Windows 10/11 desktops.
Dual engine:
1. Python Tkinter (with TopMost & parent attachment)
2. PowerShell Windows Forms (with STA & TopMost owner form)
"""
import os
import sys
import tempfile
import subprocess
from typing import Optional, List
from engine.logger import log_error, log_info


def select_folder(title: str = "폴더를 선택하세요", initial_dir: str = "") -> Optional[str]:
    """
    Open native Windows folder selection dialog in foreground.
    Returns the absolute path of selected directory, or None if cancelled.
    """
    # Engine 1: Tkinter with TopMost parent
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', 1)
        root.lift()
        root.focus_force()

        init_dir = initial_dir if (initial_dir and os.path.isdir(initial_dir)) else os.getcwd()
        path = filedialog.askdirectory(title=title, initialdir=init_dir)
        root.destroy()

        if path:
            norm = os.path.normpath(path)
            if os.path.isdir(norm):
                log_info(f"Native folder dialog selected (Tkinter): {norm}")
                return norm
            return norm
        else:
            return None
    except Exception as tk_err:
        log_error(f"Tkinter folder dialog failed, trying PowerShell: {tk_err}")

    # Engine 2: PowerShell STA with TopMost dummy owner
    try:
        escaped_init = initial_dir.replace('\\', '\\\\') if initial_dir else ""
        ps_script = f"""
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = "{title}"
$dialog.ShowNewFolderButton = $true
$dialog.AutoUpgradeEnabled = $true
$init = "{escaped_init}"
if ($init -and (Test-Path $init)) {{
    $dialog.SelectedPath = $init
}}

$owner = New-Object System.Windows.Forms.Form
$owner.TopMost = $true
$owner.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterScreen
$owner.Width = 0
$owner.Height = 0
$owner.Show()
$owner.BringToFront()
$owner.Activate()

$res = $dialog.ShowDialog($owner)
$owner.Close()
$owner.Dispose()

if ($res -eq [System.Windows.Forms.DialogResult]::OK) {{
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    Write-Output $dialog.SelectedPath
}}
"""
        with tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=False, encoding="utf-8") as tf:
            tf.write(ps_script)
            script_path = tf.name

        try:
            cmd = ["powershell", "-Sta", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script_path]
            out = subprocess.check_output(cmd, timeout=120).decode("utf-8", errors="replace").strip()
            if out and os.path.isdir(out):
                log_info(f"Native folder dialog selected (PowerShell): {out}")
                return out
        finally:
            if os.path.exists(script_path):
                os.remove(script_path)
    except Exception as ps_err:
        log_error(f"PowerShell folder dialog fallback failed: {ps_err}")

    return None


def select_files(title: str = "파일을 선택하세요", initial_dir: str = "", filter_str: str = "CSV / CDT Files (*.csv;*.cdt)|*.csv;*.cdt|All Files (*.*)|*.*") -> List[str]:
    """
    Open native Windows file selection dialog with multiselect support in foreground.
    Returns list of selected absolute file paths, or empty list if cancelled.
    """
    # Engine 1: Tkinter with TopMost parent
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', 1)
        root.lift()
        root.focus_force()

        init_dir = initial_dir if (initial_dir and os.path.isdir(initial_dir)) else os.getcwd()
        paths = filedialog.askopenfilenames(
            title=title,
            initialdir=init_dir,
            filetypes=[
                ("CSV / CDT Files", "*.csv;*.cdt"),
                ("CSV Files", "*.csv"),
                ("CDT Files", "*.cdt"),
                ("All Files", "*.*")
            ]
        )
        root.destroy()

        if paths:
            res = [os.path.normpath(p) for p in paths if os.path.exists(p)]
            log_info(f"Native files dialog selected {len(res)} files (Tkinter)")
            return res
        else:
            return []
    except Exception as tk_err:
        log_error(f"Tkinter files dialog failed, trying PowerShell: {tk_err}")

    # Engine 2: PowerShell STA with TopMost dummy owner
    try:
        escaped_init = initial_dir.replace('\\', '\\\\') if initial_dir else ""
        ps_script = f"""
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Title = "{title}"
$dialog.Multiselect = $true
$dialog.Filter = "{filter_str}"
$init = "{escaped_init}"
if ($init -and (Test-Path $init)) {{
    $dialog.InitialDirectory = $init
}}

$owner = New-Object System.Windows.Forms.Form
$owner.TopMost = $true
$owner.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterScreen
$owner.Width = 0
$owner.Height = 0
$owner.Show()
$owner.BringToFront()
$owner.Activate()

$res = $dialog.ShowDialog($owner)
$owner.Close()
$owner.Dispose()

if ($res -eq [System.Windows.Forms.DialogResult]::OK) {{
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    Write-Output ($dialog.FileNames -join ";")
}}
"""
        with tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=False, encoding="utf-8") as tf:
            tf.write(ps_script)
            script_path = tf.name

        try:
            cmd = ["powershell", "-Sta", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script_path]
            out = subprocess.check_output(cmd, timeout=120).decode("utf-8", errors="replace").strip()
            if out:
                file_list = [os.path.normpath(f.strip()) for f in out.split(";") if f.strip() and os.path.exists(f.strip())]
                log_info(f"Native files dialog selected {len(file_list)} files (PowerShell)")
                return file_list
        finally:
            if os.path.exists(script_path):
                os.remove(script_path)
    except Exception as ps_err:
        log_error(f"PowerShell files dialog fallback failed: {ps_err}")

    return []


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "folder"
    if mode == "folder":
        print(select_folder("테스트 폴더 선택"))
    else:
        print(select_files("테스트 파일 선택"))
