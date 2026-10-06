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


def _run_in_isolated_process(mode: str, title: str, initial_dir: str, filter_str: str = ""):
    """
    Launch native dialog in a completely isolated main thread subprocess.
    This guarantees:
    1. Tkinter/Win32 COM runs on STA MainThread without blocking server threads.
    2. Guaranteed TopMost foreground elevation.
    3. Handles base64 encoded JSON communication to preserve UTF-8 paths flawlessly.
    """
    import base64
    import json
    payload = {
        "mode": mode,
        "title": title,
        "initial_dir": initial_dir or "",
        "filter_str": filter_str or ""
    }
    encoded_arg = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")

    try:
        # Use sys.executable to run module in a separate process
        cmd = [sys.executable, "-m", "engine.native_dialog", encoded_arg]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line.startswith("DIALOG_RES_B64:"):
                b64_res = line.split(":", 1)[1].strip()
                if b64_res:
                    decoded = base64.b64decode(b64_res.encode("ascii")).decode("utf-8")
                    return json.loads(decoded)
                return None
    except Exception as e:
        log_error(f"Isolated dialog subprocess error: {e}")
    return None


def _show_tkinter_folder(title: str, initial_dir: str) -> Optional[str]:
    """Execute Tkinter folder dialog on the main thread of current process."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.lift()
        root.focus_force()

        init_dir = initial_dir if (initial_dir and os.path.isdir(initial_dir)) else os.getcwd()
        path = filedialog.askdirectory(title=title, initialdir=init_dir)
        root.destroy()

        if path:
            norm = os.path.normpath(path)
            return norm if os.path.isdir(norm) else None
        return None
    except Exception as tk_err:
        log_error(f"Tkinter folder dialog failed: {tk_err}")
        return None


def _show_powershell_folder(title: str, initial_dir: str) -> Optional[str]:
    """Execute PowerShell FolderBrowserDialog with TopMost owner."""
    try:
        escaped_title = title.replace('"', '`"')
        escaped_init = initial_dir.replace('\\', '\\\\') if initial_dir else ""
        ps_script = f"""
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = "{escaped_title}"
$dialog.ShowNewFolderButton = $true
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
                return os.path.normpath(out)
        finally:
            if os.path.exists(script_path):
                os.remove(script_path)
    except Exception as ps_err:
        log_error(f"PowerShell folder dialog fallback failed: {ps_err}")

    return None


def select_folder(title: str = "폴더를 선택하세요", initial_dir: str = "") -> Optional[str]:
    """
    Open native Windows folder selection dialog in foreground.
    Spawns an isolated subprocess to prevent thread message deadlock.
    """
    # 1. Try isolated process execution (Tkinter -> PowerShell fallback inside)
    res = _run_in_isolated_process(mode="folder", title=title, initial_dir=initial_dir)
    if res is not None:
        if isinstance(res, str) and res:
            log_info(f"Native folder dialog selected (isolated): {res}")
            return res
        return None

    # 2. In-process Tkinter direct fallback
    tk_res = _show_tkinter_folder(title=title, initial_dir=initial_dir)
    if tk_res:
        log_info(f"Native folder dialog selected (Tkinter direct): {tk_res}")
        return tk_res

    # 3. In-process PowerShell direct fallback
    ps_res = _show_powershell_folder(title=title, initial_dir=initial_dir)
    if ps_res:
        log_info(f"Native folder dialog selected (PowerShell direct): {ps_res}")
        return ps_res

    return None


def _show_tkinter_files(title: str, initial_dir: str) -> List[str]:
    """Execute Tkinter file open dialog on main thread."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
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
            return [os.path.normpath(p) for p in paths if os.path.exists(p)]
        return []
    except Exception as tk_err:
        log_error(f"Tkinter files dialog failed: {tk_err}")
        return []


def _show_powershell_files(title: str, initial_dir: str, filter_str: str) -> List[str]:
    """Execute PowerShell OpenFileDialog with TopMost owner."""
    try:
        escaped_title = title.replace('"', '`"')
        escaped_init = initial_dir.replace('\\', '\\\\') if initial_dir else ""
        ps_script = f"""
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Title = "{escaped_title}"
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
                return [os.path.normpath(f.strip()) for f in out.split(";") if f.strip() and os.path.exists(f.strip())]
        finally:
            if os.path.exists(script_path):
                os.remove(script_path)
    except Exception as ps_err:
        log_error(f"PowerShell files dialog fallback failed: {ps_err}")

    return []


def select_files(title: str = "파일을 선택하세요", initial_dir: str = "", filter_str: str = "CSV / CDT Files (*.csv;*.cdt)|*.csv;*.cdt|All Files (*.*)|*.*") -> List[str]:
    """
    Open native Windows file selection dialog with multiselect support in foreground.
    Spawns an isolated subprocess to prevent thread message deadlock.
    """
    res = _run_in_isolated_process(mode="files", title=title, initial_dir=initial_dir, filter_str=filter_str)
    if res is not None:
        if isinstance(res, list):
            log_info(f"Native files dialog selected {len(res)} files (isolated)")
            return res
        return []

    tk_res = _show_tkinter_files(title=title, initial_dir=initial_dir)
    if tk_res:
        log_info(f"Native files dialog selected {len(tk_res)} files (Tkinter direct)")
        return tk_res

    ps_res = _show_powershell_files(title=title, initial_dir=initial_dir, filter_str=filter_str)
    if ps_res:
        log_info(f"Native files dialog selected {len(ps_res)} files (PowerShell direct)")
        return ps_res

    return []


if __name__ == "__main__":
    import base64
    import json

    # If launched with an encoded JSON payload:
    if len(sys.argv) > 1 and len(sys.argv[1]) > 5:
        raw_arg = sys.argv[1]
        try:
            cfg = json.loads(base64.b64decode(raw_arg.encode("ascii")).decode("utf-8"))
            m = cfg.get("mode", "folder")
            t = cfg.get("title", "")
            d = cfg.get("initial_dir", "")
            f = cfg.get("filter_str", "CSV / CDT Files (*.csv;*.cdt)|*.csv;*.cdt|All Files (*.*)|*.*")

            result = None
            if m == "folder":
                result = _show_tkinter_folder(t, d)
                if not result:
                    result = _show_powershell_folder(t, d)
            else:
                result = _show_tkinter_files(t, d)
                if not result:
                    result = _show_powershell_files(t, d, f)

            result_json = json.dumps(result if result is not None else "")
            res_b64 = base64.b64encode(result_json.encode("utf-8")).decode("ascii")
            print(f"DIALOG_RES_B64:{res_b64}")
            sys.exit(0)
        except Exception as e:
            err_json = json.dumps("")
            print(f"DIALOG_RES_B64:{base64.b64encode(err_json.encode('utf-8')).decode('ascii')}")
            sys.exit(1)
    else:
        # Standard CLI test
        mode = sys.argv[1] if len(sys.argv) > 1 else "folder"
        if mode == "folder":
            print("Selected folder:", select_folder("테스트 폴더 선택"))
        else:
            print("Selected files:", select_files("테스트 파일 선택"))
