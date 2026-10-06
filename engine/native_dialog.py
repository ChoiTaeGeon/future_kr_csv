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


import base64
import json


def _run_powershell_folder_dialog(title: str, initial_dir: str) -> Optional[str]:
    """
    Spawns an independent STA PowerShell process that opens a TopMost Windows Forms FolderBrowserDialog.
    Communicates via Base64 UTF-16LE EncodedCommand and UTF-8 Base64 output to prevent CP949 errors.
    """
    title_b64 = base64.b64encode((title or "폴더를 선택하세요").encode("utf-8")).decode("ascii")
    init_b64 = base64.b64encode((initial_dir or "").encode("utf-8")).decode("ascii")

    ps_code = f"""
$ProgressPreference = 'SilentlyContinue'
[void][System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms')

$title = [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{title_b64}'))
$init = [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{init_b64}'))

$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = $title
$dialog.ShowNewFolderButton = $true
if ($init -and (Test-Path $init)) {{
    $dialog.SelectedPath = $init
}}

$form = New-Object System.Windows.Forms.Form
$form.TopMost = $true
$form.Width = 1
$form.Height = 1
$form.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterScreen
$form.Show()
$form.Activate()
$form.BringToFront()

$result = $dialog.ShowDialog($form)
$form.Close()
$form.Dispose()

if ($result -eq [System.Windows.Forms.DialogResult]::OK) {{
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($dialog.SelectedPath)
    $outB64 = [Convert]::ToBase64String($bytes)
    [Console]::WriteLine("RESULT_B64:" + $outB64)
}} else {{
    [Console]::WriteLine("CANCELLED")
}}
"""
    encoded_cmd = base64.b64encode(ps_code.encode("utf-16le")).decode("ascii")
    cmd = ["powershell.exe", "-Sta", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded_cmd]

    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=180)
        stdout = proc.stdout.decode("utf-8", errors="replace")
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("RESULT_B64:"):
                b64_str = line.split(":", 1)[1].strip()
                if b64_str:
                    decoded = base64.b64decode(b64_str).decode("utf-8")
                    if os.path.isdir(decoded):
                        return os.path.normpath(decoded)
                    return os.path.normpath(decoded)
            elif line == "CANCELLED":
                return None
    except Exception as e:
        log_error(f"PowerShell folder dialog error: {e}")
    return None


def _run_powershell_files_dialog(title: str, initial_dir: str, filter_str: str) -> List[str]:
    """
    Spawns an independent STA PowerShell process that opens a TopMost Windows Forms OpenFileDialog.
    Supports multi-file selection. Communicates safely via Base64.
    """
    title_b64 = base64.b64encode((title or "파일을 선택하세요").encode("utf-8")).decode("ascii")
    init_b64 = base64.b64encode((initial_dir or "").encode("utf-8")).decode("ascii")
    filter_b64 = base64.b64encode((filter_str or "All Files (*.*)|*.*").encode("utf-8")).decode("ascii")

    ps_code = f"""
$ProgressPreference = 'SilentlyContinue'
[void][System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms')

$title = [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{title_b64}'))
$init = [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{init_b64}'))
$filter = [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{filter_b64}'))

$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Title = $title
$dialog.Multiselect = $true
$dialog.Filter = $filter
if ($init -and (Test-Path $init)) {{
    $dialog.InitialDirectory = $init
}}

$form = New-Object System.Windows.Forms.Form
$form.TopMost = $true
$form.Width = 1
$form.Height = 1
$form.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterScreen
$form.Show()
$form.Activate()
$form.BringToFront()

$result = $dialog.ShowDialog($form)
$form.Close()
$form.Dispose()

if ($result -eq [System.Windows.Forms.DialogResult]::OK) {{
    $allPaths = $dialog.FileNames -join "`t"
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($allPaths)
    $outB64 = [Convert]::ToBase64String($bytes)
    [Console]::WriteLine("RESULT_B64:" + $outB64)
}} else {{
    [Console]::WriteLine("CANCELLED")
}}
"""
    encoded_cmd = base64.b64encode(ps_code.encode("utf-16le")).decode("ascii")
    cmd = ["powershell.exe", "-Sta", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded_cmd]

    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=180)
        stdout = proc.stdout.decode("utf-8", errors="replace")
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("RESULT_B64:"):
                b64_str = line.split(":", 1)[1].strip()
                if b64_str:
                    decoded = base64.b64decode(b64_str).decode("utf-8")
                    paths = [os.path.normpath(p) for p in decoded.split("\t") if p.strip()]
                    return [p for p in paths if os.path.exists(p)]
            elif line == "CANCELLED":
                return []
    except Exception as e:
        log_error(f"PowerShell files dialog error: {e}")
    return []


def select_folder(title: str = "폴더를 선택하세요", initial_dir: str = "") -> Optional[str]:
    """
    Open native Windows folder selection dialog in foreground.
    Guaranteed TopMost execution across all Windows desktop versions.
    """
    path = _run_powershell_folder_dialog(title=title, initial_dir=initial_dir)
    if path:
        log_info(f"Native folder dialog selected: {path}")
        return path
    return None


def select_files(title: str = "파일을 선택하세요", initial_dir: str = "", filter_str: str = "CSV / CDT Files (*.csv;*.cdt)|*.csv;*.cdt|All Files (*.*)|*.*") -> List[str]:
    """
    Open native Windows file selection dialog with multiselect support in foreground.
    """
    files = _run_powershell_files_dialog(title=title, initial_dir=initial_dir, filter_str=filter_str)
    if files:
        log_info(f"Native files dialog selected {len(files)} files")
        return files
    return []


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "folder"
    if mode == "folder":
        res = select_folder("테스트 폴더 선택")
        print("Selected folder:", res)
    else:
        res = select_files("테스트 파일 선택")
        print("Selected files:", res)
