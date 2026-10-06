"""
Local Lightweight REST API & Dashboard Server
- Built on Python standard library (http.server) to guarantee zero extra binary overhead & fast startup.
- Serves the Glassmorphism Dark Web Terminal UI and handles backtest execution & PDF downloads.
"""
import os
import sys
import json
import mimetypes
import webbrowser
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import threading

from datetime import datetime
import pandas as pd
import numpy as np

from config import BASE_DIR, RAW_CSV_DIR, DB_PATH, REPORTS_DIR, CHARTS_DIR, OUTPUT_DIR, BacktestConfig
from engine.data_engine import DataEngine, get_data_engine
from engine.resampler import Resampler
from engine.strategy import StrategyCategory
from engine.backtester import Backtester
from engine.reporter import Visualizer, ReportGenerator
from engine.tracker import tracker
from engine.logger import log_error, log_sync, log_info


class RobustThreadingServer(ThreadingHTTPServer):
    """Threading HTTP Server that gracefully ignores client socket disconnects / WinError 10053."""
    daemon_threads = True

    def handle_error(self, request, client_address):
        """Suppress noisy ConnectionAbortedError / ConnectionResetError on client disconnect."""
        exc_type, exc_val, _ = sys.exc_info()
        if exc_type in (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            return
        log_error(f"HTTP Server Exception from {client_address}", exc=exc_val)
        super().handle_error(request, client_address)


class QuantRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        """Suppress default logging for clean console output."""
        return

    def _safe_write(self, data: bytes):
        """Safely write bytes to socket, ignoring connection aborted errors."""
        try:
            self.wfile.write(data)
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass

    def _set_json_headers(self, status=200):
        try:
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
        except Exception:
            pass

    def do_OPTIONS(self):
        self._set_json_headers(200)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            if path in ['/', '/index.html']:
                self._serve_ui_file()
            elif path == '/api/status':
                self._handle_status()
            elif path == '/api/dataset-info':
                self._handle_dataset_info()
            elif path == '/api/progress':
                self._handle_progress()
            elif path == '/api/settings':
                self._handle_get_settings()
            elif path == '/api/export-csv':
                self._handle_export_csv(parsed)
            elif path == '/api/download-pdf':
                self._handle_download_pdf(parsed)
            elif path == '/api/logs':
                self._handle_get_logs(parsed)
            elif path == '/api/overseas/status':
                self._handle_overseas_status()
            elif path == '/api/kiwoom/status':
                self._handle_kiwoom_status()
            elif path == '/api/kiwoom/config':
                self._handle_kiwoom_config()
            elif path == '/api/kiwoom/realtime-bars':
                self._handle_kiwoom_realtime_bars(parsed)
            elif path == '/api/dl/config':
                self._handle_dl_get_config()
            elif path == '/api/dl/progress':
                self._handle_dl_progress()
            elif path == '/api/dl/reports':
                self._handle_dl_reports()
            else:
                self.send_error(404, "File Not Found")
        except Exception as e:
            log_error(f"HTTP GET error for {path}", exc=e)
            self._set_json_headers(200)
            self._safe_write(json.dumps({"status": "error", "message": f"요청 처리 오류: {e}"}).encode('utf-8'))

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else "{}"

        try:
            body = json.loads(post_data) if post_data else {}
        except Exception:
            body = {}

        try:
            if path == '/api/sync':
                self._handle_sync(body)
            elif path == '/api/preview-chart':
                self._handle_preview_chart(body)
            elif path == '/api/backtest':
                self._handle_backtest(body)
            elif path == '/api/stop-backtest':
                self._handle_stop_backtest()
            elif path == '/api/browse-folder':
                self._handle_browse_folder()
            elif path == '/api/browse-files':
                self._handle_browse_files()
            elif path == '/api/convert-cdt':
                self._handle_convert_cdt(body)
            elif path == '/api/settings':
                self._handle_save_settings(body)
            elif path == '/api/logs/clear':
                self._handle_clear_logs(body)
            elif path == '/api/overseas/browse-files':
                self._handle_overseas_browse_files()
            elif path == '/api/overseas/browse-folder':
                self._handle_overseas_browse_folder()
            elif path == '/api/overseas/load':
                self._handle_overseas_load(body)
            elif path == '/api/overseas/preview-chart':
                self._handle_overseas_preview_chart(body)
            elif path == '/api/overseas/backtest':
                self._handle_overseas_backtest(body)
            elif path == '/api/kiwoom/save-config':
                self._handle_kiwoom_save_config(body)
            elif path == '/api/kiwoom/connect':
                self._handle_kiwoom_connect(body)
            elif path == '/api/kiwoom/disconnect':
                self._handle_kiwoom_disconnect()
            elif path == '/api/kiwoom/set-tick-size':
                self._handle_kiwoom_set_tick_size(body)
            elif path == '/api/kiwoom/toggle-streaming':
                self._handle_kiwoom_toggle_streaming(body)
            elif path == '/api/dl/save-config':
                self._handle_dl_save_config(body)
            elif path == '/api/dl/browse-folder':
                self._handle_dl_browse_folder()
            elif path == '/api/dl/scan-folder':
                self._handle_dl_scan_folder(body)
            elif path == '/api/dl/start':
                self._handle_dl_start_pipeline(body)
            elif path == '/api/dl/stop':
                self._handle_dl_stop_pipeline()
            else:
                self.send_error(404, "Unknown Endpoint")
        except Exception as e:
            log_error(f"HTTP POST error for {path}", exc=e)
            self._set_json_headers(200)
            self._safe_write(json.dumps({"status": "error", "message": f"요청 처리 오류: {e}"}).encode('utf-8'))

    def _serve_ui_file(self):
        # Look for index.html in ui/
        ui_path = BASE_DIR / "ui" / "index.html"
        if not ui_path.exists():
            # If running in PyInstaller frozen mode, check sys._MEIPASS
            meipass = getattr(sys, '_MEIPASS', None)
            if meipass:
                ui_path = Path(meipass) / "ui" / "index.html"

        if ui_path.exists():
            with open(ui_path, 'rb') as f:
                content = f.read()
            try:
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self._safe_write(content)
            except Exception:
                pass
        else:
            self.send_error(404, "UI template not found.")

    def _handle_progress(self):
        from engine.tracker import tracker
        status = tracker.get_status()
        self._set_json_headers(200)
        self._safe_write(json.dumps(status).encode('utf-8'))

    def _handle_stop_backtest(self):
        from engine.tracker import tracker
        tracker.cancel()
        self._set_json_headers(200)
        self._safe_write(json.dumps({"status": "cancelled", "message": "Stopped by user."}).encode('utf-8'))

    def _handle_get_settings(self):
        from config import load_settings
        settings = load_settings()
        self._set_json_headers(200)
        self.wfile.write(json.dumps(settings).encode('utf-8'))

    def _handle_save_settings(self, body: dict):
        from config import save_settings
        save_settings(body)
        self._set_json_headers(200)
        self.wfile.write(json.dumps({"status": "success"}).encode('utf-8'))

    def _handle_browse_folder(self):
        """Open native Windows folder browser dialog with lossless UTF-8 / Base64 encoding (Fix Korean path corruption)."""
        import subprocess
        import base64
        from config import save_settings

        # Use Base64 encoding to completely bypass Windows CLI console encoding / CP949 corruption
        # Create TopMost dummy form to guarantee dialog appears in front of the browser window
        cmd = [
            "powershell", "-NoProfile", "-Command",
            "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$top = New-Object System.Windows.Forms.Form; "
            "$top.TopMost = $true; "
            "$f = New-Object System.Windows.Forms.FolderBrowserDialog; "
            "$f.Description = 'Select Tick CSV Folder (틱데이터 CSV 폴더 선택)'; "
            "if ($f.ShowDialog($top) -eq [System.Windows.Forms.DialogResult]::OK) { "
            "    $bytes = [System.Text.Encoding]::UTF8.GetBytes($f.SelectedPath); "
            "    [System.Convert]::ToBase64String($bytes) "
            "}"
        ]
        flags = 0x08000000 if sys.platform == 'win32' else 0
        try:
            raw_out = subprocess.check_output(cmd, creationflags=flags).decode('ascii', errors='ignore').strip()
            if raw_out:
                folder_str = base64.b64decode(raw_out).decode('utf-8', errors='replace').strip()
                if folder_str:
                    save_settings({"last_csv_folder": folder_str})
                    self._set_json_headers(200)
                    self._safe_write(json.dumps({"status": "success", "folder": folder_str}, ensure_ascii=False).encode('utf-8'))
                    return
        except Exception as e:
            log_error("Error opening FolderBrowserDialog", exc=e)

        self._set_json_headers(200)
        self._safe_write(json.dumps({"status": "cancelled", "folder": None}).encode('utf-8'))

    def _handle_browse_files(self):
        """Open native Windows OpenFileDialog with multiselect enabled and lossless UTF-8 / Base64 encoding."""
        import subprocess
        import base64
        from config import save_settings

        cmd = [
            "powershell", "-NoProfile", "-Command",
            "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$top = New-Object System.Windows.Forms.Form; "
            "$top.TopMost = $true; "
            "$f = New-Object System.Windows.Forms.OpenFileDialog; "
            "$f.Filter = 'CSV Files (*.csv)|*.csv|All Files (*.*)|*.*'; "
            "$f.Multiselect = $true; "
            "$f.Title = 'Select One or More Tick CSV Files (단일 또는 복수 CSV 파일 선택)'; "
            "if ($f.ShowDialog($top) -eq [System.Windows.Forms.DialogResult]::OK) { "
            "    $joined = [string]::Join(';', $f.FileNames); "
            "    $bytes = [System.Text.Encoding]::UTF8.GetBytes($joined); "
            "    [System.Convert]::ToBase64String($bytes) "
            "}"
        ]
        flags = 0x08000000 if sys.platform == 'win32' else 0
        try:
            raw_out = subprocess.check_output(cmd, creationflags=flags).decode('ascii', errors='ignore').strip()
            if raw_out:
                files_str = base64.b64decode(raw_out).decode('utf-8', errors='replace').strip()
                if files_str:
                    file_list = [f.strip() for f in files_str.split(';') if f.strip()]
                    save_settings({"last_csv_folder": files_str})
                    self._set_json_headers(200)
                    self._safe_write(json.dumps({
                        "status": "success", 
                        "files_str": files_str, 
                        "files": file_list,
                        "count": len(file_list)
                    }, ensure_ascii=False).encode('utf-8'))
                    return
        except Exception as e:
            log_error("Error opening OpenFileDialog", exc=e)

        self._set_json_headers(200)
        self._safe_write(json.dumps({"status": "cancelled", "files": []}).encode('utf-8'))

    def _handle_convert_cdt(self, body: dict):
        """Batch convert target CSV files or folder into Kiwoom .cdt binary files."""
        from engine.cdt_converter import CDTConverter
        from config import OUTPUT_DIR, RAW_CSV_DIR
        
        target_str = (body or {}).get('csv_target') or (body or {}).get('csv_folder') or (body or {}).get('folder_path') or str(RAW_CSV_DIR)
        target_folder = OUTPUT_DIR / "cdt_converted"
        target_folder.mkdir(parents=True, exist_ok=True)
        
        try:
            converted_files = CDTConverter.batch_convert_target(target_str, target_folder=target_folder)
            res = {
                "status": "success",
                "total_converted": len(converted_files),
                "output_dir": str(target_folder),
                "files": [f.name for f in converted_files]
            }
            self._set_json_headers(200)
            self._safe_write(json.dumps(res, ensure_ascii=False).encode('utf-8'))
        except Exception as e:
            self._set_json_headers(200)
            self._safe_write(json.dumps({"status": "error", "message": f"CDT Conversion failed: {e}"}).encode('utf-8'))

    # ==========================================
    # OVERSEAS FUTURES CDT API HANDLERS
    # ==========================================

    def _handle_overseas_status(self):
        """Returns metadata of currently loaded Overseas Futures CDT dataset."""
        from engine.overseas_engine import get_overseas_manager
        mgr = get_overseas_manager()
        meta = mgr.metadata if mgr.metadata else {"status": "empty", "total_bars": 0}
        self._set_json_headers(200)
        self._safe_write(json.dumps(meta, ensure_ascii=False).encode('utf-8'))

    def _handle_overseas_browse_files(self):
        """Open native Windows OpenFileDialog with multiselect for .cdt and .csv files."""
        import subprocess
        import base64
        cmd = [
            "powershell", "-NoProfile", "-Command",
            "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$top = New-Object System.Windows.Forms.Form; "
            "$top.TopMost = $true; "
            "$f = New-Object System.Windows.Forms.OpenFileDialog; "
            "$f.Filter = 'CDT & CSV Files (*.cdt;*.csv)|*.cdt;*.csv|CSV Files (*.csv)|*.csv|CDT Files (*.cdt)|*.cdt|All Files (*.*)|*.*'; "
            "$f.Multiselect = $true; "
            "$f.Title = 'Select Overseas Futures CDT / CSV Files (해외선물 CDT / CSV 파일 선택)'; "
            "if ($f.ShowDialog($top) -eq [System.Windows.Forms.DialogResult]::OK) { "
            "    $joined = [string]::Join(';', $f.FileNames); "
            "    $bytes = [System.Text.Encoding]::UTF8.GetBytes($joined); "
            "    [System.Convert]::ToBase64String($bytes) "
            "}"
        ]
        flags = 0x08000000 if sys.platform == 'win32' else 0
        try:
            raw_out = subprocess.check_output(cmd, creationflags=flags).decode('ascii', errors='ignore').strip()
            if raw_out:
                files_str = base64.b64decode(raw_out).decode('utf-8', errors='replace').strip()
                if files_str:
                    file_list = [f.strip() for f in files_str.split(';') if f.strip()]
                    self._set_json_headers(200)
                    self._safe_write(json.dumps({
                        "status": "success",
                        "files_str": files_str,
                        "files": file_list,
                        "count": len(file_list)
                    }, ensure_ascii=False).encode('utf-8'))
                    return
        except Exception as e:
            log_error("Error opening OpenFileDialog for CDT", exc=e)

        self._set_json_headers(200)
        self._safe_write(json.dumps({"status": "cancelled", "files": []}).encode('utf-8'))

    def _handle_overseas_browse_folder(self):
        """Open native Windows FolderBrowserDialog for selecting a folder of CDT files."""
        import subprocess
        import base64
        cmd = [
            "powershell", "-NoProfile", "-Command",
            "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$top = New-Object System.Windows.Forms.Form; "
            "$top.TopMost = $true; "
            "$f = New-Object System.Windows.Forms.FolderBrowserDialog; "
            "$f.Description = 'Select Folder Containing Overseas Futures CDT/CSV Files (해외선물 CDT / CSV 폴더 선택)'; "
            "if ($f.ShowDialog($top) -eq [System.Windows.Forms.DialogResult]::OK) { "
            "    $bytes = [System.Text.Encoding]::UTF8.GetBytes($f.SelectedPath); "
            "    [System.Convert]::ToBase64String($bytes) "
            "}"
        ]
        flags = 0x08000000 if sys.platform == 'win32' else 0
        try:
            raw_out = subprocess.check_output(cmd, creationflags=flags).decode('ascii', errors='ignore').strip()
            if raw_out:
                folder_str = base64.b64decode(raw_out).decode('utf-8', errors='replace').strip()
                if folder_str:
                    self._set_json_headers(200)
                    self._safe_write(json.dumps({"status": "success", "folder": folder_str}, ensure_ascii=False).encode('utf-8'))
                    return
        except Exception as e:
            log_error("Error opening FolderBrowserDialog for CDT", exc=e)

        self._set_json_headers(200)
        self._safe_write(json.dumps({"status": "cancelled", "folder": None}).encode('utf-8'))

    def _handle_overseas_load(self, body: dict):
        """Parses and deduplicates target CDT folder or files."""
        from engine.overseas_engine import get_overseas_manager
        target = (body or {}).get('target') or (body or {}).get('folder_path') or (body or {}).get('files_str') or (body or {}).get('files') or "cdt"
        mgr = get_overseas_manager()
        res = mgr.load_sources(target)
        self._set_json_headers(200)
        self._safe_write(json.dumps(res, ensure_ascii=False).encode('utf-8'))

    def _handle_overseas_preview_chart(self, body: dict):
        """Renders candlestick chart and indicators for loaded overseas CDT data."""
        from engine.overseas_engine import get_overseas_manager
        mgr = get_overseas_manager()
        if mgr.current_df.empty:
            target = (body or {}).get('target') or (body or {}).get('folder_path') or (body or {}).get('files_str') or "cdt"
            if target:
                mgr.load_sources(target)
        if mgr.current_df.empty:
            self._set_json_headers(200)
            self._safe_write(json.dumps({"status": "error", "message": "로드된 해외선물 CDT 데이터가 없습니다. 먼저 CDT 파일을 로드해주세요."}).encode('utf-8'))
            return

        res = mgr.get_chart_data(
            period_mode=(body or {}).get('period_mode', 'ALL'),
            target_date=(body or {}).get('target_date'),
            target_year=(body or {}).get('target_year'),
            target_month=(body or {}).get('target_month'),
            start_date=(body or {}).get('start_date'),
            end_date=(body or {}).get('end_date'),
            filter_outliers=bool((body or {}).get('filter_outliers', True))
        )
        self._set_json_headers(200)
        self._safe_write(json.dumps(res, ensure_ascii=False).encode('utf-8'))

    def _handle_overseas_backtest(self, body: dict):
        """Executes quantitative strategy backtest on loaded overseas CDT data."""
        from engine.overseas_engine import get_overseas_manager
        mgr = get_overseas_manager()
        if mgr.current_df.empty:
            target = (body or {}).get('target') or (body or {}).get('folder_path') or (body or {}).get('files_str') or "cdt"
            if target:
                mgr.load_sources(target)
        if mgr.current_df.empty:
            self._set_json_headers(200)
            self._safe_write(json.dumps({"status": "error", "message": "로드된 해외선물 CDT 데이터가 없습니다. 먼저 CDT 파일을 로드해주세요."}).encode('utf-8'))
            return

        res = mgr.run_backtest(body or {})
        self._set_json_headers(200)
        self._safe_write(json.dumps(res, ensure_ascii=False).encode('utf-8'))

    def _handle_status(self):
        from config import load_settings
        data_engine = get_data_engine()
        total_ticks = data_engine.get_total_ticks()
        symbols = data_engine.get_symbols()
        sym = symbols[0] if symbols else "KOSPI_F"
        date_range = data_engine.get_date_range(symbol=sym) if total_ticks > 0 else (None, None)
        years = data_engine.get_available_years()
        months = data_engine.get_available_months()
        dates = data_engine.get_available_dates()
        settings = load_settings()

        res = {
            "total_ticks": total_ticks,
            "symbol": sym,
            "start_time": str(date_range[0]) if date_range[0] else None,
            "end_time": str(date_range[1]) if date_range[1] else None,
            "available_years": years,
            "available_months": months,
            "available_dates": dates,
            "latest_date": dates[-1] if dates else None,
            "last_csv_folder": settings.get("last_csv_folder", "csv")
        }
        self._set_json_headers(200)
        self._safe_write(json.dumps(res).encode('utf-8'))

    def _handle_dataset_info(self):
        data_engine = get_data_engine()
        info = data_engine.get_dataset_summary()
        self._set_json_headers(200)
        self._safe_write(json.dumps(info).encode('utf-8'))

    def _resolve_folder(self, folder_str: str) -> Path:
        """Resolve folder path relative to BASE_DIR if not absolute."""
        if not folder_str:
            return RAW_CSV_DIR
        p = Path(folder_str)
        if not p.is_absolute():
            p = (BASE_DIR / p).resolve()
        return p

    def _handle_sync(self, body: dict = None):
        from config import save_settings
        folder_str = (body or {}).get('folder_path') or str(RAW_CSV_DIR)
        force_full = bool((body or {}).get('force_full', False))
        folder = self._resolve_folder(folder_str)
        if not folder.exists():
            self._set_json_headers(200)
            self._safe_write(json.dumps({"status": "error", "message": f"Folder '{folder}' does not exist."}).encode('utf-8'))
            return

        save_settings({"last_csv_folder": folder_str})
        data_engine = get_data_engine()
        try:
            sync_res = data_engine.scan_and_sync(csv_target=str(folder), force_full=force_full)
            self._set_json_headers(200)
            self._safe_write(json.dumps(sync_res).encode('utf-8'))
        except Exception as e:
            log_error(f"Sync failed for folder {folder}", exc=e)
            self._set_json_headers(200)
            self._safe_write(json.dumps({"status": "error", "message": f"동기화 오류: {e}"}).encode('utf-8'))

    def _handle_preview_chart(self, body: dict):
        folder_str = body.get('csv_folder') or body.get('folder_path') or str(RAW_CSV_DIR)
        data_engine = get_data_engine()
        
        total_ticks = data_engine.get_total_ticks()
        folder = self._resolve_folder(folder_str)
        # Only run auto-sync if database is completely empty to prevent blocking chart rendering
        if total_ticks == 0 and folder.exists() and not data_engine.is_syncing:
            try:
                data_engine.scan_and_sync(csv_target=str(folder), force_full=False)
                total_ticks = data_engine.get_total_ticks()
            except Exception as e:
                log_error(f"Auto-sync in preview_chart failed for {folder}", exc=e)

        if total_ticks == 0:
            self._set_json_headers(200)
            self._safe_write(json.dumps({"status": "error", "message": "동기화된 틱데이터가 없습니다. 먼저 CSV 폴더를 동기화하세요."}).encode('utf-8'))
            return

        resample_type = body.get('resample_type', 'TICK').upper()
        if resample_type == 'VOLUME':
            size_val = max(10, int(body.get('size', body.get('volume_size', body.get('start_vol', body.get('start_tick', 1000))))))
            unit_label = "Contracts"
        else:
            resample_type = 'TICK'
            size_val = max(10, int(body.get('size', body.get('tick_size', body.get('start_tick', 1000)))))
            unit_label = "Ticks"

        period_mode = str(body.get('period_mode', 'DAY')).upper()
        target_date = body.get('target_date')
        filter_outliers = bool(body.get('filter_outliers', True))
        start_date = None
        end_date = None

        if period_mode == 'ALL' or target_date == 'ALL':
            start_date = None
            end_date = None
            target_date = "ALL"
        elif period_mode == 'YEAR':
            yr = body.get('target_year')
            if not yr:
                dates = data_engine.get_available_dates()
                yr = str(dates[-1])[:4] if dates else "2026"
            if yr != "ALL":
                start_date = f"{yr}-01-01 00:00:00"
                end_date = f"{yr}-12-31 23:59:59"
                target_date = f"{yr}년"
        elif period_mode == 'MONTH':
            mo = body.get('target_month')
            if not mo:
                dates = data_engine.get_available_dates()
                mo = str(dates[-1])[:7] if dates else "2026-08"
            if mo != "ALL":
                start_date = f"{mo}-01 00:00:00"
                end_date = f"{mo}-31 23:59:59"
                target_date = f"{mo}월"
        elif period_mode == 'CUSTOM':
            sd = body.get('start_date')
            ed = body.get('end_date')
            if sd:
                start_date = f"{str(sd)[:10]} 00:00:00"
            if ed:
                end_date = f"{str(ed)[:10]} 23:59:59"
            target_date = f"{str(sd)[:10]} ~ {str(ed)[:10]}"
        else:
            # Default: DAY mode
            if target_date and target_date != 'ALL':
                target_date = str(target_date)[:10]
            else:
                dates = data_engine.get_available_dates()
                target_date = str(dates[-1])[:10] if dates else None
            if target_date:
                start_date = f"{target_date} 00:00:00"
                end_date = f"{target_date} 23:59:59"

        resampler = Resampler(data_engine=data_engine)
        if resample_type == 'VOLUME':
            bars_df = resampler.resample_volume_from_db(
                volume_size=size_val,
                symbol="KOSPI_F",
                start_time=start_date,
                end_time=end_date,
                filter_outliers=filter_outliers
            )
        else:
            bars_df = resampler.resample_from_db(
                tick_size=size_val,
                symbol="KOSPI_F",
                start_time=start_date,
                end_time=end_date,
                filter_outliers=filter_outliers
            )

        if bars_df.empty:
            self._set_json_headers(200)
            target_desc = f"{target_date} " if target_date else ""
            self._safe_write(json.dumps({"status": "error", "message": f"{target_desc}선택 조건({size_val:,} {unit_label})에 해당하는 틱 캔들 데이터가 없습니다."}).encode('utf-8'))
            return

        # Technical indicators calculation
        indicators_data = resampler.compute_chart_indicators(bars_df)

        # Count filtered outliers in this timeframe
        outliers_count = resampler.count_outliers(start_time=start_date, end_time=end_date) if filter_outliers else 0

        # Boundaries for Kiwoom chart style vertical separation lines
        timestamps_str = [str(ts) for ts in bars_df['timestamp']]
        trade_dates = [ts[:10] for ts in timestamps_str]
        unique_days = []
        seen_days = set()
        for idx, d in enumerate(trade_dates):
            if d not in seen_days:
                seen_days.add(d)
                unique_days.append((idx, d))

        day_boundaries = []
        # Case 1: Short period (<= 10 trading days) -> Clean day boundaries
        if len(unique_days) <= 10:
            for idx, d in unique_days[1:]:
                day_boundaries.append({
                    "index": idx,
                    "timestamp": timestamps_str[idx],
                    "date": f"{d[5:7]}/{d[8:10]}",
                    "boundary_type": "day"
                })
        # Case 2: Multi-month / Year / All period (> 10 trading days) -> Monthly boundaries without overlap
        else:
            seen_months = set()
            for idx, d in unique_days:
                ym = d[:7]
                if ym not in seen_months:
                    seen_months.add(ym)
                    if idx > 0:
                        month_num = int(d[5:7])
                        day_boundaries.append({
                            "index": idx,
                            "timestamp": timestamps_str[idx],
                            "date": f"{month_num}월",
                            "boundary_type": "month"
                        })

        # Global High / Low stats
        high_vals = bars_df['high'].values
        low_vals = bars_df['low'].values
        max_idx = int(np.argmax(high_vals))
        min_idx = int(np.argmin(low_vals))

        max_info = {
            "price": round(float(high_vals[max_idx]), 2),
            "timestamp": timestamps_str[max_idx],
            "index": max_idx
        }
        min_info = {
            "price": round(float(low_vals[min_idx]), 2),
            "timestamp": timestamps_str[min_idx],
            "index": min_idx
        }

        # Prepare chart payload
        chart_data = {
            "timestamps": timestamps_str,
            "open": bars_df['open'].tolist(),
            "high": bars_df['high'].tolist(),
            "low": bars_df['low'].tolist(),
            "close": bars_df['close'].tolist(),
            "volume": [], # Omit volume as requested for pure price action
            "vwap": bars_df['vwap'].tolist() if 'vwap' in bars_df.columns else None,
            "tick_count": bars_df['tick_count'].tolist() if 'tick_count' in bars_df.columns else [],
            "day_boundaries": day_boundaries,
            "max_info": max_info,
            "min_info": min_info,
            "indicators": indicators_data
        }

        resp = {
            "status": "success",
            "resample_type": resample_type,
            "unit_label": unit_label,
            "size": size_val,
            "total_bars": len(bars_df),
            "period_mode": period_mode,
            "target_period": target_date,
            "first_date": str(bars_df['open_time'].iloc[0]),
            "last_date": str(bars_df['close_time'].iloc[-1]),
            "outliers_excluded": outliers_count,
            "chart_data": chart_data
        }

        self._set_json_headers(200)
        self._safe_write(json.dumps(resp).encode('utf-8'))

    def _handle_backtest(self, body: dict):
        from engine.tracker import tracker
        from config import save_settings
        tracker.reset()
        csv_folder_str = body.get('csv_folder') or body.get('folder_path') or str(RAW_CSV_DIR)
        csv_folder = self._resolve_folder(csv_folder_str)
        save_settings({"last_csv_folder": csv_folder_str})

        start_tick = int(body.get('start_tick', 1000))
        end_tick = int(body.get('end_tick', 5000))
        step_tick = int(body.get('step_tick', 1000))
        cat_str = body.get('category', 'ALL').upper()

        # Date / Period Filtering
        start_date = body.get('start_date')
        end_date = body.get('end_date')
        target_year = body.get('target_year')
        target_month = body.get('target_month')
        target_date = body.get('target_date')

        if target_date and target_date != "ALL":
            if len(target_date) == 10:
                start_date = f"{target_date} 00:00:00"
                end_date = f"{target_date} 23:59:59"
            elif " ~ " in target_date:
                parts = target_date.split(" ~ ")
                start_date = f"{parts[0]} 00:00:00" if len(parts[0]) == 10 else parts[0]
                end_date = f"{parts[1]} 23:59:59" if len(parts[1]) == 10 else parts[1]
        elif target_year and target_year != "ALL":
            start_date = f"{target_year}-01-01 00:00:00"
            end_date = f"{target_year}-12-31 23:59:59"
        elif target_month and target_month != "ALL":
            start_date = f"{target_month}-01 00:00:00"
            end_date = f"{target_month}-31 23:59:59"
        else:
            if start_date:
                start_date = f"{start_date} 00:00:00" if len(start_date) == 10 else start_date
            if end_date:
                end_date = f"{end_date} 23:59:59" if len(end_date) == 10 else end_date

        cfg = BacktestConfig(
            initial_capital=float(body.get('initial_capital', 100_000_000.0)),
            commission_rate=float(body.get('commission_rate', 0.00003)),
            slippage_ticks=float(body.get('slippage_ticks', 1.0)),
            eod_close_time=body.get('eod_close_time', '15:35:00')
        )

        category_map = {
            'TREND': StrategyCategory.TREND,
            'MOMENTUM': StrategyCategory.MOMENTUM,
            'VOLATILITY': StrategyCategory.VOLATILITY,
            'CONTRARIAN': StrategyCategory.CONTRARIAN,
            'ALL': StrategyCategory.ALL
        }

        selected_category = category_map.get(cat_str, StrategyCategory.ALL)

        # 1. Sync data folder first (incremental)
        data_engine = get_data_engine()
        data_engine.scan_and_sync(csv_folder=str(csv_folder))

        if tracker.is_cancelled:
            self._set_json_headers(200)
            self._safe_write(json.dumps({"status": "cancelled", "message": "Backtest cancelled."}).encode('utf-8'))
            return

        # 2. Resample Bars (Tick or Volume)
        resampler = Resampler(data_engine=data_engine)
        resample_type = body.get('resample_type', 'TICK').upper()

        if resample_type == 'VOLUME':
            start_val = max(10, int(body.get('start_vol', body.get('start_tick', 500))))
            end_val = max(10, int(body.get('end_vol', body.get('end_tick', 2500))))
            step_val = int(body.get('step_vol', body.get('step_tick', 500)) or 500)
            unit_label = "Contracts"
        else:
            resample_type = 'TICK'
            start_val = max(10, int(body.get('start_tick', 1000)))
            end_val = max(10, int(body.get('end_tick', 5000)))
            step_val = int(body.get('step_tick', 1000) or 1000)
            unit_label = "Ticks"

        if start_val > end_val:
            start_val, end_val = end_val, start_val

        if start_val == end_val or step_val <= 0:
            sizes = [start_val]
        else:
            sizes = list(range(start_val, end_val + 1, step_val))
            if not sizes:
                sizes = [start_val]
            elif sizes[-1] < end_val:
                sizes.append(end_val)

        if resample_type == 'VOLUME':
            tick_bars = resampler.resample_volume_multiple(
                volume_sizes=sizes,
                symbol="KOSPI_F",
                start_time=start_date,
                end_time=end_date
            )
        else:
            tick_bars = resampler.resample_multiple(
                tick_sizes=sizes, 
                symbol="KOSPI_F",
                start_time=start_date,
                end_time=end_date
            )

        if not tick_bars or all(df.empty for df in tick_bars.values()):
            self._set_json_headers(200)
            self._safe_write(json.dumps({"status": "error", "message": f"선택한 기간/폴더에 데이터가 없습니다 ({resample_type} Resampling)."}).encode('utf-8'))
            return

        if tracker.is_cancelled:
            self._set_json_headers(200)
            self._safe_write(json.dumps({"status": "cancelled", "message": "Backtest cancelled."}).encode('utf-8'))
            return

        # 3. Execute Vectorized Backtesting
        selected_strategies = body.get('selected_strategies')
        if not selected_strategies and body.get('strategy'):
            selected_strategies = [body.get('strategy')]
        elif not selected_strategies and body.get('strategies'):
            selected_strategies = body.get('strategies')

        backtester = Backtester(config=cfg)
        summary_df, detailed_results = backtester.run_multi_tick_backtest(
            tick_bars_dict=tick_bars,
            category=selected_category,
            selected_strategies=selected_strategies
        )


        if summary_df.empty:
            self._set_json_headers(200)
            self._safe_write(json.dumps({"status": "error", "message": "No trades generated in the selected period."}).encode('utf-8'))
            return

        top_result = detailed_results[int(summary_df.iloc[0]['Tick Size'])][str(summary_df.iloc[0]['Strategy'])]

        # 4. Generate Interactive Chart Data & Detailed Results Map
        all_results_dict = {}
        for t_size, strat_map in detailed_results.items():
            c_bars = tick_bars.get(t_size)
            if c_bars is None: continue
            for s_name, res_obj in strat_map.items():
                key = f"{t_size}_{s_name}"
                all_results_dict[key] = {
                    "strategy_name": res_obj.strategy_name,
                    "tick_size": res_obj.tick_size,
                    "total_return_pct": res_obj.total_return_pct,
                    "sharpe_ratio": res_obj.sharpe_ratio,
                    "sortino_ratio": res_obj.sortino_ratio,
                    "max_drawdown_pct": res_obj.max_drawdown_pct,
                    "win_rate": res_obj.win_rate,
                    "profit_factor": res_obj.profit_factor,
                    "total_trades": res_obj.total_trades,
                    "constituent_strategies": res_obj.constituent_strategies,
                    "yearly_breakdown": res_obj.yearly_breakdown.to_dict(orient='records') if not res_obj.yearly_breakdown.empty else [],
                    "monthly_breakdown": res_obj.monthly_breakdown.to_dict(orient='records') if not res_obj.monthly_breakdown.empty else [],
                    "trades": res_obj.trade_log.to_dict(orient='records') if not res_obj.trade_log.empty else [],
                    "chart_data": {
                        "timestamps": [str(ts) for ts in c_bars['timestamp']],
                        "open": c_bars['open'].tolist(),
                        "high": c_bars['high'].tolist(),
                        "low": c_bars['low'].tolist(),
                        "close": c_bars['close'].tolist(),
                        "volume": c_bars['volume'].tolist(),
                        "vwap": c_bars['vwap'].tolist() if 'vwap' in c_bars else None,
                        "buy_signals": {
                            "x": [str(r['entry_time']) for _, r in res_obj.trade_log[res_obj.trade_log['side'] == 'LONG'].iterrows()] if not res_obj.trade_log.empty else [],
                            "y": [float(r['entry_price']) for _, r in res_obj.trade_log[res_obj.trade_log['side'] == 'LONG'].iterrows()] if not res_obj.trade_log.empty else []
                        },
                        "sell_signals": {
                            "x": [str(r['entry_time']) for _, r in res_obj.trade_log[res_obj.trade_log['side'] == 'SHORT'].iterrows()] if not res_obj.trade_log.empty else [],
                            "y": [float(r['entry_price']) for _, r in res_obj.trade_log[res_obj.trade_log['side'] == 'SHORT'].iterrows()] if not res_obj.trade_log.empty else []
                        },
                        "equity_curve": res_obj.equity_curve['equity_pct'].tolist() if not res_obj.equity_curve.empty else [],
                        "day_boundaries": [
                            {"index": idx, "timestamp": str(c_bars['timestamp'].iloc[idx]), "date": str(c_bars['timestamp'].iloc[idx])[:10]}
                            for idx in range(1, len(c_bars))
                            if str(c_bars['timestamp'].iloc[idx])[:10] != str(c_bars['timestamp'].iloc[idx - 1])[:10]
                        ],
                        "max_info": {
                            "price": round(float(c_bars['high'].max()), 2),
                            "timestamp": str(c_bars['timestamp'].iloc[int(np.argmax(c_bars['high'].values))]),
                            "index": int(np.argmax(c_bars['high'].values))
                        } if not c_bars.empty else None,
                        "min_info": {
                            "price": round(float(c_bars['low'].min()), 2),
                            "timestamp": str(c_bars['timestamp'].iloc[int(np.argmin(c_bars['low'].values))]),
                            "index": int(np.argmin(c_bars['low'].values))
                        } if not c_bars.empty else None
                    }
                }

        chart_data = all_results_dict.get(f"{top_result.tick_size}_{top_result.strategy_name}", {}).get("chart_data", {})

        # 5. Background PDF Generation
        pdf_path = REPORTS_DIR / "Quant_DayTrading_Backtest_Report.pdf"
        top_3 = []
        for i in range(min(3, len(summary_df))):
            r = summary_df.iloc[i]
            top_3.append(detailed_results[int(r['Tick Size'])][str(r['Strategy'])])
            
        ReportGenerator.create_pdf_report(
            summary_df=summary_df,
            best_results=top_3,
            output_pdf_path=str(pdf_path)
        )

        response_payload = {
            "status": "success",
            "resample_type": resample_type,
            "unit_label": unit_label,
            "top_result": {
                "strategy_name": top_result.strategy_name,
                "tick_size": top_result.tick_size,
                "total_return_pct": top_result.total_return_pct,
                "sharpe_ratio": top_result.sharpe_ratio,
                "sortino_ratio": top_result.sortino_ratio,
                "max_drawdown_pct": top_result.max_drawdown_pct,
                "win_rate": top_result.win_rate,
                "profit_factor": top_result.profit_factor,
                "total_trades": top_result.total_trades,
                "constituent_strategies": top_result.constituent_strategies,
                "yearly_breakdown": top_result.yearly_breakdown.to_dict(orient='records') if not top_result.yearly_breakdown.empty else [],
                "monthly_breakdown": top_result.monthly_breakdown.to_dict(orient='records') if not top_result.monthly_breakdown.empty else [],
                "trades": top_result.trade_log.to_dict(orient='records') if not top_result.trade_log.empty else []
            },
            "summary": summary_df.to_dict(orient='records'),
            "all_results": all_results_dict,
            "chart_data": chart_data
        }

        self._set_json_headers(200)
        self._safe_write(json.dumps(response_payload).encode('utf-8'))

    def _handle_export_csv(self, parsed):
        params = parse_qs(parsed.query)
        export_type = params.get('type', ['ticks'])[0]
        resample_type = params.get('resample_type', ['TICK'])[0].upper()
        size_val = int(params.get('tick_size', params.get('volume_size', [1000]))[0])
        symbol = params.get('symbol', ['KOSPI_F'])[0]
        year = params.get('year', [None])[0]
        month = params.get('month', [None])[0]
        start_date = params.get('start_date', [None])[0]
        end_date = params.get('end_date', [None])[0]

        if year and year != 'ALL':
            start_date = f"{year}-01-01 00:00:00"
            end_date = f"{year}-12-31 23:59:59"
        elif month and month != 'ALL':
            start_date = f"{month}-01 00:00:00"
            end_date = f"{month}-31 23:59:59"

        try:
            data_engine = get_data_engine()
            from engine.cdt_converter import CDTConverter
            import shutil
            
            if export_type in ('bars', 'volume_bars', 'cdt_bars'):
                resampler = Resampler(data_engine=data_engine)
                if resample_type == 'VOLUME' or export_type == 'volume_bars':
                    bars_df = resampler.resample_volume_from_db(
                        volume_size=size_val,
                        symbol=symbol,
                        start_time=start_date,
                        end_time=end_date
                    )
                    base_name = f"OHLCV_{size_val}VolumeContracts_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                else:
                    bars_df = resampler.resample_from_db(
                        tick_size=size_val,
                        symbol=symbol,
                        start_time=start_date,
                        end_time=end_date
                    )
                    base_name = f"OHLCV_{size_val}Tick_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                if bars_df.empty:
                    self._set_json_headers(404)
                    self._safe_write(json.dumps({"status": "error", "message": "No OHLCV bars to export."}).encode('utf-8'))
                    return

                if export_type == 'cdt_bars':
                    csv_bytes = CDTConverter.dataframe_to_cdt_bytes(bars_df)
                    filename = f"{base_name}.cdt"
                else:
                    csv_bytes = bars_df.to_csv(index=False).encode('utf-8')
                    filename = f"{base_name}.csv"

                self.send_response(200)
                content_type = 'application/octet-stream' if filename.endswith('.cdt') else 'text/csv; charset=utf-8'
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
                self.send_header('Content-Length', str(len(csv_bytes)))
                self.end_headers()
                self._safe_write(csv_bytes)

            elif export_type == 'cdt_ticks':
                ticks_df = data_engine.load_ticks_df(
                    symbol=symbol,
                    start_time=start_date,
                    end_time=end_date
                )
                if ticks_df.empty:
                    self._set_json_headers(404)
                    self._safe_write(json.dumps({"status": "error", "message": "No ticks to export for CDT."}).encode('utf-8'))
                    return
                csv_bytes = CDTConverter.dataframe_to_cdt_bytes(ticks_df)
                filename = f"Unified_Ticks_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.cdt"

                self.send_response(200)
                self.send_header('Content-Type', 'application/octet-stream')
                self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
                self.send_header('Content-Length', str(len(csv_bytes)))
                self.end_headers()
                self._safe_write(csv_bytes)

            else:
                # Standard Unified Ticks CSV Export
                export_path = OUTPUT_DIR / f"Unified_Ticks_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                res = data_engine.export_ticks_to_csv(
                    output_path=export_path,
                    symbol=symbol,
                    start_time=start_date,
                    end_time=end_date
                )
                if res.get('status') == 'error' or not export_path.exists():
                    self._set_json_headers(404)
                    self._safe_write(json.dumps({"status": "error", "message": res.get('message', 'No ticks to export.')}).encode('utf-8'))
                    return

                file_size = export_path.stat().st_size
                self.send_response(200)
                self.send_header('Content-Type', 'text/csv; charset=utf-8')
                self.send_header('Content-Disposition', f'attachment; filename="{export_path.name}"')
                self.send_header('Content-Length', str(file_size))
                self.end_headers()

                with open(export_path, 'rb') as f:
                    shutil.copyfileobj(f, self.wfile, length=65536)

        except Exception as e:
            print(f"[Export Error]: {e}")
            try:
                self._set_json_headers(500)
                self._safe_write(json.dumps({"status": "error", "message": f"Export failed: {e}"}).encode('utf-8'))
            except Exception:
                pass
        except Exception:
            pass

    def _handle_download_pdf(self, parsed=None):
        qs = parse_qs(parsed.query) if parsed else {}
        market = (qs.get('market', ['DOMESTIC'])[0]).upper()

        if market == 'OVERSEAS':
            pdf_path = REPORTS_DIR / "Quant_Overseas_Backtest_Report.pdf"
            filename = "Quant_Overseas_Backtest_Report.pdf"
        else:
            pdf_path = REPORTS_DIR / "Quant_DayTrading_Backtest_Report.pdf"
            filename = "Quant_DayTrading_Backtest_Report.pdf"

        # Fallback if requested report not found but the other exists
        if not pdf_path.exists():
            alt_path = REPORTS_DIR / ("Quant_DayTrading_Backtest_Report.pdf" if market == 'OVERSEAS' else "Quant_Overseas_Backtest_Report.pdf")
            if alt_path.exists():
                pdf_path = alt_path
                filename = alt_path.name

        if pdf_path.exists():
            with open(pdf_path, 'rb') as f:
                content = f.read()
            try:
                self.send_response(200)
                self.send_header('Content-Type', 'application/pdf')
                self.send_header('Content-Disposition', f'inline; filename="{filename}"')
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self._safe_write(content)
            except Exception:
                pass
        else:
            tab_name = "해외선물" if market == 'OVERSEAS' else "국내선물"
            self.send_error(404, f"[{tab_name}] 백테스트 PDF 리포트가 아직 생성되지 않았습니다. 먼저 해당 탭에서 백테스트를 실행해주세요.")

    def _handle_get_logs(self, parsed):
        from config import LOGS_DIR
        qs = parse_qs(parsed.query)
        log_type = (qs.get('type', ['error'])[0]).lower()
        if log_type not in ['error', 'sync', 'app']:
            log_type = 'error'
        max_lines = int(qs.get('lines', [300])[0])

        log_file = LOGS_DIR / f"{log_type}.log"
        content = ""
        file_size = 0
        if log_file.exists():
            file_size = log_file.stat().st_size
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    if max_lines and len(lines) > max_lines:
                        lines = lines[-max_lines:]
                    content = "".join(lines)
            except Exception as e:
                content = f"로그 파일 읽기 오류: {e}"
        else:
            content = f"기록된 로그가 없습니다. (파일: {log_file.name})"

        self._set_json_headers(200)
        self._safe_write(json.dumps({
            "status": "success",
            "log_type": log_type,
            "available_types": ["error", "sync", "app"],
            "content": content,
            "file_size": file_size,
            "file_path": str(log_file)
        }).encode('utf-8'))

    def _handle_clear_logs(self, body: dict):
        from config import LOGS_DIR
        log_type = (body.get('type', 'error')).lower()
        cleared = []
        if log_type == 'all':
            for name in ['error.log', 'sync.log', 'app.log']:
                f = LOGS_DIR / name
                if f.exists():
                    try:
                        open(f, 'w', encoding='utf-8').close()
                        cleared.append(name)
                    except Exception:
                        pass
        else:
            f = LOGS_DIR / f"{log_type}.log"
            if f.exists():
                try:
                    open(f, 'w', encoding='utf-8').close()
                    cleared.append(f.name)
                except Exception:
                    pass

        self._set_json_headers(200)
        self._safe_write(json.dumps({
            "status": "success",
            "message": f"로그 파일이 초기화되었습니다: {', '.join(cleared) if cleared else 'None'}"
        }).encode('utf-8'))

    def _handle_kiwoom_status(self):
        """Returns Kiwoom connection status and real-time domestic futures state."""
        from engine.kiwoom_manager import get_kiwoom_manager
        km = get_kiwoom_manager()
        self._set_json_headers(200)
        self._safe_write(json.dumps(km.get_status(), ensure_ascii=False).encode('utf-8'))

    def _handle_kiwoom_config(self):
        """Returns saved Kiwoom credentials with passwords masked."""
        from config import load_kiwoom_credentials
        creds = load_kiwoom_credentials(mask_passwords=True)
        self._set_json_headers(200)
        self._safe_write(json.dumps({"status": "success", "config": creds}, ensure_ascii=False).encode('utf-8'))

    def _handle_kiwoom_save_config(self, body: dict):
        """Saves Kiwoom credentials with secure obfuscation and remember flags."""
        from config import save_kiwoom_credentials
        from engine.kiwoom_manager import get_kiwoom_manager
        km = get_kiwoom_manager()
        save_kiwoom_credentials(body)
        if 'tick_size' in body and body['tick_size']:
            km.set_tick_size(int(body['tick_size']))
        self._set_json_headers(200)
        self._safe_write(json.dumps({"status": "success", "message": "키움 계정 및 접속 설정이 저장되었습니다."}, ensure_ascii=False).encode('utf-8'))

    def _handle_kiwoom_connect(self, body: dict):
        """Initiates Kiwoom OpenAPI server connection with credentials memory."""
        from engine.kiwoom_manager import get_kiwoom_manager
        km = get_kiwoom_manager()
        res = km.connect(
            server_type=body.get('server_type'),
            user_id=body.get('user_id'),
            account_no=body.get('account_no'),
            account_pw=body.get('account_pw'),
            cert_pw=body.get('cert_pw'),
            remember=body.get('remember', True),
            auto_connect=body.get('auto_connect', True)
        )
        self._set_json_headers(200)
        self._safe_write(json.dumps(res, ensure_ascii=False).encode('utf-8'))

    def _handle_kiwoom_disconnect(self):
        """Disconnects from Kiwoom server."""
        from engine.kiwoom_manager import get_kiwoom_manager
        km = get_kiwoom_manager()
        res = km.disconnect()
        self._set_json_headers(200)
        self._safe_write(json.dumps(res, ensure_ascii=False).encode('utf-8'))

    def _handle_kiwoom_realtime_bars(self, parsed):
        """Returns Plotly-compatible real-time candlestick bars and indicators for active tick resolution."""
        from engine.kiwoom_manager import get_kiwoom_manager
        km = get_kiwoom_manager()
        params = parse_qs(parsed.query)
        tick_size = int(params.get('tick_size', [str(km.tick_size)])[0])
        limit = int(params.get('limit', ['300'])[0])
        data = km.get_chart_data(tick_size=tick_size, limit=limit)
        self._set_json_headers(200)
        self._safe_write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

    def _handle_kiwoom_set_tick_size(self, body: dict):
        """Changes active real-time tick resolution and rebuilds bars dynamically."""
        from engine.kiwoom_manager import get_kiwoom_manager
        km = get_kiwoom_manager()
        new_size = int(body.get('tick_size', 60))
        size = km.set_tick_size(new_size)
        self._set_json_headers(200)
        self._safe_write(json.dumps({"status": "success", "tick_size": size}, ensure_ascii=False).encode('utf-8'))

    def _handle_kiwoom_toggle_streaming(self, body: dict):
        """Controls real-time simulation play/pause and streaming speed."""
        from engine.kiwoom_manager import get_kiwoom_manager
        km = get_kiwoom_manager()
        if 'speed' in body:
            km.set_feeder_speed(float(body['speed']))
        if 'paused' in body:
            is_paused = km.toggle_feeder(paused=body['paused'])
        else:
            is_paused = km.toggle_feeder()
        self._set_json_headers(200)
        self._safe_write(json.dumps({
            "status": "success",
            "is_paused": is_paused,
            "speed": km.feeder_speed
        }, ensure_ascii=False).encode('utf-8'))

    # =========================================================================
    # Deep Learning & Transfer Learning Handlers
    # =========================================================================
    def _handle_dl_get_config(self):
        from deep_learning.config_loader import load_config
        cfg = load_config()
        self._set_json_headers(200)
        self._safe_write(json.dumps(cfg, ensure_ascii=False).encode('utf-8'))

    def _handle_dl_save_config(self, body: dict):
        from deep_learning.config_loader import DEFAULT_CONFIG_PATH
        import yaml
        with open(DEFAULT_CONFIG_PATH, 'w', encoding='utf-8') as f:
            yaml.dump(body, f, allow_unicode=True, sort_keys=False)
        self._set_json_headers(200)
        self._safe_write(json.dumps({"status": "success", "message": "설정이 성공적으로 저장되었습니다."}, ensure_ascii=False).encode('utf-8'))

    def _handle_dl_browse_folder(self):
        """Native Windows FolderBrowserDialog for Deep Learning data_dir."""
        import subprocess
        import base64

        cmd = [
            "powershell", "-NoProfile", "-Command",
            "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$top = New-Object System.Windows.Forms.Form; "
            "$top.TopMost = $true; "
            "$f = New-Object System.Windows.Forms.FolderBrowserDialog; "
            "$f.Description = 'Select Tick CSV Folder for Deep Learning (딥러닝 틱데이터 CSV 폴더 선택)'; "
            "if ($f.ShowDialog($top) -eq [System.Windows.Forms.DialogResult]::OK) { "
            "    $bytes = [System.Text.Encoding]::UTF8.GetBytes($f.SelectedPath); "
            "    [System.Convert]::ToBase64String($bytes) "
            "}"
        ]
        flags = 0x08000000 if sys.platform == 'win32' else 0
        try:
            raw_out = subprocess.check_output(cmd, creationflags=flags).decode('ascii', errors='ignore').strip()
            if raw_out:
                folder_str = base64.b64decode(raw_out).decode('utf-8', errors='replace').strip()
                if folder_str:
                    self._set_json_headers(200)
                    self._safe_write(json.dumps({"status": "success", "folder": folder_str}, ensure_ascii=False).encode('utf-8'))
                    return
        except Exception as e:
            log_error("Error opening FolderBrowserDialog for DL", exc=e)

        self._set_json_headers(200)
        self._safe_write(json.dumps({"status": "cancelled", "folder": None}).encode('utf-8'))

    def _handle_dl_scan_folder(self, body: dict):
        """Scans the specified data_dir using TickDataScanner with full automated integrity verification."""
        folder = body.get("folder", "csv")
        try:
            from deep_learning.data_loader import TickDataScanner
            scanner = TickDataScanner(data_dir=folder)
            valid_files, report = scanner.scan_all_files_with_integrity()
            if not valid_files:
                self._set_json_headers(200)
                self._safe_write(json.dumps({
                    "status": "empty",
                    "count": 0,
                    "corrupted_count": report.corrupted_files_count,
                    "message": f"폴더 '{folder}'에서 유효한 틱데이터 CSV 파일을 찾을 수 없습니다."
                }, ensure_ascii=False).encode('utf-8'))
                return

            start_dt = valid_files[0][0]
            end_dt = valid_files[-1][0]
            
            # Format top corrupted issues if any
            corrupted_details = [
                {"file": r.file_name, "issues": r.issues}
                for r in report.integrity_results if not r.is_valid
            ][:10]

            self._set_json_headers(200)
            self._safe_write(json.dumps({
                "status": "success",
                "count": len(valid_files),
                "total_scanned": report.total_files_scanned,
                "corrupted_count": report.corrupted_files_count,
                "start_date": start_dt,
                "end_date": end_dt,
                "corrupted_details": corrupted_details,
                "message": f"무결성 검사 완료: 유효 {len(valid_files):,}개 / 비정상 {report.corrupted_files_count}개 ({start_dt} ~ {end_dt})"
            }, ensure_ascii=False).encode('utf-8'))
        except Exception as e:
            self._set_json_headers(200)
            self._safe_write(json.dumps({
                "status": "error",
                "message": f"폴더 무결성 검사 오류: {e}"
            }, ensure_ascii=False).encode('utf-8'))

    def _handle_dl_progress(self):
        # Read latest progress or log file
        from config import BASE_DIR
        log_file = BASE_DIR / "logs" / "deep_learning_pipeline.log"
        reg_file = BASE_DIR / "models" / "registry.json"
        log_content = ""
        if log_file.exists():
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    log_content = "".join(lines[-40:])
            except Exception:
                pass
        
        reg_data = []
        if reg_file.exists():
            try:
                with open(reg_file, "r", encoding="utf-8") as rf:
                    reg_data = json.load(rf)
            except Exception:
                reg_data = []

        self._set_json_headers(200)
        self._safe_write(json.dumps({
            "status": "success",
            "log": log_content,
            "registry": reg_data
        }, ensure_ascii=False).encode('utf-8'))

    def _handle_dl_start_pipeline(self, body: dict):
        import subprocess
        from config import BASE_DIR
        script_path = BASE_DIR / "main_pipeline.py"
        flags = 0x08000000 if sys.platform == 'win32' else 0
        cmd = [sys.executable, str(script_path)]
        if body.get("data_dir"):
            cmd.extend(["--data-dir", str(body["data_dir"])])
        if body.get("max_days"):
            cmd.extend(["--max-days", str(body["max_days"])])
        
        subprocess.Popen(cmd, creationflags=flags)
        self._set_json_headers(200)
        self._safe_write(json.dumps({"status": "started", "message": "딥러닝 파이프라인 백그라운드 구동 시작"}, ensure_ascii=False).encode('utf-8'))

    def _handle_dl_stop_pipeline(self):
        self._set_json_headers(200)
        self._safe_write(json.dumps({"status": "stopped", "message": "파이프라인 중지 신호 전달"}, ensure_ascii=False).encode('utf-8'))

    def _handle_dl_reports(self):
        from config import BASE_DIR
        report_csv = BASE_DIR / "output" / "dl_reports" / "model_comparison_matrix.csv"
        data = []
        if report_csv.exists():
            try:
                df = pd.read_csv(report_csv)
                data = df.to_dict(orient="records")
            except Exception:
                data = []
        self._set_json_headers(200)
        self._safe_write(json.dumps({"status": "success", "reports": data}, ensure_ascii=False).encode('utf-8'))




def start_server(port: int = 5000, auto_open: bool = True):
    """Starts the local web server and launches browser."""
    server_address = ('127.0.0.1', port)
    httpd = RobustThreadingServer(server_address, QuantRequestHandler)
    url = f"http://127.0.0.1:{port}"
    print("\n" + "=" * 75)
    print(" [GLASSMORPHISM DARK TERMINAL RUNNING]")
    print(f" [+] URL: {url}")
    print(" [+] Press Ctrl+C in console to stop the server.")
    print("=" * 75 + "\n")

    if auto_open:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Server stopped.")
        httpd.server_close()
