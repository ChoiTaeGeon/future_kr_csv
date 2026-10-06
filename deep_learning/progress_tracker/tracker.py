"""
Progress Tracker & ETA Estimator
Real-time tracking of multi-stage pipelines (Stage % and Overall ETA).
Supports console updates (tqdm), log file appending, and web-ready JSON status polling.
"""
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
from tqdm import tqdm


class PipelineProgressTracker:
    def __init__(
        self,
        total_stages: int = 8,
        log_file_path: Optional[str | Path] = None,
        update_interval_sec: float = 1.0
    ):
        self.total_stages = total_stages
        self.current_stage = 0
        self.current_stage_name = "Ready"
        self.stage_start_time = time.time()
        self.pipeline_start_time = time.time()
        self.log_file = Path(log_file_path) if log_file_path else None
        self.update_interval_sec = update_interval_sec
        self.last_update_ts = 0.0

        # Metrics for ETA estimation
        self.stage_progress_pct = 0.0
        self.overall_progress_pct = 0.0
        self.estimated_remaining_sec = 0.0
        self.active_tqdm: Optional[tqdm] = None

        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*70}\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Deep Learning Pipeline Initialized\n{'='*70}\n")

    def log(self, message: str):
        """Append log to console and log file."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{now_str}] {message}"
        tqdm.write(log_line)
        if self.log_file:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(log_line + "\n")
            except Exception:
                pass

    def start_stage(self, stage_num: int, stage_name: str, total_steps: int = 100) -> tqdm:
        """Begin a new stage with tqdm progress bar."""
        if self.active_tqdm:
            self.active_tqdm.close()

        self.current_stage = stage_num
        self.current_stage_name = stage_name
        self.stage_start_time = time.time()
        self.stage_progress_pct = 0.0

        self.log(f"▶ [Stage {stage_num}/{self.total_stages}] {stage_name} (Total items: {total_steps})")

        self.active_tqdm = tqdm(
            total=total_steps,
            desc=f"Stage {stage_num}: {stage_name[:20]}",
            unit="item",
            dynamic_ncols=True,
            leave=True
        )
        return self.active_tqdm

    def update_stage_step(self, n: int = 1, desc_extra: str = ""):
        """Advance progress and update overall ETA."""
        if self.active_tqdm:
            self.active_tqdm.update(n)
            if desc_extra:
                self.active_tqdm.set_postfix_str(desc_extra)

            # Calculate progress percentages
            total = max(1, self.active_tqdm.total or 1)
            completed = self.active_tqdm.n
            self.stage_progress_pct = (completed / total) * 100.0

            # Overall progress calculation across stages
            stage_weight = 1.0 / max(1, self.total_stages)
            completed_stages_pct = (self.current_stage - 1) * stage_weight * 100.0
            current_stage_contrib = (self.stage_progress_pct / 100.0) * stage_weight * 100.0
            self.overall_progress_pct = min(100.0, completed_stages_pct + current_stage_contrib)

            # Real-time ETA estimation based on overall elapsed time
            elapsed = time.time() - self.pipeline_start_time
            if self.overall_progress_pct > 0.5:
                total_estimated_sec = elapsed / (self.overall_progress_pct / 100.0)
                self.estimated_remaining_sec = max(0.0, total_estimated_sec - elapsed)
            else:
                self.estimated_remaining_sec = 0.0

            now = time.time()
            if now - self.last_update_ts >= self.update_interval_sec:
                self.last_update_ts = now
                eta_str = str(timedelta(seconds=int(self.estimated_remaining_sec)))
                if self.log_file:
                    try:
                        with open(self.log_file, "a", encoding="utf-8") as f:
                            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] S{self.current_stage} {completed}/{total} ({self.stage_progress_pct:.1f}%) | Overall: {self.overall_progress_pct:.1f}% | ETA: {eta_str}\n")
                    except Exception:
                        pass

    def finish_stage(self, summary_msg: str = ""):
        """Mark current stage complete."""
        if self.active_tqdm:
            self.active_tqdm.close()
            self.active_tqdm = None
        duration = time.time() - self.stage_start_time
        msg = f"✓ [Stage {self.current_stage}/{self.total_stages}] {self.current_stage_name} Finished ({duration:.1f}s)"
        if summary_msg:
            msg += f" - {summary_msg}"
        self.log(msg)

    def finish_pipeline(self):
        """Mark entire pipeline complete."""
        if self.active_tqdm:
            self.active_tqdm.close()
        total_time = time.time() - self.pipeline_start_time
        self.overall_progress_pct = 100.0
        self.estimated_remaining_sec = 0.0
        self.log(f"★ All Stages Complete! Total Elapsed Time: {str(timedelta(seconds=int(total_time)))}")

    def get_status_dict(self) -> Dict[str, Any]:
        """JSON-serializable status for REST API endpoints."""
        return {
            "current_stage": self.current_stage,
            "total_stages": self.total_stages,
            "stage_name": self.current_stage_name,
            "stage_progress_pct": round(self.stage_progress_pct, 1),
            "overall_progress_pct": round(self.overall_progress_pct, 1),
            "elapsed_sec": round(time.time() - self.pipeline_start_time, 1),
            "eta_sec": int(self.estimated_remaining_sec),
            "eta_formatted": str(timedelta(seconds=int(self.estimated_remaining_sec)))
        }
