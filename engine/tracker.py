import time
import threading
from typing import Dict, Any, Optional

class ProgressTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self.is_running = False
            self.is_cancelled = False
            self.stage = 'IDLE'
            self.message = 'Ready'
            self.current_step = 0
            self.total_steps = 1
            self.start_time = None
            self.last_update_time = None
            self.elapsed_sec = 0.0
            self.eta_sec = 0.0
            self.percent = 0.0
            self.details = ''

    def cancel(self):
        with self._lock:
            self.is_cancelled = True
            self.is_running = False
            self.stage = 'CANCELLED'
            self.message = 'Process stopped by user.'
            self.eta_sec = 0.0

    def start(self, stage: str, total_steps: int, message: str = ''):
        with self._lock:
            self.is_running = True
            self.is_cancelled = False
            self.stage = stage
            self.message = message or f'Starting {stage}...'
            self.current_step = 0
            self.total_steps = max(1, total_steps)
            self.start_time = time.time()
            self.last_update_time = self.start_time
            self.elapsed_sec = 0.0
            self.eta_sec = 0.0
            self.percent = 0.0
            self.details = ''

    def update(self, current_step: int, message: Optional[str] = None, details: Optional[str] = None):
        with self._lock:
            if not self.is_running or self.start_time is None:
                return
            self.current_step = min(current_step, self.total_steps)
            now = time.time()
            self.elapsed_sec = now - self.start_time
            self.last_update_time = now
            if message is not None:
                self.message = message
            if details is not None:
                self.details = details
            self.percent = round((self.current_step / self.total_steps) * 100.0, 1)
            if self.current_step > 0 and self.percent < 100.0:
                rate = self.elapsed_sec / self.current_step
                rem = self.total_steps - self.current_step
                self.eta_sec = max(0.0, round(rate * rem, 1))
            else:
                self.eta_sec = 0.0

    def step(self, delta: int = 1, message: Optional[str] = None, details: Optional[str] = None):
        with self._lock:
            new_step = self.current_step + delta
        self.update(new_step, message=message, details=details)

    def finish(self, message: str = 'Completed successfully'):
        with self._lock:
            self.is_running = False
            self.percent = 100.0
            self.current_step = self.total_steps
            if self.start_time:
                self.elapsed_sec = time.time() - self.start_time
            self.eta_sec = 0.0
            self.message = message
            self.stage = 'COMPLETED'

    def fail(self, error_message: str):
        with self._lock:
            self.is_running = False
            self.stage = 'ERROR'
            self.message = f'Error: {error_message}'
            self.eta_sec = 0.0

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            if self.is_running and self.start_time:
                self.elapsed_sec = time.time() - self.start_time
                if self.current_step > 0 and self.percent < 100.0:
                    rate = self.elapsed_sec / self.current_step
                    rem = self.total_steps - self.current_step
                    self.eta_sec = max(0.0, round(rate * rem, 1))
            return {
                'is_running': self.is_running,
                'stage': self.stage,
                'message': self.message,
                'current_step': self.current_step,
                'total_steps': self.total_steps,
                'percent': self.percent,
                'elapsed_sec': round(self.elapsed_sec, 1),
                'eta_sec': round(self.eta_sec, 1),
                'details': self.details
            }

tracker = ProgressTracker()
