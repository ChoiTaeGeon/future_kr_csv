"""
Deep Learning & Transfer Learning PyTorch Trainer
Supports:
- Sequence Sliding Window Dataset Generator
- Epoch & Iteration progress callback with real-time ETA
- Zero-Shot vs Fine-Tuning evaluation comparison
"""
from typing import Dict, Any, Tuple, Optional, Callable
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from .patch_tst import PatchTST
from .foundation_models import PretrainedTimeSeriesFoundationModel


class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 32, is_multiclass: bool = True):
        self.seq_len = seq_len
        self.is_multiclass = is_multiclass

        # Generate rolling windows
        samples_X = []
        samples_y = []
        n = len(X)
        for i in range(seq_len, n):
            samples_X.append(X[i - seq_len : i])
            samples_y.append(y[i])

        if samples_X:
            self.X = torch.tensor(np.array(samples_X), dtype=torch.float32)
            if is_multiclass:
                # Map [-1, 0, 1] to [0, 1, 2]
                self.y = torch.tensor(np.array(samples_y) + 1, dtype=torch.long)
            else:
                self.y = torch.tensor(np.array(samples_y), dtype=torch.float32).unsqueeze(1)
        else:
            self.X = torch.empty((0, seq_len, X.shape[1] if X.ndim > 1 else 1))
            self.y = torch.empty((0,), dtype=torch.long if is_multiclass else torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DeepLearningTrainer:
    def __init__(
        self,
        task_type: str = "multiclass",
        seq_len: int = 32,
        batch_size: int = 64,
        lr: float = 0.001,
        epochs: int = 15,
        device: str = "cpu"
    ):
        self.task_type = task_type
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device(device)

    def train_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        progress_callback: Optional[Callable[[int, int, float, str], None]] = None
    ) -> Tuple[nn.Module, float]:
        """
        Trains PyTorch model and executes progress_callback(epoch, total_epochs, loss, eta_str)
        """
        t0 = time.time()
        model.to(self.device)
        model.train()

        criterion = nn.CrossEntropyLoss() if self.task_type == "multiclass" else nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)

        for epoch in range(1, self.epochs + 1):
            ep_start = time.time()
            total_loss = 0.0
            batches = 0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                out = model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                batches += 1

            avg_loss = total_loss / max(1, batches)
            ep_time = time.time() - ep_start
            rem_epochs = self.epochs - epoch
            eta_sec = int(ep_time * rem_epochs)
            eta_str = f"{eta_sec}s"

            if progress_callback:
                progress_callback(epoch, self.epochs, avg_loss, eta_str)

        elapsed = time.time() - t0
        return model, elapsed

    def predict(self, model: nn.Module, X_test: np.ndarray) -> Tuple[np.ndarray, float]:
        """Runs batch inference and measures latency."""
        t0 = time.time()
        model.eval()
        model.to(self.device)

        dummy_y = np.zeros(len(X_test))
        test_ds = TimeSeriesDataset(X_test, dummy_y, seq_len=self.seq_len, is_multiclass=(self.task_type == "multiclass"))

        if len(test_ds) == 0:
            return np.array([]), 0.0

        loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)
        all_preds = []

        with torch.no_grad():
            for bx, _ in loader:
                bx = bx.to(self.device)
                out = model(bx)
                if self.task_type == "multiclass":
                    preds = torch.argmax(out, dim=1).cpu().numpy() - 1  # Map back to -1, 0, 1
                else:
                    preds = out.squeeze(1).cpu().numpy()
                all_preds.extend(preds)

        # Pad initial seq_len bars with 0 (neutral) to maintain original length
        padded_preds = np.zeros(len(X_test), dtype=int if self.task_type == "multiclass" else float)
        padded_preds[self.seq_len :] = np.array(all_preds)

        latency_ms = (time.time() - t0) * 1000.0 / max(1, len(X_test))
        return padded_preds, latency_ms
