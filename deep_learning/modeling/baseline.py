"""
Baseline Machine Learning Trading Models
- Logistic Regression
- Random Forest Classifier / Regressor
"""
from typing import Dict, Any, Tuple
import time
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler


class BaselineModelSuite:
    def __init__(self, task_type: str = "multiclass"):
        self.task_type = task_type
        self.scaler = StandardScaler()
        self.models = {}

    def train_logistic(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[Any, float]:
        """Trains Logistic Regression / Ridge with training elapsed time (s)."""
        t0 = time.time()
        X_scaled = self.scaler.fit_transform(X_train)
        if self.task_type == "multiclass":
            model = LogisticRegression(max_iter=300, C=1.0, random_state=42)
        else:
            model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_scaled, y_train)
        elapsed = time.time() - t0
        return model, elapsed

    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[Any, float]:
        """Trains Random Forest with elapsed time."""
        t0 = time.time()
        if self.task_type == "multiclass":
            model = RandomForestClassifier(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        return model, elapsed

    def predict(self, model: Any, X_test: np.ndarray, is_linear: bool = False) -> Tuple[np.ndarray, float]:
        """Runs inference and returns predictions + latency (ms)."""
        t0 = time.time()
        X_in = self.scaler.transform(X_test) if is_linear else X_test
        preds = model.predict(X_in)
        latency_ms = (time.time() - t0) * 1000.0 / max(1, len(X_test))
        return preds, latency_ms
