"""
Tree-Based Models: LightGBM and XGBoost with Optuna Hyperparameter Tuning
Optimized for Financial Time-Series Cross-Validation
"""
from typing import Dict, Any, Tuple, Optional
import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


class TreeModelSuite:
    def __init__(self, task_type: str = "multiclass", optuna_trials: int = 10):
        self.task_type = task_type
        self.optuna_trials = optuna_trials

    def train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Tuple[Any, Dict[str, Any], float]:
        """Trains LightGBM model with Optuna parameter search."""
        t0 = time.time()
        # Map labels for LightGBM multiclass: [-1, 0, 1] -> [0, 1, 2]
        is_multi = (self.task_type == "multiclass")
        y_tr_mapped = (y_train + 1) if is_multi else y_train
        y_va_mapped = (y_val + 1) if (is_multi and y_val is not None) else y_val

        def objective(trial):
            params = {
                'verbosity': -1,
                'random_state': 42,
                'n_estimators': trial.suggest_int('n_estimators', 40, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
            }
            if is_multi:
                clf = lgb.LGBMClassifier(**params)
            else:
                clf = lgb.LGBMRegressor(**params)

            clf.fit(X_train, y_tr_mapped)
            if X_val is not None:
                score = clf.score(X_val, y_va_mapped)
            else:
                score = clf.score(X_train, y_tr_mapped)
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.optuna_trials, show_progress_bar=False)
        best_params = study.best_params
        best_params.update({'verbosity': -1, 'random_state': 42})

        if is_multi:
            best_model = lgb.LGBMClassifier(**best_params)
        else:
            best_model = lgb.LGBMRegressor(**best_params)

        best_model.fit(X_train, y_tr_mapped)
        elapsed = time.time() - t0
        return best_model, best_params, elapsed

    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Tuple[Any, Dict[str, Any], float]:
        """Trains XGBoost model with Optuna parameter search."""
        t0 = time.time()
        is_multi = (self.task_type == "multiclass")
        y_tr_mapped = (y_train + 1) if is_multi else y_train
        y_va_mapped = (y_val + 1) if (is_multi and y_val is not None) else y_val

        def objective(trial):
            params = {
                'verbosity': 0,
                'random_state': 42,
                'n_estimators': trial.suggest_int('n_estimators', 40, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
            }
            if is_multi:
                clf = xgb.XGBClassifier(**params)
            else:
                clf = xgb.XGBRegressor(**params)

            clf.fit(X_train, y_tr_mapped)
            if X_val is not None:
                return clf.score(X_val, y_va_mapped)
            return clf.score(X_train, y_tr_mapped)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.optuna_trials, show_progress_bar=False)
        best_params = study.best_params
        best_params.update({'verbosity': 0, 'random_state': 42})

        if is_multi:
            best_model = xgb.XGBClassifier(**best_params)
        else:
            best_model = xgb.XGBRegressor(**best_params)

        best_model.fit(X_train, y_tr_mapped)
        elapsed = time.time() - t0
        return best_model, best_params, elapsed

    def predict(self, model: Any, X_test: np.ndarray) -> Tuple[np.ndarray, float]:
        """Runs predictions and converts multiclass [0, 1, 2] back to [-1, 0, 1]."""
        t0 = time.time()
        preds = model.predict(X_test)
        if self.task_type == "multiclass":
            preds = preds - 1  # Map back to -1, 0, 1
        latency_ms = (time.time() - t0) * 1000.0 / max(1, len(X_test))
        return preds, latency_ms
