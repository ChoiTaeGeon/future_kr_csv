"""
Time-Series Validation Engine
- Walk-Forward Rolling Window Splitter (e.g. 2 Years Train + 6 Months Test)
- Purged K-Fold Cross-Validation with Embargo (Prevents label overlap leakage)
"""
from dataclasses import dataclass
from typing import List, Tuple, Generator
import numpy as np
import pandas as pd


@dataclass
class TimeSplit:
    fold_idx: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_range: Tuple[pd.Timestamp, pd.Timestamp]
    test_range: Tuple[pd.Timestamp, pd.Timestamp]


class WalkForwardSplitter:
    def __init__(
        self,
        train_days: int = 500,     # ~2 years of trading days
        test_days: int = 125,      # ~6 months of trading days
        embargo_pct: float = 0.01  # 1% embargo between train and test
    ):
        self.train_days = train_days
        self.test_days = test_days
        self.embargo_pct = embargo_pct

    def split(self, df: pd.DataFrame, datetime_col: str = 'datetime') -> List[TimeSplit]:
        """
        Splits dataset chronologically into multiple rolling train/test folds.
        """
        if len(df) == 0:
            return []

        dates = pd.to_datetime(df[datetime_col]).dt.date.unique()
        dates.sort()
        total_unique_days = len(dates)

        splits: List[TimeSplit] = []
        step_days = self.test_days
        start_idx = 0
        fold = 0

        while start_idx + self.train_days + self.test_days <= total_unique_days:
            train_dates = set(dates[start_idx : start_idx + self.train_days])
            test_dates = set(dates[start_idx + self.train_days : start_idx + self.train_days + self.test_days])

            # Apply embargo: drop initial bars of test set that overlap with train labels
            embargo_count = int(self.test_days * self.embargo_pct)
            if embargo_count > 0:
                test_dates_sorted = sorted(list(test_dates))
                test_dates = set(test_dates_sorted[embargo_count:])

            date_series = pd.to_datetime(df[datetime_col]).dt.date
            train_mask = date_series.isin(train_dates).values
            test_mask = date_series.isin(test_dates).values

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            if len(train_idx) > 0 and len(test_idx) > 0:
                dt_series = pd.to_datetime(df[datetime_col])
                t_train = (dt_series.iloc[train_idx[0]], dt_series.iloc[train_idx[-1]])
                t_test = (dt_series.iloc[test_idx[0]], dt_series.iloc[test_idx[-1]])

                splits.append(TimeSplit(
                    fold_idx=fold,
                    train_indices=train_idx,
                    test_indices=test_idx,
                    train_range=t_train,
                    test_range=t_test
                ))
                fold += 1

            start_idx += step_days

        # Fallback if dataset is shorter than full window
        if not splits and len(df) > 100:
            split_point = int(len(df) * 0.75)
            dt_series = pd.to_datetime(df[datetime_col])
            splits.append(TimeSplit(
                fold_idx=0,
                train_indices=np.arange(0, split_point),
                test_indices=np.arange(split_point, len(df)),
                train_range=(dt_series.iloc[0], dt_series.iloc[split_point - 1]),
                test_range=(dt_series.iloc[split_point], dt_series.iloc[-1])
            ))

        return splits


class PurgedKFoldSplitter:
    """
    Purged K-Fold Cross Validation with Embargo as formulated by Marcos Lopez de Prado.
    """
    def __init__(self, n_splits: int = 4, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, df: pd.DataFrame) -> List[TimeSplit]:
        n = len(df)
        indices = np.arange(n)
        fold_size = n // self.n_splits
        splits = []

        for fold in range(self.n_splits):
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < self.n_splits - 1 else n
            test_idx = indices[test_start:test_end]

            # Purge & Embargo train index
            embargo = int(n * self.embargo_pct)
            train_idx = np.setdiff1d(indices, indices[max(0, test_start - embargo) : min(n, test_end + embargo)])

            splits.append(TimeSplit(
                fold_idx=fold,
                train_indices=train_idx,
                test_indices=test_idx,
                train_range=(pd.to_datetime(df['datetime'].iloc[train_idx[0]]), pd.to_datetime(df['datetime'].iloc[train_idx[-1]])),
                test_range=(pd.to_datetime(df['datetime'].iloc[test_idx[0]]), pd.to_datetime(df['datetime'].iloc[test_idx[-1]]))
            ))
        return splits
