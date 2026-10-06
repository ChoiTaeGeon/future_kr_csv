"""
Master Pipeline Runner for KOSPI 200 Futures Deep Learning & Transfer Learning System
Executes:
1. Scan & Clean ~1,400 Tick CSVs (7 Years)
2. Dynamic Tick Bar Generation (User-specified start/end/step + Coarse-to-Fine)
3. Strict Lag-Shifted Feature Engineering
4. Triple-Barrier Dynamic Volatility Labeling
5. Walk-Forward / Purged K-Fold Cross Validation
6. Multi-Model Grid Training (Baselines, LightGBM/XGBoost, PatchTST, Foundation Models)
7. Vectorized Backtesting with Commissions and Slippage
8. Best Model Selection & Model Registry Packaging
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Root path configuration
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from deep_learning.config_loader import load_config
from deep_learning.progress_tracker import PipelineProgressTracker
from deep_learning.data_loader import TickDataScanner
from deep_learning.tick_bar_builder import (
    generate_tick_windows,
    generate_coarse_windows,
    generate_fine_neighborhood_windows,
    build_tick_bars,
    cache_tick_bars_parquet,
    load_cached_tick_bars
)
from deep_learning.feature_engineering import compute_features
from deep_learning.labeling import apply_triple_barrier_labeling
from deep_learning.cost_model import CostModelConfig, FuturesCostEngine
from deep_learning.validation import WalkForwardSplitter, PurgedKFoldSplitter
from deep_learning.modeling import BaselineModelSuite, TreeModelSuite
from deep_learning.transfer_learning_models import (
    PatchTST,
    PretrainedTimeSeriesFoundationModel,
    DeepLearningTrainer,
    TimeSeriesDataset
)
from deep_learning.backtest import VectorizedBacktester
from deep_learning.model_registry import ModelMetadata, save_model
from deep_learning.report import StrategyComparisonReporter
import torch
from torch.utils.data import DataLoader


def run_pipeline(
    config_path: str = "configs/config.yaml",
    data_dir_override: str = None,
    max_days: int = None
):
    # 0. Load Configuration
    overrides = {}
    if data_dir_override:
        overrides["data_dir"] = data_dir_override
    cfg = load_config(config_path, overrides=overrides)

    tracker = PipelineProgressTracker(
        total_stages=8,
        log_file_path=cfg["progress_display"].get("log_file_path", "logs/pipeline.log")
    )
    tracker.log("==========================================================================")
    tracker.log(" KOSPI 200 Futures Deep Learning & Transfer Learning Pipeline Starting")
    tracker.log("==========================================================================")

    # -------------------------------------------------------------
    # STAGE 1: Scan & Automated Integrity Verification of CSV Files
    # -------------------------------------------------------------
    scanner = TickDataScanner(data_dir=cfg["data_dir"], symbol=cfg.get("symbol", "KOSPI_F"))
    valid_files, sanitize_report = scanner.scan_all_files_with_integrity()
    if not valid_files:
        tracker.log(f"[!] No valid CSV files found in {cfg['data_dir']}")
        return

    tracker.log(f"[+] Automated File Integrity Scan Complete: {sanitize_report.summary()}")
    file_list = valid_files
    if max_days:
        file_list = file_list[:max_days]

    tracker.start_stage(1, "Scan & Clean Tick Data", total_steps=len(file_list))
    all_clean_ticks = []
    enc, col_map = scanner.infer_columns_and_encoding(file_list[0][1])

    for dt_str, fpath in file_list:
        df_clean, stats = scanner.clean_daily_ticks(fpath, dt_str, enc, col_map)
        if not df_clean.empty:
            all_clean_ticks.append(df_clean)
        tracker.update_stage_step(1, f"{dt_str} ({len(df_clean):,} ticks)")

    tracker.finish_stage(f"Total Valid Files Processed: {len(all_clean_ticks)}")
    if not all_clean_ticks:
        tracker.log("[!] No valid tick data after sanitization.")
        return

    df_full_ticks = pd.concat(all_clean_ticks, ignore_index=True)
    del all_clean_ticks
    import gc
    gc.collect()
    tracker.log(f"[+] Total Cleaned Ticks in Memory: {len(df_full_ticks):,}")

    # -------------------------------------------------------------
    # STAGE 2: Tick Bar Window Planning & Generation
    # -------------------------------------------------------------
    tw_cfg = cfg["tick_window"]
    start_w, end_w, step_w = tw_cfg["start"], tw_cfg["end"], tw_cfg["step"]

    # Coarse-to-fine decision
    use_coarse = cfg["coarse_to_fine"]["enabled"]
    if use_coarse:
        coarse_step = cfg["coarse_to_fine"]["coarse_step"]
        windows_to_test = generate_coarse_windows(start_w, end_w, coarse_step)
        tracker.log(f"[*] Coarse-to-Fine Enabled. 1st Stage Screen Windows ({len(windows_to_test)}): {windows_to_test}")
    else:
        windows_to_test = generate_tick_windows(start_w, end_w, step_w)
        tracker.log(f"[*] Full Exhaustive Search ({len(windows_to_test)} Windows): {windows_to_test[:5]} ... {windows_to_test[-3:]}")

    tracker.start_stage(2, "Generate Tick Bars & Parquet Caching", total_steps=len(windows_to_test))
    tick_bars_dict = {}

    for w in windows_to_test:
        # Check cache
        df_cached = load_cached_tick_bars(w, cfg["cache_dir"], cfg.get("symbol", "KOSPI_F"))
        if df_cached is not None and len(df_cached) > 0:
            tick_bars_dict[w] = df_cached
        else:
            df_bars = build_tick_bars(df_full_ticks, tick_size=w)
            if not df_bars.empty:
                cache_tick_bars_parquet(df_bars, w, cfg["cache_dir"], cfg.get("symbol", "KOSPI_F"))
                tick_bars_dict[w] = df_bars
        tracker.update_stage_step(1, f"Window {w} ({len(tick_bars_dict.get(w, [])):,} bars)")

    del df_full_ticks
    gc.collect()
    tracker.finish_stage(f"Generated Bars for {len(tick_bars_dict)} Windows")

    # -------------------------------------------------------------
    # STAGE 3: Feature Engineering & Triple Barrier Labeling
    # -------------------------------------------------------------
    tracker.start_stage(3, "Features & Triple Barrier Labeling", total_steps=len(tick_bars_dict))
    features_and_labels = {}

    for w, df_bars in tick_bars_dict.items():
        df_feat, fcols = compute_features(df_bars)
        if len(df_feat) > 50:
            labels, rets = apply_triple_barrier_labeling(
                df_feat,
                atr_tp_mult=2.0,
                atr_sl_mult=1.5,
                max_holding_bars=15,
                task_type=cfg["modeling"]["task_type"]
            )
            df_feat['target_label'] = labels.values
            features_and_labels[w] = (df_feat, fcols)
        tracker.update_stage_step(1, f"Window {w} ({len(df_feat):,} samples)")

    tracker.finish_stage(f"Constructed Features for {len(features_and_labels)} Windows")

    # -------------------------------------------------------------
    # STAGE 4: Cost Model & Validation Split Setup
    # -------------------------------------------------------------
    tracker.start_stage(4, "Configure Cost Engine & Walk-Forward Folds", total_steps=len(features_and_labels))
    cost_cfg = CostModelConfig(
        commission_rate=cfg["cost_model"]["commission_rate"],
        slippage_mode=cfg["cost_model"]["slippage_mode"],
        fixed_slippage_tick=cfg["cost_model"]["fixed_slippage_tick"],
        tick_value_krw=cfg["cost_model"]["tick_value_krw"],
        volume_impact_coef=cfg["cost_model"]["volume_impact_coef"]
    )
    cost_engine = FuturesCostEngine(cost_cfg)
    backtester = VectorizedBacktester(cost_engine)

    splitter = WalkForwardSplitter(
        train_days=int(cfg["validation"]["train_years"] * 250),
        test_days=int(cfg["validation"]["test_months"] * 21),
        embargo_pct=cfg["validation"]["embargo_pct"]
    )
    tracker.finish_stage("Cost Engine & Splitter Ready")

    # -------------------------------------------------------------
    # STAGE 5 & 6: Multi-Model Walk-Forward Training & Evaluation
    # -------------------------------------------------------------
    models_to_run = cfg["modeling"]["models_to_run"]
    total_combos = len(features_and_labels) * len(models_to_run)
    tracker.start_stage(5, "Model Training & Walk-Forward Validation", total_steps=total_combos)

    baseline_suite = BaselineModelSuite(task_type=cfg["modeling"]["task_type"])
    tree_suite = TreeModelSuite(task_type=cfg["modeling"]["task_type"], optuna_trials=cfg["modeling"]["optuna_trials"])
    dl_trainer = DeepLearningTrainer(
        task_type=cfg["modeling"]["task_type"],
        seq_len=cfg["modeling"]["deep_learning"]["seq_len"],
        epochs=cfg["modeling"]["deep_learning"]["epochs"],
        batch_size=cfg["modeling"]["deep_learning"]["batch_size"]
    )

    all_evaluation_results = []
    trained_model_registry_cache = {}

    for w, (df_feat, fcols) in features_and_labels.items():
        splits = splitter.split(df_feat)
        if not splits:
            continue

        for m_name in models_to_run:
            fold_metrics = []
            last_trained_model = None

            for split in splits:
                X_tr = df_feat[fcols].iloc[split.train_indices].values
                y_tr = df_feat['target_label'].iloc[split.train_indices].values
                X_te = df_feat[fcols].iloc[split.test_indices].values
                y_te = df_feat['target_label'].iloc[split.test_indices].values
                df_test_bars = df_feat.iloc[split.test_indices]

                # Train Model
                if m_name == "logistic_regression":
                    model, train_sec = baseline_suite.train_logistic(X_tr, y_tr)
                    preds, inf_lat = baseline_suite.predict(model, X_te, is_linear=True)
                    framework = "sklearn"
                elif m_name == "random_forest":
                    model, train_sec = baseline_suite.train_random_forest(X_tr, y_tr)
                    preds, inf_lat = baseline_suite.predict(model, X_te)
                    framework = "sklearn"
                elif m_name == "lightgbm":
                    model, best_p, train_sec = tree_suite.train_lightgbm(X_tr, y_tr)
                    preds, inf_lat = tree_suite.predict(model, X_te)
                    framework = "lightgbm"
                elif m_name == "xgboost":
                    model, best_p, train_sec = tree_suite.train_xgboost(X_tr, y_tr)
                    preds, inf_lat = tree_suite.predict(model, X_te)
                    framework = "xgboost"
                elif m_name == "patch_tst":
                    pt_model = PatchTST(num_features=len(fcols), seq_len=cfg["modeling"]["deep_learning"]["seq_len"])
                    train_ds = TimeSeriesDataset(X_tr, y_tr, seq_len=cfg["modeling"]["deep_learning"]["seq_len"])
                    t_loader = DataLoader(train_ds, batch_size=cfg["modeling"]["deep_learning"]["batch_size"], shuffle=True)
                    model, train_sec = dl_trainer.train_model(pt_model, t_loader)
                    preds, inf_lat = dl_trainer.predict(model, X_te)
                    framework = "pytorch"
                elif m_name == "foundation_ts":
                    f_model = PretrainedTimeSeriesFoundationModel(num_features=len(fcols), seq_len=cfg["modeling"]["deep_learning"]["seq_len"])
                    f_model.unfreeze_backbone()  # Transfer learning fine-tuning
                    train_ds = TimeSeriesDataset(X_tr, y_tr, seq_len=cfg["modeling"]["deep_learning"]["seq_len"])
                    t_loader = DataLoader(train_ds, batch_size=cfg["modeling"]["deep_learning"]["batch_size"], shuffle=True)
                    model, train_sec = dl_trainer.train_model(f_model, t_loader)
                    preds, inf_lat = dl_trainer.predict(model, X_te)
                    framework = "pytorch"
                else:
                    continue

                last_trained_model = (model, framework)

                # STAGE 7: Vectorized Backtest with Costs
                metrics = backtester.run_backtest(df_test_bars, preds)
                fold_metrics.append((metrics, train_sec, inf_lat))

            # Aggregate across folds
            if fold_metrics:
                avg_sharpe = float(np.mean([m[0].sharpe_ratio for m in fold_metrics]))
                avg_mdd = float(np.mean([m[0].max_drawdown_pct for m in fold_metrics]))
                tot_trades = int(np.sum([m[0].total_trades for m in fold_metrics]))
                avg_win = float(np.mean([m[0].win_rate_pct for m in fold_metrics]))
                avg_cum_ret = float(np.mean([m[0].cumulative_return_pct for m in fold_metrics]))
                avg_train_sec = float(np.mean([m[1] for m in fold_metrics]))
                avg_inf_lat = float(np.mean([m[2] for m in fold_metrics]))

                combo_result = {
                    "tick_window": w,
                    "model_name": m_name,
                    "sharpe_ratio": avg_sharpe,
                    "max_drawdown_pct": avg_mdd,
                    "total_trades": tot_trades,
                    "win_rate_pct": avg_win,
                    "cumulative_return_pct": avg_cum_ret,
                    "training_time_sec": round(avg_train_sec, 2),
                    "inference_latency_ms": round(avg_inf_lat, 3),
                    "equity_curve": fold_metrics[-1][0].equity_curve
                }
                all_evaluation_results.append(combo_result)
                trained_model_registry_cache[f"{w}_{m_name}"] = (last_trained_model, fcols, combo_result)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            tracker.update_stage_step(1, f"Tick {w} × {m_name}")

    tracker.finish_stage(f"Trained & Evaluated {len(all_evaluation_results)} Combinations")

    # -------------------------------------------------------------
    # STAGE 8: Strategy Selection, Model Packaging & Comparison Report
    # -------------------------------------------------------------
    tracker.start_stage(8, "Best Model Selection, Packaging & Reporting", total_steps=100)

    reporter = StrategyComparisonReporter(output_dir="output/dl_reports")
    summary_df = reporter.generate_comparison_summary(all_evaluation_results)
    reporter.create_interactive_comparison_chart(all_evaluation_results)

    tracker.update_stage_step(50, "Generated Comparison Reports")

    # Select Best Candidate based on Criteria
    crit = cfg["selection_criteria"]
    candidates = summary_df[
        (summary_df["total_trades"] >= crit["min_trades"]) &
        (summary_df["max_drawdown_pct"] <= crit["max_mdd_pct"]) &
        (~summary_df["overfitting_flag"])
    ]

    if not candidates.empty:
        best_row = candidates.iloc[0]
    elif not summary_df.empty:
        best_row = summary_df.iloc[0]
    else:
        best_row = None

    if best_row is not None:
        best_key = f"{int(best_row['tick_window'])}_{best_row['model_name']}"
        (best_model_obj, framework), feat_names, best_metrics = trained_model_registry_cache[best_key]

        meta = ModelMetadata(
            version="1.0.0",
            model_name=best_row['model_name'],
            tick_window=int(best_row['tick_window']),
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            feature_names=feat_names,
            performance_metrics=best_metrics,
            hyperparameters={"task_type": cfg["modeling"]["task_type"]},
            framework=framework
        )
        saved_weights_path, saved_meta_path = save_model(
            model=best_model_obj,
            metadata=meta,
            save_dir=cfg["model_save_dir"],
            registry_file=cfg["registry_path"]
        )
        tracker.log("==========================================================================")
        tracker.log(f" ★ BEST STRATEGY SELECTED: Tick {meta.tick_window} | {meta.model_name}")
        tracker.log(f"   Sharpe: {best_row['sharpe_ratio']:.2f} | MDD: {best_row['max_drawdown_pct']:.1f}% | Trades: {best_row['total_trades']}")
        tracker.log(f"   Saved Model: {saved_weights_path}")
        tracker.log(f"   Saved Metadata: {saved_meta_path}")
        tracker.log("==========================================================================")

    tracker.update_stage_step(50, "Packaged Best Model")
    tracker.finish_stage("Model Packaging & Registry Update Complete")
    tracker.finish_pipeline()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KOSPI 200 Futures Deep Learning Pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to CSV folder")
    parser.add_argument("--max-days", type=int, default=None, help="Limit number of days for rapid debugging")
    args = parser.parse_args()

    run_pipeline(config_path=args.config, data_dir_override=args.data_dir, max_days=args.max_days)
