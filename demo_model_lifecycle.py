"""
Demonstration: Complete Model Lifecycle
1. Train (or construct) Model on Sample Data
2. Save Model weights (.pt/.joblib) + Metadata (.meta.json) + Register to Registry
3. Terminate / Restart simulation
4. Load Model from saved file with Feature Schema Verification
5. Run Real-Time Stream Inference and Risk Management Execution
"""
import sys
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from deep_learning.model_registry import ModelMetadata, save_model, load_model
from deep_learning.modeling import TreeModelSuite
from deep_learning.live_inference import LiveInferencePipeline


def run_lifecycle_demo():
    print("[*] =================================================================")
    print("    Starting Model Lifecycle Demonstration (Train -> Save -> Restart -> Load -> Live Inference)")
    print("    =================================================================")

    demo_dir = PROJECT_ROOT / "models" / "demo_lifecycle"
    if demo_dir.exists():
        shutil.rmtree(demo_dir)
    demo_dir.mkdir(parents=True, exist_ok=True)
    reg_file = demo_dir / "registry.json"

    # Step 1: Mock training
    print("\n[1/5] Training sample model (LightGBM on 19 tick features)...")
    np.random.seed(42)
    n_samples = 500
    n_features = 19
    feature_names = [f"feat_{i}" for i in range(n_features)]
    X_mock = np.random.randn(n_samples, n_features)
    y_mock = np.random.choice([-1, 0, 1], size=n_samples)

    tree_suite = TreeModelSuite(task_type="multiclass", optuna_trials=2)
    model, best_params, train_time = tree_suite.train_lightgbm(X_mock, y_mock)
    print(f"      Trained LightGBM in {train_time:.2f}s with best params: {list(best_params.keys())[:3]}")

    # Step 2: Packaging & Saving
    print("\n[2/5] Packaging model with full metadata and saving to disk...")
    meta = ModelMetadata(
        version="1.0.0",
        model_name="lightgbm",
        tick_window=300,
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        feature_names=feature_names,
        performance_metrics={"sharpe_ratio": 2.15, "max_drawdown_pct": 8.4, "win_rate_pct": 62.5},
        hyperparameters=best_params,
        framework="lightgbm"
    )
    weights_path, meta_path = save_model(
        model=model,
        metadata=meta,
        save_dir=demo_dir,
        registry_file=reg_file
    )
    print(f"      Saved Model Weights: {weights_path.name}")
    print(f"      Saved Metadata File: {meta_path.name}")
    print(f"      Registry Index Updated: {reg_file.exists()}")

    # Step 3: Simulate Process Restart
    print("\n[3/5] Simulating Python process termination & restart (clearing memory)...")
    del model
    del meta

    # Step 4: Loading & Schema Validation
    print("\n[4/5] Loading model from disk and verifying feature schema...")
    loaded_model, loaded_meta = load_model(weights_path, expected_features=feature_names)
    print(f"      Successfully Loaded Model: {loaded_meta.model_name} (Tick Window: {loaded_meta.tick_window})")
    print(f"      Verified Feature Schema: {len(loaded_meta.feature_names)} features matched perfectly.")

    # Step 5: Live Streaming Inference Simulation
    print("\n[5/5] Feeding live tick stream to LiveInferencePipeline...")
    live_pipeline = LiveInferencePipeline(weights_path)

    # Stream 350 ticks into pipeline (tick_window = 300)
    print("      Streaming ticks into pipeline...")
    base_price = 345.0
    signal_count = 0
    now = datetime.now()

    for i in range(1, 350):
        price_step = np.random.choice([-0.05, 0.0, 0.05])
        base_price += price_step
        tick = {
            "datetime": str(now + pd.Timedelta(seconds=i)),
            "price": round(base_price, 2),
            "volume": int(np.random.randint(1, 20)),
            "side": int(np.random.choice([1, -1]))
        }
        sig = live_pipeline.on_tick(tick)
        if sig is not None:
            signal_count += 1
            print(f"      [!] Bar Triggered Signal: Action={sig.action}, Pos={sig.target_pos}, Price={sig.current_price}, TP={sig.take_profit_price}, SL={sig.stop_loss_price}")

    print("\n[SUCCESS] Model Lifecycle Demonstration Completed Flawlessly!")


if __name__ == "__main__":
    run_lifecycle_demo()
