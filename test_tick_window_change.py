"""
Test Script: Verifies that changing config.yaml tick_window (start/end/step)
is dynamically reflected throughout the entire pipeline without code modification.
"""
import sys
from pathlib import Path
import tempfile
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from deep_learning.config_loader import load_config
from deep_learning.tick_bar_builder import generate_tick_windows, generate_coarse_windows


def test_dynamic_tick_window_behavior():
    print("[*] Running Regression Test: Dynamic tick_window parameter reflection...")

    # Scenario A: Standard 100~1000 step 10
    config_a = {
        "data_dir": "csv",
        "tick_window": {"start": 100, "end": 1000, "step": 10},
        "coarse_to_fine": {"enabled": False, "coarse_step": 100, "top_n_regions": 2, "fine_neighborhood": 50},
        "cost_model": {"commission_rate": 0.00003, "slippage_mode": "fixed", "fixed_slippage_tick": 0.05, "tick_value_krw": 250000, "volume_impact_coef": 0.00001},
        "model_save_dir": "models/saved_models",
        "progress_display": {"enabled": True, "update_interval_sec": 1}
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f_a:
        yaml.dump(config_a, f_a)
        path_a = f_a.name

    loaded_a = load_config(path_a)
    tw_a = loaded_a["tick_window"]
    windows_a = generate_tick_windows(tw_a["start"], tw_a["end"], tw_a["step"])
    assert len(windows_a) == 91, f"Expected 91 windows, got {len(windows_a)}"
    assert windows_a[0] == 100 and windows_a[-1] == 1000
    print(f"  [OK] Scenario A verified: start=100, end=1000, step=10 -> {len(windows_a)} windows correctly generated.")

    # Scenario B: User changes to start=200, end=5000, step=100
    config_b = config_a.copy()
    config_b["tick_window"] = {"start": 200, "end": 5000, "step": 100}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f_b:
        yaml.dump(config_b, f_b)
        path_b = f_b.name

    loaded_b = load_config(path_b)
    tw_b = loaded_b["tick_window"]
    windows_b = generate_tick_windows(tw_b["start"], tw_b["end"], tw_b["step"])
    assert len(windows_b) == 49, f"Expected 49 windows, got {len(windows_b)}"
    assert windows_b[0] == 200 and windows_b[-1] == 5000
    print(f"  [OK] Scenario B verified: start=200, end=5000, step=100 -> {len(windows_b)} windows correctly generated.")

    # Scenario C: Coarse-to-fine mode check
    coarse_windows = generate_coarse_windows(tw_a["start"], tw_a["end"], coarse_step=100)
    assert coarse_windows == [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    print(f"  [OK] Scenario C verified: Coarse screening window list correctly generated: {coarse_windows}")

    print("[SUCCESS] All Tick Window Dynamic Reflection Tests Passed Successfully!")


if __name__ == "__main__":
    test_dynamic_tick_window_behavior()
