"""
Automated Quantitative Performance Comparison Reporter
Generates comprehensive comparisons across (Tick Window × Model Type):
- Heatmaps of Sharpe Ratio & Cumulative Return
- Equity curves overlay
- Drawdown charts and trade distributions
- CSV and JSON summary tables
"""
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class StrategyComparisonReporter:
    def __init__(self, output_dir: str | Path = "output/dl_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_comparison_summary(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Takes raw list of test result dicts and constructs ranking DataFrame.
        """
        df = pd.DataFrame(results)
        if df.empty:
            return df

        # Sort by Sharpe Ratio descending
        df = df.sort_values(by="sharpe_ratio", ascending=False).reset_index(drop=True)
        csv_path = self.output_dir / "model_comparison_matrix.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        # Flag suspected overfitting: extreme Sharpe (> 5.0) or low trade count (< 10)
        df['overfitting_flag'] = (df['sharpe_ratio'] > 5.0) | (df['total_trades'] < 10)

        return df

    def create_interactive_comparison_chart(self, top_results: List[Dict[str, Any]], html_name: str = "dl_comparison.html") -> Path:
        """
        Creates Plotly interactive dashboard comparing top strategies.
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=False,
            vertical_spacing=0.12,
            subplot_titles=("Cumulative Equity Curves (Top Combinations)", "Sharpe Ratio vs Max Drawdown")
        )

        for res in top_results[:5]:
            label = f"Tick {res.get('tick_window')} | {res.get('model_name')} (SR: {res.get('sharpe_ratio', 0):.2f})"
            equity = res.get('equity_curve', [])
            if equity:
                fig.add_trace(
                    go.Scatter(y=equity, mode='lines', name=label),
                    row=1, col=1
                )

        # Scatter for all results
        windows = [r.get('tick_window') for r in top_results]
        sharpes = [r.get('sharpe_ratio', 0) for r in top_results]
        mdds = [r.get('max_drawdown_pct', 0) for r in top_results]
        models = [r.get('model_name') for r in top_results]

        fig.add_trace(
            go.Scatter(
                x=mdds,
                y=sharpes,
                mode='markers+text',
                text=[f"{m}-{w}" for m, w in zip(models, windows)],
                textposition="top center",
                marker=dict(size=12, color=sharpes, colorscale='Viridis', showscale=True),
                name="Sharpe vs MDD"
            ),
            row=2, col=1
        )

        fig.update_layout(
            template="plotly_dark",
            height=800,
            title_text="KOSPI 200 Futures Deep Learning & Transfer Learning Grid Comparison",
            showlegend=True
        )

        chart_path = self.output_dir / html_name
        fig.write_html(str(chart_path))
        return chart_path
