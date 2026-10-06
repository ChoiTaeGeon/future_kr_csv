"""
Visualization & PDF Reporting Engine
- Plotly-based interactive multi-pane candlestick + indicator + trade markers + equity chart.
- ReportLab-based automated PDF executive backtest report generator.
"""
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use('Agg')  # Headless backend for fast plotting & small binary size
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io

from reportlab.lib.pagesizes import A4, letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

from engine.backtester import BacktestResult


class Visualizer:
    """Plotly interactive visualization for OHLCV, Indicators, Signals, and Equity Curve."""

    @staticmethod
    def plot_strategy_backtest(df_bars: pd.DataFrame, result: BacktestResult, output_html_path: str = "output/charts/strategy_chart.html"):
        """
        Creates an interactive 3-row Plotly chart:
        1. Candlestick + Trades
        2. Volume
        3. Equity Curve & Drawdown
        """
        Path(output_html_path).parent.mkdir(parents=True, exist_ok=True)

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.55, 0.15, 0.30],
            subplot_titles=[
                f"Price Action & Trade Execution ({result.strategy_name} - {result.tick_size} Ticks)",
                "Volume",
                f"Cumulative Equity ({result.total_return_pct:+.2f}%, Sharpe: {result.sharpe_ratio:.2f}, MDD: {result.max_drawdown_pct:.2f}%)"
            ]
        )

        # 1. Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df_bars['timestamp'],
                open=df_bars['open'],
                high=df_bars['high'],
                low=df_bars['low'],
                close=df_bars['close'],
                name="OHLC",
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=1, col=1
        )

        # Overlay VWAP or EMA if available
        if 'vwap' in df_bars.columns:
            fig.add_trace(go.Scatter(x=df_bars['timestamp'], y=df_bars['vwap'], line=dict(color='orange', width=1.2), name="VWAP"), row=1, col=1)
        if 'ema_fast' in df_bars.columns and 'ema_slow' in df_bars.columns:
            fig.add_trace(go.Scatter(x=df_bars['timestamp'], y=df_bars['ema_fast'], line=dict(color='#2196F3', width=1), name="EMA Fast"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_bars['timestamp'], y=df_bars['ema_slow'], line=dict(color='#9C27B0', width=1), name="EMA Slow"), row=1, col=1)

        # Trade markers
        if not result.trade_log.empty:
            longs = result.trade_log[result.trade_log['side'] == 'LONG']
            shorts = result.trade_log[result.trade_log['side'] == 'SHORT']
            
            if not longs.empty:
                fig.add_trace(go.Scatter(
                    x=longs['entry_time'], y=longs['entry_price'],
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=11, color='#00E676', line=dict(width=1, color='black')),
                    name='Buy Entry'
                ), row=1, col=1)
                
            if not shorts.empty:
                fig.add_trace(go.Scatter(
                    x=shorts['entry_time'], y=shorts['entry_price'],
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=11, color='#FF1744', line=dict(width=1, color='black')),
                    name='Sell Entry'
                ), row=1, col=1)

        # 2. Volume
        fig.add_trace(
            go.Bar(
                x=df_bars['timestamp'],
                y=df_bars['volume'],
                name="Volume",
                marker_color='#78909c'
            ),
            row=2, col=1
        )

        # 3. Equity Curve & Drawdown
        if not result.equity_curve.empty:
            eq = result.equity_curve
            fig.add_trace(
                go.Scatter(
                    x=eq['timestamp'],
                    y=eq['equity_pct'],
                    line=dict(color='#00c853', width=2),
                    name="Equity (%)"
                ),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=eq['timestamp'],
                    y=eq['drawdown_pct'],
                    fill='tozeroy',
                    line=dict(color='#d50000', width=1),
                    name="Drawdown (%)"
                ),
                row=3, col=1
            )

        fig.update_layout(
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            height=900,
            margin=dict(l=40, r=40, t=60, b=40),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        fig.write_html(output_html_path)
        return output_html_path


class ReportGenerator:
    """Generates PDF Backtest Performance Reports using ReportLab."""

    @staticmethod
    def generate_equity_plot_image(result: BacktestResult) -> str:
        """Saves a clean matplotlib equity curve image for inclusion in PDF."""
        img_path = Path(f"output/charts/eq_{result.strategy_name}_{result.tick_size}.png")
        img_path.parent.mkdir(parents=True, exist_ok=True)
        
        if result.equity_curve.empty:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.text(0.5, 0.5, "No Trades Executed", ha='center', va='center')
            plt.savefig(img_path, bbox_inches='tight', dpi=150)
            plt.close()
            return str(img_path)

        eq = result.equity_curve
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 3.8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        # Equity Curve
        ax1.plot(pd.to_datetime(eq['timestamp']), eq['equity_pct'], color='#1b5e20', lw=1.8, label='Cumulative Return (%)')
        ax1.set_title(f"Strategy: {result.strategy_name} ({result.tick_size} Ticks) - Cumulative Equity Curve", fontsize=11, fontweight='bold')
        ax1.set_ylabel("Return (%)", fontsize=9)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(loc='upper left', fontsize=8)

        # Drawdown
        ax2.fill_between(pd.to_datetime(eq['timestamp']), eq['drawdown_pct'], 0, color='#b71c1c', alpha=0.5, label='Drawdown (%)')
        ax2.set_ylabel("DD (%)", fontsize=9)
        ax2.set_xlabel("Time", fontsize=9)
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(loc='lower left', fontsize=8)

        plt.tight_layout()
        plt.savefig(img_path, dpi=180, bbox_inches='tight')
        plt.close()
        return str(img_path)

    @staticmethod
    def create_pdf_report(
        summary_df: pd.DataFrame,
        best_results: List[BacktestResult],
        output_pdf_path: str = "output/reports/Quant_Backtest_Report.pdf",
        report_title: str = "High-Frequency Tick Data Strategy Backtest Report",
        market_desc: Optional[str] = None,
        currency: str = "KRW"
    ) -> str:
        """
        Builds a comprehensive PDF report.
        """
        Path(output_pdf_path).parent.mkdir(parents=True, exist_ok=True)
        doc = SimpleDocTemplate(
            output_pdf_path,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#0D47A1"),
            alignment=TA_CENTER,
            spaceAfter=12
        )
        
        subtitle_style = ParagraphStyle(
            'SubtitleStyle',
            parent=styles['Normal'],
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#455A64"),
            alignment=TA_CENTER,
            spaceAfter=16
        )
        
        section_style = ParagraphStyle(
            'SectionStyle',
            parent=styles['Heading2'],
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#1565C0"),
            spaceBefore=12,
            spaceAfter=8
        )
        
        cell_style = ParagraphStyle(
            'CellStyle',
            parent=styles['Normal'],
            fontSize=7.2,
            leading=9,
            alignment=TA_CENTER
        )
        
        cell_header_style = ParagraphStyle(
            'CellHeaderStyle',
            parent=styles['Normal'],
            fontSize=7.5,
            leading=9.5,
            fontName="Helvetica-Bold",
            textColor=colors.white,
            alignment=TA_CENTER
        )

        story = []

        # 1. Header & Title
        story.append(Paragraph(report_title, title_style))
        now_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        sub_text = f"Generated at: {now_str} | {market_desc or 'Market: KOSPI/Futures Day Trading | Engine: DuckDB + Vectorized'}"
        story.append(Paragraph(sub_text, subtitle_style))
        story.append(Spacer(1, 10))

        # 2. Executive Summary Metrics Box
        if best_results:
            top = best_results[0]
            is_usd = (str(currency).upper() == 'USD')
            top_pnl_val = float(top.equity_curve['net_pnl_cash'].sum()) if not top.equity_curve.empty else 0.0
            if is_usd:
                pnl_str = f"+${top_pnl_val:,.0f}" if top_pnl_val >= 0 else f"-${abs(top_pnl_val):,.0f}"
            else:
                pnl_str = f"+{int(top_pnl_val/10000):,}만원" if top_pnl_val >= 0 else f"-{int(abs(top_pnl_val)/10000):,}만원"

            exec_summary_data = [
                [
                    Paragraph(f"<b>Top Strategy:</b> {top.strategy_name} ({top.tick_size})", cell_style),
                    Paragraph(f"<b>Cumulative Return:</b> {top.total_return_pct:+.2f}%", cell_style),
                    Paragraph(f"<b>Net Profit:</b> {pnl_str}", cell_style),
                ],
                [
                    Paragraph(f"<b>Max Drawdown:</b> {top.max_drawdown_pct:.2f}%", cell_style),
                    Paragraph(f"<b>Win Rate:</b> {top.win_rate:.1f}% ({top.win_trades}W/{top.loss_trades}L)", cell_style),
                    Paragraph(f"<b>Profit Factor:</b> {top.profit_factor:.2f}", cell_style),
                ]
            ]
            exec_table = Table(exec_summary_data, colWidths=[180, 180, 180])
            exec_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#E3F2FD")),
                ('BOX', (0, 0), (-1, -1), 1.5, colors.HexColor("#1976D2")),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(exec_table)
            story.append(Spacer(1, 15))

        # 3. Best Strategies Equity Chart
        story.append(Paragraph("1. Best Strategy Equity Curve & Drawdown Analysis", section_style))
        for res in best_results[:2]:
            chart_img = ReportGenerator.generate_equity_plot_image(res)
            story.append(Image(chart_img, width=7.2*inch, height=3.0*inch))
            story.append(Spacer(1, 10))

        # 4. Multi-Tick & Strategy Performance Comparison Matrix
        story.append(Paragraph("2. Multi-Tick Resolution & Strategy Performance Ranking", section_style))
        
        # Prepare table data
        has_net_usd = 'Net Profit ($)' in summary_df.columns
        if has_net_usd:
            cols = ['Tick Size', 'Strategy', 'Cumulative Return (%)', 'Net Profit ($)', 'Sharpe Ratio', 'MDD (%)', 'Win Rate (%)', 'Profit Factor', 'Trades']
            col_widths = [45, 100, 75, 70, 50, 50, 50, 50, 50]
        else:
            cols = ['Tick Size', 'Strategy', 'Cumulative Return (%)', 'Sharpe Ratio', 'MDD (%)', 'Win Rate (%)', 'Profit Factor', 'Trades']
            col_widths = [50, 110, 80, 55, 55, 60, 60, 50]

        valid_cols = [c for c in cols if c in summary_df.columns]
        df_display = summary_df[valid_cols].head(25).copy()
        
        table_data = [[Paragraph(c, cell_header_style) for c in valid_cols]]
        for _, row in df_display.iterrows():
            row_cells = []
            for c in valid_cols:
                val = row[c]
                if isinstance(val, float):
                    text = f"{val:+.2f}" if 'Return' in c else f"{val:.2f}"
                else:
                    text = str(val)
                row_cells.append(Paragraph(text, cell_style))
            table_data.append(row_cells)

        perf_table = Table(table_data, colWidths=col_widths[:len(valid_cols)])
        perf_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1565C0")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#B0BEC5")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#FAFAFA"), colors.HexColor("#ECEFF1")]),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        
        story.append(perf_table)
        story.append(Spacer(1, 15))

        # 5. Build Document
        doc.build(story)
        return output_pdf_path
