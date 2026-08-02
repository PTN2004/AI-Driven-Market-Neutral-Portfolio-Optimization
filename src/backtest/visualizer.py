from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("BacktestVisualizer")


class BacktestVisualizer:
    """
    Generates quantitative performance charts for the Market Neutral strategy.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir if output_dir else Config.OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'figure.figsize': (12, 6)
        })

    def plot_equity_curves(self, results_df: pd.DataFrame):
        plt.figure(figsize=(12, 6))
        dates = pd.to_datetime(results_df['date'])

        cum_ai = (1 + results_df['portfolio_return']).cumprod() * 100
        cum_bm = (1 + results_df['benchmark_return']).cumprod() * 100

        plt.plot(dates, cum_ai, label='AI Market Neutral',
                 color='#2b5c8f', linewidth=2.5)
        plt.plot(dates, cum_bm, label='VN-Index (Buy & Hold)',
                 color='#d95f02', linewidth=1.8, linestyle='--')

        plt.title('Out-of-Sample Equity Curves (Normalized to 100)')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(self.output_dir / "01_equity_curves.png", dpi=300)
        plt.close()
        plt.show()

    def plot_drawdowns(self, results_df: pd.DataFrame):
        plt.figure(figsize=(12, 5))
        dates = pd.to_datetime(results_df['date'])

        cum_ai = (1 + results_df['portfolio_return']).cumprod()
        dd_ai = (cum_ai - cum_ai.cummax()) / cum_ai.cummax() * 100

        cum_bm = (1 + results_df['benchmark_return']).cumprod()
        dd_bm = (cum_bm - cum_bm.cummax()) / cum_bm.cummax() * 100

        plt.plot(dates, dd_ai, label='AI Market Neutral MDD',
                 color='#2b5c8f', linewidth=2)
        plt.plot(dates, dd_bm, label='VN-Index MDD',
                 color='#d95f02', linewidth=1.5, alpha=0.7)

        plt.title('Historical Underwater Drawdowns (%)')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig(self.output_dir / "02_drawdowns.png", dpi=300)
        plt.close()
        plt.show()

    def plot_rolling_beta(self, results_df: pd.DataFrame):
        plt.figure(figsize=(12, 5))
        dates = pd.to_datetime(results_df['date'])

        if 'portfolio_beta' in results_df.columns:
            beta_series = results_df['portfolio_beta'].rolling(
                window=10, min_periods=1).mean()
        else:
            beta_series = pd.Series(np.zeros(len(dates)))

        plt.plot(dates, beta_series, color='#2ca02c', linewidth=2,
                 label='10-Day Rolling Portfolio Beta')
        plt.axhline(0.0, color='red', linestyle='--', alpha=0.7,
                    label='Market Neutral Target (Beta = 0)')
        plt.axhspan(-0.05, 0.05, color='green', alpha=0.1,
                    label='Neutrality Tolerance Band (±0.05)')

        plt.title('Systematic Risk Exposure (Rolling Portfolio Beta)')
        plt.xlabel('Date')
        plt.ylabel('Beta')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(self.output_dir / "03_rolling_beta.png", dpi=300)
        plt.close()
        plt.show()

    def plot_exposure_history(self, results_df: pd.DataFrame):
        plt.figure(figsize=(12, 5))
        dates = pd.to_datetime(results_df['date'])

        gross = results_df.get('gross_exposure', pd.Series(
            1.0, index=results_df.index)) * 100
        net = results_df.get('net_exposure', pd.Series(
            0.0, index=results_df.index)) * 100

        plt.plot(dates, gross, label='Gross Exposure (%)',
                 color='#9467bd', linewidth=2)
        plt.plot(dates, net, label='Net Exposure (%)',
                 color='#1f77b4', linewidth=2, linestyle='-.')
        plt.axhline(0.0, color='black', linestyle=':', alpha=0.5)

        plt.title('Gross vs. Net Market Exposure over Time')
        plt.xlabel('Date')
        plt.ylabel('Exposure (%)')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(self.output_dir / "04_exposure_history.png", dpi=300)
        plt.close()
        plt.show()

    def generate_all_plots(self, results_df: pd.DataFrame, weights_df: Optional[pd.DataFrame] = None):
        logger.info("Generating all quantitative charts in data/output/...")
        self.plot_equity_curves(results_df)
        self.plot_drawdowns(results_df)
        self.plot_rolling_beta(results_df)
        self.plot_exposure_history(results_df)
        logger.info("Chart generation completed successfully.")
