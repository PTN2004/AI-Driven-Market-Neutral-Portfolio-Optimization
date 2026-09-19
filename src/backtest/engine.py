from typing import Dict, Any
import numpy as np
import pandas as pd

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("BacktestEngine")

class BacktestEngine:
    def __init__(
        self,
        predictor: Any,
        risk_model: Any,
        beta_calc: Any,
        optimizer: Any,
        rebalance_freq: str = "weekly",
        fee_rate: float = 0.0020
    ):
        self.predictor = predictor
        self.risk_model = risk_model
        self.beta_calc = beta_calc
        self.optimizer = optimizer
        self.rebalance_freq = rebalance_freq
        self.fee_rate = fee_rate
        self.benchmark = Config.BENCHMARK_TICKER

    def _should_rebalance(self, date: pd.Timestamp, step_idx: int) -> bool:
        if step_idx == 0:
            return True
        if self.rebalance_freq == "weekly":
            return date.dayofweek == 0 # Monday
        elif self.rebalance_freq == "monthly":
            return date.day <= 5 and step_idx % 20 == 0
        return step_idx % 5 == 0

    def run(
        self,
        cleaned_data: Dict[str, pd.DataFrame],
        normalized_data: Dict[str, pd.DataFrame],
        start_date: str = Config.TEST_START_DATE,
        end_date: str = Config.END_DATE
    ) -> Dict[str, pd.DataFrame]:
        logger.info(f"Starting Out-of-Sample backtest ({start_date} -> {end_date})...")
        
        bench_df = cleaned_data[self.benchmark]
        bench_sub = bench_df[(bench_df['date'] >= start_date) & (bench_df['date'] <= end_date)].copy()
        dates = pd.to_datetime(bench_sub['date']).values

        symbols = [s for s in cleaned_data.keys() if s != self.benchmark]
        
        price_matrix = {}
        for sym in symbols:
            df = cleaned_data[sym]
            sub = df[(df['date'] >= start_date) & (df['date'] <= end_date)].set_index('date')['close']
            price_matrix[sym] = sub
        price_df = pd.DataFrame(price_matrix).ffill().bfill()

        current_weights = pd.Series(0.0, index=symbols)
        results = []
        weights_history = []

        for idx, date_dt in enumerate(dates):
            date_str = pd.to_datetime(date_dt).strftime("%Y-%m-%d")
            rebalanced = False
            tx_cost = 0.0

            if self._should_rebalance(pd.to_datetime(date_str), idx):
                try:
                    # 1. Predict Alpha
                    mu_dict = self.predictor.predict_for_date(normalized_data, date_str)
                    mu_series = pd.Series(mu_dict)
                    for s in symbols:
                        if s not in mu_series:
                            mu_series[s] = 0.0
                    mu_series = mu_series.loc[symbols].fillna(0.0)

                    # 2. Risk Modeling
                    cov_mat = self.risk_model.compute_covariance(cleaned_data, date_str)
                    betas = self.beta_calc.compute_betas(cleaned_data, date_str)

                    # 3. Optimize Portfolio
                    new_weights = self.optimizer.optimize(mu_series, cov_mat, betas)
                    new_weights = new_weights.reindex(symbols).fillna(0.0)

                    # Calculate transaction cost from weight turnover
                    turnover = (new_weights - current_weights).abs().sum()
                    tx_cost = turnover * self.fee_rate

                    current_weights = new_weights
                    rebalanced = True
                except Exception as e:
                    logger.debug(f"Rebalance skipped on {date_str}: {e}")

            if idx > 0 and date_str in price_df.index and price_df.index[idx-1] in price_df.index:
                p_today = price_df.iloc[idx]
                p_prev = price_df.iloc[idx - 1]
                daily_ret_vec = (p_today - p_prev) / (p_prev + 1e-8)
                print(f"daily return - {daily_ret_vec} - {current_weights} - {tx_cost}")
                port_ret = float((current_weights * daily_ret_vec).sum() - tx_cost)
            else:
                port_ret = 0.0

            # Benchmark return
            bench_ret = 0.0
            if idx > 0:
                b_today = bench_sub.iloc[idx]['close']
                b_prev = bench_sub.iloc[idx - 1]['close']
                bench_ret = (b_today - b_prev) / (b_prev + 1e-8)

            # Portfolio stats
            gross_exp = current_weights.abs().sum()
            net_exp = current_weights.sum()
            try:
                betas_today = self.beta_calc.compute_betas(cleaned_data, date_str).reindex(symbols).fillna(1.0)
                port_beta = (current_weights * betas_today).sum()
            except Exception:
                port_beta = 0.0

            results.append({
                "date": date_str,
                "portfolio_return": port_ret,
                "benchmark_return": bench_ret,
                "turnover_cost": tx_cost,
                "gross_exposure": gross_exp,
                "net_exposure": net_exp,
                "portfolio_beta": port_beta,
                "rebalanced": rebalanced
            })

            w_row = current_weights.to_dict()
            w_row["date"] = date_str
            weights_history.append(w_row)

        results_df = pd.DataFrame(results)
        weights_df = pd.DataFrame(weights_history)
        logger.info("Out-of-Sample backtest completed successfully.")
        return {"results_df": results_df, "weights_df": weights_df}
