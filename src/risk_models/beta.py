from typing import Dict
import numpy as np
import pandas as pd

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("BetaCalculator")

class BetaCalculator:
    """
    Calculates systematic market risk (Beta) of each stock relative to VN-Index.
    """
    def __init__(self, lookback: int = 60):
        self.lookback = lookback if lookback else Config.COV_LOOKBACK
        self.benchmark = Config.BENCHMARK_TICKER

    def compute_betas(self, cleaned_data: Dict[str, pd.DataFrame], target_date: str) -> pd.Series:
        if self.benchmark not in cleaned_data:
            raise ValueError("Benchmark data not found in cleaned_data.")

        bench_df = cleaned_data[self.benchmark]
        sub_bench = bench_df[bench_df['date'] <= target_date].tail(self.lookback + 1)
        bench_ret = sub_bench['close'].pct_change().dropna().values[-self.lookback:]
        var_bench = np.var(bench_ret) + 1e-8

        symbols = [s for s in cleaned_data.keys() if s != self.benchmark]
        betas = {}

        for sym in symbols:
            df = cleaned_data[sym]
            sub = df[df['date'] <= target_date].tail(self.lookback + 1)
            if len(sub) > 1:
                ret = sub['close'].pct_change().dropna().values[-self.lookback:]
                if len(ret) == len(bench_ret):
                    cov = np.cov(ret, bench_ret)[0, 1]
                    betas[sym] = cov / var_bench
                else:
                    betas[sym] = 1.0
            else:
                betas[sym] = 1.0

        return pd.Series(betas)
