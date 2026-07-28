from typing import Dict, List
import numpy as np
import pandas as pd

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("DataCleaner")

class DataCleaner:
    def __init__(self):
        self.benchmark = Config.BENCHMARK_TICKER

    def clean_and_align(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        if self.benchmark not in raw_data:
            raise ValueError(f"Benchmark ticker {self.benchmark} missing from raw data.")

        bench_df = raw_data[self.benchmark].copy()
        bench_df['date'] = pd.to_datetime(bench_df['date'])
        bench_df = bench_df.sort_values('date').drop_duplicates('date')
        target_dates = bench_df['date'].values

        n_symbols = len(raw_data) - 1
        logger.info(f"Aligning {n_symbols} symbols to benchmark ({len(target_dates)} trading days)...")

        cleaned = {self.benchmark: bench_df.reset_index(drop=True)}
        dropped = 0

        for symbol, df in raw_data.items():
            if symbol == self.benchmark:
                continue
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').drop_duplicates('date')

            # Merge with target dates
            aligned = pd.DataFrame({'date': target_dates}).merge(df, on='date', how='left')

            # Check missingness
            missing_ratio = aligned['close'].isna().mean()
            if missing_ratio > 0.20:
                dropped += 1
                continue

            # Forward fill then backward fill for remaining NaNs
            aligned = aligned.ffill().bfill()
            cleaned[symbol] = aligned

        logger.info(f"Alignment complete. Retained: {len(cleaned)-1} stocks. Dropped: {dropped}.")
        return cleaned
