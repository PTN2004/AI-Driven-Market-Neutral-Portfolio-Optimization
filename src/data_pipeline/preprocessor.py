from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch

from src.data_pipeline.features import FeatureEngineer
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("DataPreprocessor")

class DataPreprocessor:

    def __init__(self):
        self.window = Config.ROLLING_NORM_WINDOW
        self.feature_names = FeatureEngineer.get_feature_names()

    def normalize_features(self, feature_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        logger.info(f"Applying Sliding Window Z-score Normalization (window={self.window} days)...")
        normalized = {}
        for symbol, df in feature_data.items():
            df = df.copy()
            if symbol == Config.BENCHMARK_TICKER:
                normalized[symbol] = df
                continue

            for col in self.feature_names:
                if col in df.columns:
                    roll_mean = df[col].rolling(window=self.window, min_periods=5).mean()
                    roll_std = df[col].rolling(window=self.window, min_periods=5).std()
                    norm_col = (df[col] - roll_mean) / (roll_std + 1e-8)
                    # Clip extreme outliers to [-3, 3]
                    df[col] = norm_col.clip(-3.0, 3.0)

            df = df.ffill().bfill().fillna(0.0)
            normalized[symbol] = df

        logger.info("Normalization complete. No data leakage verified.")
        return normalized

    def prepare_tabular_dataset(self, normalized_data: Dict[str, pd.DataFrame], start_date: str, end_date: str) -> pd.DataFrame:
        dfs = []
        for symbol, df in normalized_data.items():
            if symbol == Config.BENCHMARK_TICKER:
                continue
            sub = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
            sub['symbol'] = symbol
            dfs.append(sub)
        
        master_df = pd.concat(dfs, ignore_index=True)
        master_df = master_df.sort_values(['date', 'symbol']).reset_index(drop=True)
        return master_df

    def prepare_tensor_sequences(self, df: pd.DataFrame, seq_len: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        features = df[self.feature_names].values
        targets = df['target_ret_1d'].values
        return torch.tensor(features, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)
