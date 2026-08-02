from typing import Dict, List
import numpy as np
import pandas as pd

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("FeatureEngineer")

class FeatureEngineer:
    @staticmethod
    def get_feature_names() -> List[str]:
        return [
            "rsi_14",
            "macd",
            "macd_signal",
            "macd_hist",
            "atr_14_norm",
            "std_20d",
            "ret_1d",
            "ret_5d",
            "ret_20d",
            "pe",
            "pb",
            "log_mcap"
        ]

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df['close']
        high = df['high']
        low = df['low']

        # 1. RSI (14)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=Config.RSI_WINDOW).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=Config.RSI_WINDOW).mean()
        rs = gain / (loss + 1e-8)
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # 2. MACD (12, 26, 9)
        ema_fast = close.ewm(span=Config.MACD_FAST, adjust=False).mean()
        ema_slow = close.ewm(span=Config.MACD_SLOW, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=Config.MACD_SIGNAL, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # 3. ATR (14) Normalized
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=Config.ATR_WINDOW).mean()
        df['atr_14_norm'] = atr / (close + 1e-8)

        # 4. Standard Deviation (20d)
        df['std_20d'] = close.pct_change().rolling(window=Config.STD_WINDOW).std()

        # 5. Lagged Returns
        for lag in Config.LAG_RETURNS:
            df[f'ret_{lag}d'] = close.pct_change(periods=lag)

        # 6. Fundamentals
        if 'pe' not in df.columns:
            df['pe'] = 15.0
        if 'pb' not in df.columns:
            df['pb'] = 2.0
        if 'market_cap' in df.columns:
            df['log_mcap'] = np.log(df['market_cap'] + 1.0)
        else:
            df['log_mcap'] = np.log(close * 1e7 + 1.0)

        # Target Return for supervised training (1 day ahead)
        df['target_ret_1d'] = close.pct_change().shift(-1)

        # Forward fill and backward fill any initial warm-up NaNs
        df = df.ffill().bfill()
        return df

    def compute_all_features(self, cleaned_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        logger.info("Computing technical and fundamental features...")
        feature_data = {}
        for symbol, df in cleaned_data.items():
            feature_data[symbol] = self.compute_features(df)
        logger.info(f"Feature engineering complete for {len(feature_data)} symbols.")
        return feature_data
