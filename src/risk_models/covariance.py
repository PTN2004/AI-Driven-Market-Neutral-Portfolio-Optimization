from typing import Dict, Optional
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("RiskModel")

class RiskModel:
    """
    Computes cross-sectional asset covariance matrix using Ledoit-Wolf shrinkage
    to prevent ill-conditioned inversion in portfolio optimization.
    """
    def __init__(self, lookback: int = 60):
        self.lookback = lookback if lookback else Config.COV_LOOKBACK

    def compute_covariance(self, cleaned_data: Dict[str, pd.DataFrame], target_date: str) -> pd.DataFrame:
        symbols = [s for s in cleaned_data.keys() if s != Config.BENCHMARK_TICKER]
        returns_dict = {}

        for sym in symbols:
            df = cleaned_data[sym]
            sub = df[df['date'] <= target_date].tail(self.lookback + 1)
            if len(sub) > 1:
                ret = sub['close'].pct_change().dropna()
                returns_dict[sym] = ret.values[-self.lookback:]
            else:
                returns_dict[sym] = np.zeros(self.lookback)

        # Build DataFrame of trailing returns
        ret_df = pd.DataFrame(returns_dict).fillna(0.0)
        
        # Apply Ledoit-Wolf Shrinkage
        lw = LedoitWolf()
        try:
            cov_matrix = lw.fit(ret_df.values).covariance_
        except Exception as e:
            logger.warning(f"LedoitWolf shrinkage failed ({e}), falling back to sample covariance.")
            cov_matrix = ret_df.cov().values

        return pd.DataFrame(cov_matrix, index=symbols, columns=symbols)
