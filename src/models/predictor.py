from typing import Dict, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data_pipeline.features import FeatureEngineer
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("AlphaPredictor")

class AlphaPredictor:
    """
    Inference engine to predict expected returns (mu) from trained PyTorch models.
    """
    def __init__(self, model: nn.Module):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.feature_names = FeatureEngineer.get_feature_names()

    def predict_all(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Ensure it's sorted properly
        df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        features_np = df[self.feature_names].values
        symbols = df['symbol'].values
        
        seq_len = Config.SEQ_LEN
        preds = np.zeros(len(df))
        
        # We can only predict if we have a full sequence for the symbol
        for i in range(len(df)):
            if i >= seq_len - 1 and symbols[i] == symbols[i - seq_len + 1]:
                seq = features_np[i - seq_len + 1 : i + 1]
                # Pass as (1, seq_len, num_features)
                feats = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    preds[i] = self.model(feats).cpu().numpy()[0]
            else:
                preds[i] = 0.0 # Default if not enough history
                
        df['predicted_mu'] = preds
        return df

    def predict_for_date(self, data: Union[Dict[str, pd.DataFrame], pd.DataFrame], target_date: str) -> Dict[str, float]:
        self.model.eval()
        mu_dict = {}
        seq_len = Config.SEQ_LEN
        
        with torch.no_grad():
            if isinstance(data, dict):
                for symbol, df in data.items():
                    if symbol == Config.BENCHMARK_TICKER:
                        continue
                    # Extract history up to target_date
                    sub = df[df['date'] <= target_date].tail(seq_len)
                    if len(sub) == seq_len:
                        seq = sub[self.feature_names].values
                        feats = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
                        pred = self.model(feats).cpu().numpy()[0]
                        mu_dict[symbol] = float(pred)
                        
            elif isinstance(data, pd.DataFrame):
                # We need historical rows for each symbol up to target_date
                # This is less efficient but necessary for sequence models
                sub_df = data[data['date'] <= target_date]
                symbols = sub_df['symbol'].unique()
                
                for sym in symbols:
                    sym_data = sub_df[sub_df['symbol'] == sym].tail(seq_len)
                    if len(sym_data) == seq_len:
                        seq = sym_data[self.feature_names].values
                        feats = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
                        pred = self.model(feats).cpu().numpy()[0]
                        mu_dict[sym] = float(pred)
                        
        return mu_dict
