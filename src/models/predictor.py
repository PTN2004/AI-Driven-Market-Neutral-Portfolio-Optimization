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
        features = torch.tensor(df[self.feature_names].values, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model(features).cpu().numpy()
        df['predicted_mu'] = preds
        return df

    def predict_for_date(self, data: Union[Dict[str, pd.DataFrame], pd.DataFrame], target_date: str) -> Dict[str, float]:
        self.model.eval()
        mu_dict = {}
        with torch.no_grad():
            if isinstance(data, dict):
                for symbol, df in data.items():
                    if symbol == Config.BENCHMARK_TICKER:
                        continue
                    sub = df[df['date'] == target_date]
                    if not sub.empty:
                        feats = torch.tensor(sub[self.feature_names].values, dtype=torch.float32).to(self.device)
                        pred = self.model(feats).cpu().numpy()[0]
                        mu_dict[symbol] = float(pred)
            elif isinstance(data, pd.DataFrame):
                sub = data[data['date'] == target_date]
                if not sub.empty and 'symbol' in sub.columns:
                    feats = torch.tensor(sub[self.feature_names].values, dtype=torch.float32).to(self.device)
                    preds = self.model(feats).cpu().numpy()
                    for sym, pred in zip(sub['symbol'], preds):
                        mu_dict[sym] = float(pred)
                elif not sub.empty:
                    feats = torch.tensor(sub[self.feature_names].values, dtype=torch.float32).to(self.device)
                    pred = self.model(feats).cpu().numpy()[0]
                    mu_dict["asset"] = float(pred)
        return mu_dict
