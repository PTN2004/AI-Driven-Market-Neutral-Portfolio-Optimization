from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.data_pipeline.features import FeatureEngineer
from src.utils.config import Config

class StockDataset(Dataset):
    
    def __init__(self, df: pd.DataFrame):
        self.feature_names = FeatureEngineer.get_feature_names()
        self.features = torch.tensor(df[self.feature_names].values, dtype=torch.float32)
        self.targets = torch.tensor(df['target_ret_1d'].values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]

def _to_dict(loader: DataLoader) -> Dict[str, Any]:
    return {"batch_size": loader.batch_size, "num_batches": len(loader)}

def create_dataloaders(master_df: pd.DataFrame, batch_size: int = 64) -> Tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame]:
    train_df = master_df[master_df['date'] <= Config.TRAIN_END_DATE].copy()
    val_df = master_df[(master_df['date'] > Config.TRAIN_END_DATE) & (master_df['date'] <= Config.VAL_END_DATE)].copy()
    test_df = master_df[(master_df['date'] > Config.VAL_END_DATE) & (master_df['date'] <= Config.END_DATE)].copy()

    train_ds = StockDataset(train_df)
    val_ds = StockDataset(val_df)
    test_ds = StockDataset(test_df)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, test_df
