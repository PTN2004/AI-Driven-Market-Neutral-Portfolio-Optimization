from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from src.data_pipeline.features import FeatureEngineer
from src.utils.config import Config

class StockDataset(Dataset):
    
    def __init__(self, df: pd.DataFrame, seq_len: int = 10):
        self.seq_len = seq_len
        self.feature_names = FeatureEngineer.get_feature_names()
        self.features = torch.tensor(df[self.feature_names].values, dtype=torch.float32)
        self.targets = torch.tensor(df['target_ret_1d'].values, dtype=torch.float32)
        
        self.valid_indices = []
        if 'symbol' in df.columns:
            symbols = df['symbol'].values
            for i in range(self.seq_len - 1, len(df)):
                if symbols[i] == symbols[i - self.seq_len + 1]:
                    self.valid_indices.append(i)
        else:
            self.valid_indices = list(range(self.seq_len - 1, len(df)))

        self.valid_indices = np.array(self.valid_indices, dtype=np.int64)
        if 'date' in df.columns:
            self.dates = df['date'].iloc[self.valid_indices].values
        else:
            self.dates = np.zeros(len(self.valid_indices))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        real_idx = self.valid_indices[idx]
        seq_features = self.features[real_idx - self.seq_len + 1 : real_idx + 1]
        target = self.targets[real_idx]
        return seq_features, target

class DayBatchSampler(Sampler):
    """
    Groups indices belonging to the exact same trading date into a single batch.
    Enables true daily cross-sectional rank/IC optimization.
    """
    def __init__(self, dates: np.ndarray, shuffle: bool = True, min_stocks: int = 2):
        super(DayBatchSampler, self).__init__(None)
        self.shuffle = shuffle
        self.min_stocks = min_stocks
        
        date_to_indices: Dict[Any, List[int]] = {}
        for idx, d in enumerate(dates):
            if d not in date_to_indices:
                date_to_indices[d] = []
            date_to_indices[d].append(idx)
            
        self.batches = [
            indices for d, indices in date_to_indices.items()
            if len(indices) >= self.min_stocks
        ]

    def __iter__(self):
        batches = list(self.batches)
        if self.shuffle:
            np.random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self) -> int:
        return len(self.batches)

def _to_dict(loader: DataLoader) -> Dict[str, Any]:
    return {"batch_size": loader.batch_size if loader.batch_size is not None else "daily", "num_batches": len(loader)}

def create_dataloaders(
    master_df: pd.DataFrame, 
    batch_size: int = 64,
    use_day_batching: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame]:
    master_df = master_df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    train_df = master_df[master_df['date'] <= Config.TRAIN_END_DATE].copy()
    val_df = master_df[(master_df['date'] > Config.TRAIN_END_DATE) & (master_df['date'] <= Config.VAL_END_DATE)].copy()
    test_df = master_df[(master_df['date'] > Config.VAL_END_DATE) & (master_df['date'] <= Config.END_DATE)].copy()

    train_ds = StockDataset(train_df, seq_len=Config.SEQ_LEN)
    val_ds = StockDataset(val_df, seq_len=Config.SEQ_LEN)
    test_ds = StockDataset(test_df, seq_len=Config.SEQ_LEN)

    if use_day_batching:
        train_sampler = DayBatchSampler(train_ds.dates, shuffle=True)
        val_sampler = DayBatchSampler(val_ds.dates, shuffle=False)
        test_sampler = DayBatchSampler(test_ds.dates, shuffle=False)

        train_loader = DataLoader(train_ds, batch_sampler=train_sampler)
        val_loader = DataLoader(val_ds, batch_sampler=val_sampler)
        test_loader = DataLoader(test_ds, batch_sampler=test_sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, test_df
