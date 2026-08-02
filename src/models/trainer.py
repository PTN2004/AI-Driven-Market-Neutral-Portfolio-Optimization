import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("AlphaTrainer")

class QuantLoss(nn.Module):
    def __init__(self, ic_weight: float = 0.5):
        super(QuantLoss, self).__init__()
        self.ic_weight = ic_weight
        self.mse = nn.MSELoss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(preds, targets)
        
        # Pearson IC loss
        pred_mean = preds - preds.mean()
        target_mean = targets - targets.mean()
        cov = (pred_mean * target_mean).mean()
        std_pred = torch.sqrt((pred_mean ** 2).mean() + 1e-8)
        std_target = torch.sqrt((target_mean ** 2).mean() + 1e-8)
        corr = cov / (std_pred * std_target)
        ic_loss = 1.0 - corr

        return (1.0 - self.ic_weight) * mse_loss + self.ic_weight * ic_loss

class AlphaTrainer:
    def __init__(self, model: nn.Module, lr: float = 0.001, weight_decay: float = 0.0001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = QuantLoss(ic_weight=Config.IC_LOSS_WEIGHT)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            preds = self.model(x)
            loss = self.criterion(preds, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(x)
        return total_loss / len(train_loader.dataset)

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                preds = self.model(x)
                loss = self.criterion(preds, y)
                total_loss += loss.item() * len(x)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        preds_arr = np.concatenate(all_preds)
        targets_arr = np.concatenate(all_targets) 
        
        # Calculate validation IC
        p_dev = preds_arr - preds_arr.mean()
        t_dev = targets_arr - targets_arr.mean()
        ic = np.mean(p_dev * t_dev) / (np.std(preds_arr) * np.std(targets_arr) + 1e-8)

        return {
            "loss": total_loss / len(val_loader.dataset),
            "ic": float(ic)
        }

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 20) -> Dict[str, List[float]]:
        logger.info(f"Starting PyTorch training on {self.device} for {epochs} epochs...")
        history = {"train_loss": [], "val_loss": [], "val_ic": []}
        best_val_loss = float('inf')

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["loss"])
            history["val_ic"].append(val_metrics["ic"])

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                self.save_model("best_alpha_model.pth")

            if epoch % 5 == 0 or epoch == 1:
                logger.info(f"Epoch {epoch:02d}/{epochs:02d} - Train Loss: {train_loss:.4f} - Val Loss: {val_metrics['loss']:.4f} - Val IC: {val_metrics['ic']:.4f}")

        logger.info("Training completed.")
        return history

    def save_model(self, filename: str):
        path = Config.MODELS_DIR / filename
        torch.save(self.model.state_dict(), path)

    def load_model(self, filename: str):
        path = Config.MODELS_DIR / filename
        if path.exists():
            self.model.load_state_dict(torch.load(path, map_location=self.device))
