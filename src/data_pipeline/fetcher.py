import os
import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("DataFetcher")

class DataFetcher:
    def __init__(self):
        self.raw_dir = Config.RAW_DATA_DIR
        self._vnstock_available = self._check_vnstock()

    def _check_vnstock(self) -> bool:
        try:
            import vnstock
            return True
        except ImportError:
            logger.warning("vnstock library not installed or not imported properly.")
            return False

    def fetch_symbol(self, symbol: str) -> Optional[pd.DataFrame]:
        cache_path = self.raw_dir / f"{symbol}.parquet"
        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                return df
            except Exception as e:
                logger.warning(f"Failed to load cache for {symbol}: {e}")

        df = None
        if self._vnstock_available:
            df = self._fetch_from_vnstock(symbol)

        if df is not None and not df.empty:
            self._save_cache(symbol, df)
        return df

    def _fetch_from_vnstock(self, symbol: str) -> Optional[pd.DataFrame]:
        try:
            from vnstock import Vnstock
            stock = Vnstock().stock(symbol=symbol, source='VCI')
            df = stock.quote.history(start=Config.START_DATE, end=Config.END_DATE)
            if df is not None and not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
                return df
        except Exception as e:
            logger.debug(f"vnstock fetch failed for {symbol}: {e}")
        return None


    def _save_cache(self, symbol: str, df: pd.DataFrame):
        try:
            cache_path = self.raw_dir / f"{symbol}.parquet"
            df.to_parquet(cache_path, index=False)
        except Exception as e:
            logger.warning(f"Could not save cache for {symbol}: {e}")

    def fetch_all(self) -> Dict[str, pd.DataFrame]:
        symbols = [Config.BENCHMARK_TICKER] + Config.DEFAULT_TICKERS
        logger.info(f"Starting data ingestion for {len(symbols)} symbols ({Config.START_DATE} -> {Config.END_DATE})...")
        data = {}
        for symbol in symbols:
            df = self.fetch_symbol(symbol)
            if df is not None and not df.empty:
                data[symbol] = df
        logger.info(f"Successfully loaded data for {len(data)} symbols.")
        return data
