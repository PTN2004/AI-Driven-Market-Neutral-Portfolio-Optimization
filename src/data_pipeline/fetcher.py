import pandas as pd
from typing import Dict, List, Optional
import time
import random

from src.utils.config import Config
from src.utils.logger import get_logger
from vnstock import Quote, Listing

logger = get_logger("DataFetcher")


class DataFetcher:
    def __init__(self):
        self.raw_dir = Config.RAW_DATA_DIR

    def fetch_symbol(self, symbol: str) -> Optional[pd.DataFrame]:
        logger.info(f"fetching data symbol {symbol}")
        cache_path = self.raw_dir / f"{symbol}.parquet"
        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                return df
            except Exception as e:
                logger.warning(f"Failed to load cache for {symbol}: {e}")

        df = None
        df = self._fetch_from_vnstock(symbol)

        if df is not None and not df.empty:
            self._save_cache(symbol, df)
        return df

    def get_group_symbol(self, group_name: str = 'VN30') -> Optional[pd.Series]:
        listing = Listing()
        group_list = listing.symbols_by_group(group_name)
        logger.info(f"Get group symbol {group_list.to_list()}")
        return group_list

    def _fetch_from_vnstock(self, symbol: str, retries: int = 3) -> Optional[pd.DataFrame]:
        for attempt in range(retries):
            try:
                sleep_time = random.uniform(2.0, 4.0)
                logger.info(
                    f"Đang chờ {sleep_time:.2f}s trước khi tải {symbol}...")
                time.sleep(sleep_time)
                stock = Quote(symbol=symbol, source='VCI', random_agent=True)
                df = stock.history(start=Config.START_DATE,
                                   end=Config.END_DATE)
                if df is not None and not df.empty:
                    df['date'] = pd.to_datetime(df['time'])
                    df = df.sort_values('date').reset_index(drop=True)
                    return df
            except Exception as e:
                logger.debug(f"vnstock fetch failed for {symbol}: {e}")
                logger.warning(
                    f"Lần thử {attempt + 1}/{retries} thất bại cho {symbol}. Lỗi: {e}")

                if attempt < retries:
                    wait_time = (attempt + 1) * 10
                    logger.warning(
                        f"Có thể bị Rate Limit. Đang làm mát {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Đã bỏ qua mã {symbol} sau {retries} lần thử.")
        return None

    def _save_cache(self, symbol: str, df: pd.DataFrame):
        try:
            cache_path = self.raw_dir / f"{symbol}.parquet"
            df.to_parquet(cache_path, index=False)
        except Exception as e:
            logger.warning(f"Could not save cache for {symbol}: {e}")

    def fetch_all(self, symbols: List[str] = None) -> Dict[str, pd.DataFrame]:
        if symbols is None:
            symbols = [Config.BENCHMARK_TICKER] + Config.DEFAULT_TICKERS
        logger.info(
            f"Starting data ingestion for {len(symbols)} symbols ({Config.START_DATE} -> {Config.END_DATE})...")
        data = {}
        for symbol in symbols:
            df = self.fetch_symbol(symbol)
            if df is not None and not df.empty:
                data[symbol] = df
        logger.info(f"Successfully loaded data for {len(data)} symbols.")
        return data

    def fetch_all_group(self, symbol_group: str = "VN100"):
        group_symbol = self.get_group_symbol(symbol_group)
        if group_symbol is None or group_symbol.empty:
            logger.warning(f"No symbols found for group {symbol_group}.")
            return {}

        symbols = group_symbol.to_list()
        data = self.fetch_all(symbols=symbols)
        return data
