import os
from pathlib import Path

class Config:
    """
    Central Configuration for AI-Driven Market Neutral Portfolio Optimization.
    """
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODELS_DIR = DATA_DIR / "models"
    OUTPUT_DIR = DATA_DIR / "output"

    for d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    BENCHMARK_TICKER = "VNINDEX"
    DEFAULT_TICKERS = [
        'VCB', 'BID', 'CTG', 'TCB', 'VPB', 'MBB', 'ACB', 'HPG', 'FPT', 'MWG',
        'PNJ', 'MSN', 'VNM', 'VIC', 'VHM', 'VRE', 'GAS', 'PLX', 'SSI', 'VND',
        'HCM', 'VCI', 'DGC', 'DCM', 'DPM', 'REE', 'GMD', 'VJC', 'KDH', 'NLG',
        'PDR', 'DIG', 'KBC', 'SZC', 'IDC', 'STB', 'TPB', 'VIB', 'LPB', 'SHB',
        'OCB', 'MSB', 'EIB', 'SSB', 'HDB', 'POW', 'SAB', 'BVH', 'VGC', 'VGP'
    ]

    START_DATE = "2021-01-01"
    END_DATE = "2025-12-30"
    TRAIN_END_DATE = "2024-12-31"
    VAL_END_DATE = "2025-06-30"
    TEST_START_DATE = "202-07-01"

    # Technical indicator parameters
    RSI_WINDOW = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    ATR_WINDOW = 14
    STD_WINDOW = 20
    LAG_RETURNS = [1, 5, 20]
    ROLLING_NORM_WINDOW = 120

    # AI Model Hyperparameters
    SEQ_LEN = 10
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0001
    EPOCHS = 50
    HIDDEN_DIMS = [128, 64, 32]
    DROPOUT_RATE = 0.25
    IC_LOSS_WEIGHT = 0.8

    # Risk Model & Optimization Parameters
    COV_LOOKBACK = 60
    RISK_AVERSION_LAMBDA = 5.0
    MAX_WEIGHT_PER_ASSET = 0.1
    BETA_TOLERANCE = 0.02
    TRANSACTION_FEE = 0.002
    REBALANCE_FREQ = "weekly"
