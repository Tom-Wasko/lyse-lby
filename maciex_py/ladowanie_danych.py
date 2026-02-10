import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta
from tqdm.notebook import tqdm

# Opis: Definicja listy symboli (tickery) do analizy.

SYMBOLS_500 = [
 'MMM',
 'AOS',
 'ABT',
 'ABBV',
 'ACN',
 'AMD',
 'AES',
 'AFL',
 'A',
 'APD',
 'ABNB',
 'AKAM',
 'ALB',
 'ARE',
 'ALGN',
 'ALLE',
 'LNT',
 'ALL',
 'GOOGL',
 'GOOG',
 'MO',
 'AMZN',
 'AMCR',
 'AEE',
 'AEP',
 'AXP',
 'AIG',
 'AMT',
 'AWK',
 'AMP',
 'AME',
 'AMGN',
 'APH',
 'ADI',
 'AON',
 'APA',
 'APO',
 'AAPL',
 'AMAT',
 'APP',
 'APTV',
 'ACGL',
 'ADM',
 'ARES',
 'ANET',
 'AJG',
 'AIZ',
 'T',
 'ATO',
 'ADSK',
 'ADP',
 'AVB',
 'AVY',
 'AXON',
 'BKR',
 'BALL',
 'BAC',
 'BAX',
 'BDX',
 'BBY',
 'TECH',
 'BIIB',
 'BX',
 'XYZ',
 'BK',
 'BA',
 'BSX',
 'BMY',
 'AVGO',
 'BR',
 'BRO',
 'BLDR',
 'BG',
 'BXP',
 'CHRW',
 'CDNS',
 'CPT',
 'CPB',
 'COF',
 'CAH',
 'CCL',
 'CARR',
 'CVNA',
 'CAT',
 'CBOE',
 'CBRE',
 'CDW',
 'COR',
 'CNC',
 'CNP',
 'CF',
 'CRL',
 'SCHW',
 'CHTR',
 'CVX',
 'CMG',
 'CB',
 'CHD',
 'CI',
 'CINF',
 'CTAS',
 'CSCO',
 'C',
 'CFG',
 'CLX',
 'CME',
 'CMS',
 'KO',
 'CTSH',
 'COIN',
 'CL',
 'CMCSA',
 'FIX',
 'CAG',
 'COP',
 'ED',
 'STZ',
 'CEG',
 'COO',
 'CPRT',
 'GLW',
 'CPAY',
 'CTVA',
 'CSGP',
 'CTRA',
 'CRH',
 'CRWD',
 'CCI',
 'CSX',
 'CMI',
 'CVS',
 'DHR',
 'DRI',
 'DDOG',
 'DVA',
 'DAY',
 'DECK',
 'DE',
 'DELL',
 'DAL',
 'DVN',
 'DXCM',
 'FANG',
 'DLR',
 'DG',
 'DLTR',
 'D',
 'DPZ',
 'DASH',
 'DOV',
 'DOW',
 'DHI',
 'DTE',
 'DUK',
 'DD',
 'ETN',
 'EBAY',
 'ECL',
 'EIX',
 'EW',
 'EA',
 'ELV',
 'EME',
 'EMR',
 'ETR',
 'EOG',
 'EPAM',
 'EQT',
 'EFX',
 'EQR',
 'ERIE',
 'ESS',
 'EL',
 'EG',
 'EVRG',
 'ES',
 'EXC',
 'EXE',
 'EXPE',
 'EXPD',
 'EXR',
 'XOM',
 'FFIV',
 'FDS',
 'FAST',
 'FRT',
 'FDX',
 'FIS',
 'FITB',
 'FSLR',
 'FE',
 'FISV',
 'F',
 'FTNT',
 'FTV',
 'FOXA',
 'FOX',
 'BEN',
 'FCX',
 'GRMN',
 'IT',
 'GE',
 'GEHC',
 'GEV',
 'GEN',
 'GNRC',
 'GD',
 'GIS',
 'GM',
 'GPC',
 'GILD',
 'GPN',
 'GL',
 'GDDY',
 'GS',
 'HAL',
 'HIG',
 'HAS',
 'HCA',
 'DOC',
 'HSIC',
 'HSY',
 'HPE',
 'HLT',
 'HOLX',
 'HD',
 'HON',
 'HRL',
 'HST',
 'HWM',
 'HPQ',
 'HUBB',
 'HUM',
 'HBAN',
 'HII',
 'IBM',
 'IEX',
 'ITW',
 'INCY',
 'IR',
 'PODD',
 'INTC',
 'IBKR',
 'ICE',
 'IFF',
 'IP',
 'ISRG',
 'IVZ',
 'INVH',
 'IQV',
 'IRM',
 'JBHT',
 'JBL',
 'JKHY',
 'J',
 'JNJ',
 'JCI',
 'JPM',
 'KVUE',
 'KDP',
 'KEY',
 'KEYS',
 'KMB',
 'KIM',
 'KMI',
 'KKR',
 'KHC',
 'KR',
 'LHX',
 'LH',
 'LRCX',
 'LW',
 'LVS',
 'LDOS',
 'LEN',
 'LII',
 'LIN',
 'LYV',
 'LMT',
 'L',
 'LOW',
 'LULU',
 'LYB',
 'MTB',
 'MPC',
 'MAR',
 'MMC',
 'MLM',
 'MAS',
 'MA',
 'MTCH',
 'MKC',
 'MCD',
 'MCK',
 'MDT',
 'MRK',
 'META',
 'MET',
 'MGM',
 'MCHP',
 'MU',
 'MSFT',
 'MAA',
 'MRNA',
 'MOH',
 'TAP',
 'MDLZ',
 'MNST',
 'MCO',
 'MS',
 'MOS',
 'MSI',
 'NDAQ',
 'NTAP',
 'NFLX',
 'NEM',
 'NWSA',
 'NWS',
 'NEE',
 'NKE',
 'NI',
 'NDSN',
 'NSC',
 'NTRS',
 'NOC',
 'NCLH',
 'NRG',
 'NUE',
 'NVDA',
 'NXPI',
 'ORLY',
 'OXY',
 'ODFL',
 'OMC',
 'ON',
 'OKE',
 'ORCL',
 'OTIS',
 'PCAR',
 'PKG',
 'PLTR',
 'PANW',
 'PSKY',
 'PH',
 'PAYX',
 'PAYC',
 'PYPL',
 'PNR',
 'PEP',
 'PFE',
 'PCG',
 'PM',
 'PSX',
 'PNW',
 'PNC',
 'POOL',
 'PPG',
 'PPL',
 'PFG',
 'PG',
 'PGR',
 'PLD',
 'PRU',
 'PEG',
 'PTC',
 'PSA',
 'PHM',
 'PWR',
 'QCOM',
 'DGX',
 'Q',
 'RL',
 'RJF',
 'RTX',
 'O',
 'REG',
 'RF',
 'RSG',
 'RMD',
 'RVTY',
 'HOOD',
 'ROK',
 'ROL',
 'ROST',
 'RCL',
 'SPGI',
 'CRM',
 'SNDK',
 'SBAC',
 'SLB',
 'STX',
 'SRE',
 'NOW',
 'SHW',
 'SPG',
 'SWKS',
 'SJM',
 'SW',
 'SNA',
 'SOLV',
 'SO',
 'LUV',
 'SWK',
 'SBUX',
 'STT',
 'STLD',
 'STE',
 'SYK',
 'SMCI',
 'SYF',
 'SNPS',
 'SYY',
 'TMUS',
 'TROW',
 'TTWO',
 'TPR',
 'TRGP',
 'TGT',
 'TEL',
 'TDY',
 'TER',
 'TSLA',
 'TXN',
 'TPL',
 'TXT',
 'TJX',
 'TKO',
 'TTD',
 'TSCO',
 'TT',
 'TRV',
 'TRMB',
 'TFC',
 'TYL',
 'TSN',
 'USB',
 'UBER',
 'UDR',
 'ULTA',
 'UNP',
 'UAL',
 'UPS',
 'UHS',
 'VLO',
 'VTR',
 'VLTO',
 'VRSN',
 'VRSK',
 'VZ',
 'VRTX',
 'VTRS',
 'VICI',
 'V',
 'VST',
 'VMC',
 'WRB',
 'WAB',
 'WMT',
 'DIS',
 'WBD',
 'WM',
 'WAT',
 'WEC',
 'WFC',
 'WELL',
 'WST',
 'WDC',
 'WY',
 'WSM',
 'WMB',
 'WTW',
 'WDAY',
 'WYNN',
 'XEL',
 'XYL',
 'YUM',
 'ZBRA',
 'ZBH',
 'ZTS']

CRYPTO_TOP_50 = [
    "BTC-USD",   # Bitcoin
    "ETH-USD",   # Ethereum
    "USDT-USD",  # Tether
    "BNB-USD",   # Binance Coin
    "SOL-USD",   # Solana
    "USDC-USD",  # USD Coin
    "XRP-USD",   # XRP
    "TON11419-USD",  # Toncoin (Yahoo uses this weird symbol)
    "DOGE-USD",  # Dogecoin
    "ADA-USD",   # Cardano
    "AVAX-USD",  # Avalanche
    "SHIB-USD",  # Shiba Inu
    "TRX-USD",   # TRON
    "DOT-USD",   # Polkadot
    "LINK-USD",  # Chainlink
    "MATIC-USD", # Polygon
    "BCH-USD",   # Bitcoin Cash
    "ICP-USD",   # Internet Computer
    "LTC-USD",   # Litecoin
    "UNI7083-USD", # Uniswap (Yahoo-specific code)
    "ATOM-USD",  # Cosmos
    "ETC-USD",   # Ethereum Classic
    "XLM-USD",   # Stellar
    "FIL-USD",   # Filecoin
    "APT-USD",   # Aptos
    "HBAR-USD",  # Hedera
    "NEAR-USD",  # Near Protocol
    "OP-USD",    # Optimism
    #"IMX-USD",   # Immutable
    "INJ-USD",   # Injective
    "ARB-USD",   # Arbitrum
    "RNDR-USD",  # Render
    #"GRT-USD",   # The Graph
    "AAVE-USD",  # Aave
    "FLOW-USD",  # Flow
    "MKR-USD",   # Maker
    "SAND-USD",  # The Sandbox
    "THETA-USD", # Theta
    "KAS-USD",   # Kaspa
    "EGLD-USD",  # MultiversX
    "AXS-USD",   # Axie Infinity
    "XTZ-USD",   # Tezos
    "NEO-USD",   # Neo
    "MINA-USD",  # Mina
    "SNX-USD",   # Synthetix
    "CRV-USD",   # Curve
    "CHZ-USD",   # Chiliz
    "DYDX-USD",  # dYdX
]

COMMODITY_SYMBOLS = [

    # =========================
    # 🥇 PRECIOUS METALS
    # =========================
    "GC=F",    # Gold
    "SI=F",    # Silver
    "PL=F",    # Platinum
    "PA=F",    # Palladium

    # =========================
    # 🏗 INDUSTRIAL / BASE METALS
    # =========================
    "HG=F",    # Copper
    "ALI=F",   # Aluminum (unstable)
    "ZNC=F",   # Zinc (unstable)
    "NICKEL=F",# Nickel (very unstable)
    "TIN=F",   # Tin (rarely works)

    # =========================
    # 🛢 ENERGY
    # =========================
    "CL=F",    # Crude Oil WTI
    "BZ=F",    # Brent Crude
    "NG=F",    # Natural Gas
    "HO=F",    # Heating Oil
    "RB=F",    # Gasoline RBOB
    "MGC=F",   # Micro Gold
    "QG=F",    # Mini Natural Gas

    # =========================
    # 🌾 GRAINS
    # =========================
    "ZC=F",    # Corn
    "ZW=F",    # Wheat
    "ZS=F",    # Soybeans
    "ZM=F",    # Soybean Meal
    "ZL=F",    # Soybean Oil
    "ZR=F",    # Rough Rice
    "ZO=F",    # Oats

    # =========================
    # ☕ SOFT COMMODITIES
    # =========================
    "KC=F",    # Coffee
    "SB=F",    # Sugar
    "CT=F",    # Cotton
    "CC=F",    # Cocoa
    "OJ=F",    # Orange Juice
    "LB=F",    # Lumber

    # =========================
    # 🐄 LIVESTOCK
    # =========================
    "LE=F",    # Live Cattle
    "GF=F",    # Feeder Cattle
    "HE=F",    # Lean Hogs

    # =========================
    # 🧪 CHEMICALS / FERTILIZERS (SPOTTY)
    # =========================
    "UREA=F",  # Urea (often empty)
    "DAPHOS=F",# DAP Fertilizer (rare)

    # =========================
    # 🌍 MACRO / COMMODITY INDICES
    # =========================
    "^BCOM",   # Bloomberg Commodity Index
    "^CRB",    # CRB Index

    # =========================
    # 📦 COMMODITY ETF PROXIES (VERY RELIABLE)
    # =========================
    "GLD",     # Gold ETF
    "SLV",     # Silver ETF
    "USO",     # Oil ETF
    "UNG",     # Natural Gas ETF
    "DBC",     # Broad commodities
    "DBA",     # Agriculture basket
    "DBB",     # Base metals basket
    "WOOD",    # Timber
    "CORN",    # Corn ETF
    "WEAT",    # Wheat ETF
    "SOYB",    # Soybeans ETF

]

def load_symbol_data(symbol: str, data_dir: str, window: Optional[int] = None) -> Optional[pd.DataFrame]:
    """Load CSV for a symbol from data_dir and return cleaned daily DataFrame.

    Returns None if file missing or schema invalid.
    """
    path = os.path.join(data_dir, f"{symbol}.csv")
    if not os.path.isfile(path):
        # missing file
        return None

    df = pd.read_csv(path, skiprows=[1])
    # normalize datetime column name
    if "Datetime" in df.columns and "Date" not in df.columns:
        df.rename(columns={"Datetime": "Date"}, inplace=True)

    REQUIRED = ["Date", "Open", "High", "Low", "Close", "Volume"]
    if not all(col in df.columns for col in REQUIRED):
        return None

    df = df[REQUIRED].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    df["Weekday"] = df.index.day_name()

    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)

    if window is not None:
        return df.tail(window)
    return df



def add_indicators(
    df,
    settings,
    ema_periods=(20, 50, 80, 200),
    sma_periods=(10, 50, 100),
    tema_periods=(10, 20, 50),
    aroon_len=14,
    rsi_len=14,
    hma_len=55,
    supertrend_period=10,
    supertrend_mult=3,
    wr_len=14,
    bb_len=20,
    bb_k=2,
    atr_len=14,
    cmo_len=5,
    vol_forecast_n=5,
):

    import pandas as pd
    import numpy as np
    
    # PERMANENT FIX: Remove duplicate columns at the start
    # This prevents duplicate columns when add_indicators is called multiple times
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # ===== Helper (NEVER returns None) =====
    def safe_series(result):
        if result is None:
            return pd.Series(0.0, index=df.index)
        if isinstance(result, pd.Series):
            return result.reindex(df.index).fillna(0.0)
        return pd.Series(result, index=df.index).fillna(0.0)

    def safe_df_col(result, like):
        try:
            if result is None or result.empty:
                return pd.Series(0.0, index=df.index)
            col = result.filter(like=like)
            if col.shape[1] == 0:
                return pd.Series(0.0, index=df.index)
            return col.iloc[:, 0].reindex(df.index).fillna(0.0)
        except:
            return pd.Series(0.0, index=df.index)

    # ===== EMA =====
    for p in ema_periods:
        try:
            df[f"EMA_{p}"] = safe_series(ta.ema(df["Close"], length=p))
        except:
            df[f"EMA_{p}"] = 0.0

    # ===== TEMA =====
    for p in tema_periods:
        try:
            df[f"TEMA_{p}"] = safe_series(ta.tema(df["Close"], length=p))
        except:
            df[f"TEMA_{p}"] = 0.0

        df[f"TEMA_SLOPE_{p}"] = df[f"TEMA_{p}"].diff(5).fillna(0)
        df[f"CLOSE_GT_TEMA_{p}"] = (df["Close"] > df[f"TEMA_{p}"]).astype(int)

    # ===== SMA =====
    for p in sma_periods:
        try:
            df[f"SMA_{p}"] = safe_series(ta.sma(df["Close"], length=p))
        except:
            df[f"SMA_{p}"] = 0.0

    # ===== EMA derivative =====
    if "EMA_20" in df:
        df["EMA_DERIV"] = df["EMA_20"].diff().fillna(0)
        df["EMA_DERIV_PCT"] = (df["EMA_DERIV"] / df["Close"]).replace([np.inf, -np.inf], 0).fillna(0)

    # ===== AROON =====
    try:
        aroon_result = ta.aroon(df["High"], df["Low"], length=aroon_len)
        if aroon_result is not None and not aroon_result.empty:
            df["AROON"] = aroon_result.iloc[:, -1].reindex(df.index).fillna(0)
        else:
            df["AROON"] = 0.0
    except:
        df["AROON"] = 0.0

    # ===== Other indicators =====
    df["RSI"] = safe_series(ta.rsi(df["Close"], length=rsi_len))
    df["CMO"] = safe_series(ta.cmo(df["Close"], length=cmo_len))
    df["HMA"] = safe_series(ta.hma(df["Close"], length=hma_len))
    df["WILLR"] = safe_series(ta.willr(df["High"], df["Low"], df["Close"], length=wr_len))

    # ===== Bollinger Bands =====
    try:
        bb = ta.bbands(df["Close"], length=bb_len, std=bb_k)
        df["BB_LOWER"] = safe_df_col(bb, "BBL")
        df["BB_MID"] = safe_df_col(bb, "BBM")
        df["BB_UPPER"] = safe_df_col(bb, "BBU")
    except:
        df["BB_LOWER"] = df["BB_MID"] = df["BB_UPPER"] = 0.0

    df["BB_PCT"] = ((df["Close"] - df["BB_LOWER"]) / (df["BB_UPPER"] - df["BB_LOWER"])).replace([np.inf, -np.inf], 0).fillna(0)
    df["BB_WIDTH"] = ((df["BB_UPPER"] - df["BB_LOWER"]) / df["BB_MID"]).replace([np.inf, -np.inf], 0).fillna(0)

    # ===== ATR =====
    df["ATR"] = safe_series(ta.atr(df["High"], df["Low"], df["Close"], length=atr_len))
    df["ATR_PCT"] = (df["ATR"] / df["Close"]).replace([np.inf, -np.inf], 0).fillna(0)

    # ===== Volume context =====
    df["vol_sma20"] = safe_series(ta.sma(df["Volume"].astype(float), length=settings["vol_ratio_window"]))
    df["vol_ratio"] = (df["Volume"] / df["vol_sma20"]).replace([np.inf, -np.inf], 0).fillna(0)
    df["VOL_SIGNIFICANT"] = (df["vol_ratio"] >= settings['vol_ratio_threshold']).astype(int)

    # ===== Short downtrend =====
    slope_period = 10
    if "EMA_20" in df and "EMA_50" in df:
        df["EMA20_SLOPE_N"] = df["EMA_20"].diff(slope_period).fillna(0)
        df["DOWNTREND_SHORT"] = (
            (df["CMO"].shift(0) < settings['cmo_thres']) |
            (df["CMO"].shift(1) < settings['cmo_thres_prev'])
        ).astype(int)

    # ===== Vol forecast =====
    df["VOL_EST_NEXT"] = df["ATR_PCT"] * (vol_forecast_n ** 0.5)

    df["VOL_EST_Z"] = (
        (df["VOL_EST_NEXT"] - df["VOL_EST_NEXT"].rolling(100).mean()) /
        df["VOL_EST_NEXT"].rolling(100).std()
    ).replace([np.inf, -np.inf], 0).fillna(0)

    df["VOL_EST_PRICE"] = df["Close"] * df["ATR_PCT"] * np.sqrt(vol_forecast_n)

    return df



# Opis: Implementacje wskaźników technicznych (RSI/ATR/Supertrend i inne).

def aroon_oscillator(high, low, length=14):
    aroon_up = pd.Series(np.nan, index=high.index)
    aroon_down = pd.Series(np.nan, index=low.index)

    for i in range(length - 1, len(high)):
        window_high = high.iloc[i - length + 1 : i + 1].values
        window_low  = low.iloc[i - length + 1 : i + 1].values

        periods_since_high = length - 1 - np.argmax(window_high)
        periods_since_low  = length - 1 - np.argmin(window_low)

        aroon_up.iloc[i]   = 100 * (length - periods_since_high) / length
        aroon_down.iloc[i] = 100 * (length - periods_since_low) / length

    return aroon_up - aroon_down

def rsi(close, length=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def wma(series, length):
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def hma(close, length=55):
    def wma(series, length):
        weights = np.arange(1, length + 1)
        return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    
    half = int(length / 2)
    sqrt_len = int(np.sqrt(length))
    return wma(2 * wma(close, half) - wma(close, length), sqrt_len)

def supertrend(df, period=10, multiplier=3):
    """
    Vectorized Supertrend implementation using pandas_ta.
    """
    hl2 = (df['High'] + df['Low']) / 2
    atr = ta.atr(df['High'], df['Low'], df['Close'], length=period)
    
    # Upper/Lower bands
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    # Final bands - initialized with basic bands
    final_upper = upper.copy()
    final_lower = lower.copy()

    # Trend and final bands calculation (vectorized iterative logic)
    # We still need to iterate slightly because of dependencies on previous final bands
    # but we can do it faster than full row iteration.
    close = df['Close'].values
    upper_v = upper.values
    lower_v = lower.values
    final_upper_v = final_upper.values
    final_lower_v = final_lower.values
    
    trend = np.zeros(len(df))
    
    for i in range(1, len(df)):
        # Calculate Final Upper Band
        if upper_v[i] < final_upper_v[i-1] or close[i-1] > final_upper_v[i-1]:
            final_upper_v[i] = upper_v[i]
        else:
            final_upper_v[i] = final_upper_v[i-1]
            
        # Calculate Final Lower Band
        if lower_v[i] > final_lower_v[i-1] or close[i-1] < final_lower_v[i-1]:
            final_lower_v[i] = lower_v[i]
        else:
            final_lower_v[i] = final_lower_v[i-1]
            
        # Determine Trend
        if close[i] > final_upper_v[i-1]:
            trend[i] = 1
        elif close[i] < final_lower_v[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
            
    # Combine trend into one series
    st = np.where(trend == 1, final_lower_v, final_upper_v)
    return pd.Series(st, index=df.index)

def williams_r(high, low, close, length=14):
    hh = high.rolling(length).max()
    ll = low.rolling(length).min()
    return -100 * (hh - close) / (hh - ll)

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

# ===== Chande Momentum Oscillator (CMO) =====
def chande_momentum(close, length):
    delta = close.diff()

    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    sum_up = up.rolling(length).sum()
    sum_down = down.rolling(length).sum()

    cmo = 100 * (sum_up - sum_down) / (sum_up + sum_down)
    return cmo


# Opis: Obliczanie geometrii świecy (korpus i knoty) jako baza do reguł formacji.

def candle_parts(open_, high, low, close):
    body = (close - open_).abs()
    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low
    return body, upper_wick, lower_wick


def daily_to_weekly(df, week_start="MON"):
    """
    Convert daily OHLCV data to weekly with a custom week start day.

    week_start: str
        One of: 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'
    """

    WEEKDAYS = {
        "MON": 0, "TUE": 1, "WED": 2,
        "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6
    }

    if week_start not in WEEKDAYS:
        raise ValueError("week_start must be one of MON..SUN")

    # shift index so chosen weekday becomes Monday-equivalent
    shift_days = WEEKDAYS[week_start]
    shifted = df.copy()
    shifted.index = shifted.index - pd.Timedelta(days=shift_days)

    weekly = shifted.resample("W").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    })

    # shift index back
    weekly.index = weekly.index + pd.Timedelta(days=shift_days)

    weekly.dropna(inplace=True)
    return weekly


def create_stock_dfs(
    settings: Dict,
    symbols: Optional[List[str]] = None,
    data_dir: Optional[str] = None,
    window: Optional[int] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Load daily CSVs for symbols, compute indicators and weekly aggregates.

    Returns (dfs_1d, dfs_1w) where each is a dict mapping symbol -> DataFrame.
    """
    # determine symbols list
    if symbols is None:
        market = settings.get("market", "stocks")
        if market == "crypto":
            symbols = CRYPTO_TOP_50.copy()
        elif market == "commodities":
            symbols = COMMODITY_SYMBOLS.copy()
        else:
            symbols = SYMBOLS_500.copy()

    # data dir default
    if data_dir is None:
        data_dir = os.path.join("all_data", f"data_{settings.get('market','stocks')}_1d")

    dfs_1d: Dict[str, pd.DataFrame] = {}
    dfs_1w: Dict[str, pd.DataFrame] = {}

    for symbol in tqdm(list(symbols), desc="Loading data"):
        df_daily = load_symbol_data(symbol, data_dir, window=window)
        if df_daily is None:
            continue

        # add indicators to daily
        df_daily = add_indicators(df_daily.copy(), settings=settings)
        dfs_1d[symbol] = df_daily

        # create weekly from daily using week_start setting
        week_start = settings.get("week_start", "BASE")
        if week_start != "BASE":
            df_week = daily_to_weekly(df_daily, week_start=week_start)
        else:
            # default weekly resample: week ending on Sunday
            df_week = df_daily.resample("W").agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }).dropna()

        df_week = add_indicators(df_week.copy(), settings=settings)
        dfs_1w[symbol] = df_week

    return dfs_1d, dfs_1w