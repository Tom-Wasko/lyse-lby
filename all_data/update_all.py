
# Opis: Definicja listy symboli (tickery) do analizy.


import yfinance as yf
import pandas as pd
import os
from tqdm import tqdm
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
import pytz
from pathlib import Path


BASE_DIR = Path.home() / "lyse-lby" / "all_data"

DATA_DIR = BASE_DIR

sp500 = [f.stem for f in (DATA_DIR / "data_sp500_1d").glob("*.csv")]
crypto = [f.stem for f in (DATA_DIR / "data_crypto_1d").glob("*.csv")]
europe = [f.stem for f in (DATA_DIR / "data_europe_1d").glob("*.csv")]

# -----------------------
# CONFIG
# -----------------------
os.makedirs(DATA_DIR, exist_ok=True)

OVERWRITE_DAYS = 5


# NYSE calendar
nyse = mcal.get_calendar("NYSE")

# -----------------------
# Helper Functions
# -----------------------
def latest_safe_trading_date(area):
    """
    Return the last fully closed trading date for a given area.
    """
    now = datetime.now(pytz.utc)  # use UTC for consistency

    if area.lower() == "sp500":
        cal = mcal.get_calendar("NYSE")
        today = pd.Timestamp(now.date())
        schedule = cal.schedule(start_date=today - timedelta(days=7), end_date=today)
        closed_sessions = schedule[schedule["market_close"] < now]
        if closed_sessions.empty:
            return schedule.index[-2].normalize()
        return closed_sessions.index[-1].normalize()

    elif area.lower() == "europe":
        # Approximation: use last weekday (Mon-Fri)
        last_weekday = pd.Timestamp(now.date())
        while last_weekday.weekday() >= 5:  # Sat/Sun
            last_weekday -= pd.Timedelta(days=1)
        return last_weekday.normalize()

    elif area.lower() == "crypto":
        # Crypto trades 24/7 → safe trading date is always "today"
        return pd.Timestamp(now.date()).normalize()

    else:
        raise ValueError(f"Unknown area: {area}")


def download_data(symbol, start_date, end_date):
    """Download historical OHLCV data from Yahoo for a single symbol."""
    df = yf.download(
        symbol,
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=False
    )
    if df.empty:
        return None

    # Flatten MultiIndex columns if necessary
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[0] != '' else col[1] for col in df.columns]

    df.reset_index(inplace=True)
    return df


# -----------------------
# MAIN UPDATER
# -----------------------
'''
safe_date = latest_safe_trading_date()
tqdm.write(f"Safe trading date: {safe_date.date()}")

pbar = tqdm(sp500, desc="Updating")
'''

for area in ['sp500', 'europe', 'crypto']:
    if area == 'sp500':
        lista = sp500
    if area == 'europe':
        lista = europe
    if area == 'crypto':
        lista = crypto

    pbar = tqdm(lista, desc="Updating")

    safe_date = latest_safe_trading_date(area=area)
    tqdm.write(f"Safe trading date: {safe_date.date()}")

    for symbol in pbar:
        file_path = os.path.join(DATA_DIR, f"data_{area}_1d", f"{symbol}.csv")

        # Load existing data or create empty DataFrame
        if os.path.exists(file_path):
            df_existing = pd.read_csv(file_path, parse_dates=["Date"])
            df_existing = df_existing.dropna(subset=["Date", "Close"])
        else:
            df_existing = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"])

        # Determine last date in file
        if not df_existing.empty:
            last_file_date = df_existing["Date"].max().normalize()
        else:
            last_file_date = pd.Timestamp("1990-01-01")  # start of history if empty

        # Determine safe start date for overwrite
        start_date = last_file_date - timedelta(days=OVERWRITE_DAYS)

        # Keep rows older than overwrite window
        df_existing_keep = df_existing[df_existing["Date"] < start_date]

        # Download new data
        df_new = download_data(symbol, start_date, safe_date)
        if df_new is None:
            pbar.set_postfix_str(f"{symbol}: no new data available yet")
            continue

        # Only proceed if new data extends beyond last_file_date
        df_new_last_date = df_new["Date"].max().normalize()

        #'''
        if df_new_last_date <= last_file_date:
            pbar.set_postfix_str(f"{symbol}: already up-to-date through {last_file_date.date()}")
            continue
        #'''

        # Merge and deduplicate
        df_updated = pd.concat([df_existing_keep, df_new], ignore_index=True)
        df_updated.drop_duplicates(subset=["Date"], keep="last", inplace=True)
        df_updated.sort_values("Date", inplace=True)

        last_downloaded = df_updated["Date"].max().normalize()
        pbar.set_postfix_str(f"{symbol}: {last_downloaded.date()}")

        # Save updated CSV
        df_updated.to_csv(file_path, index=False)
