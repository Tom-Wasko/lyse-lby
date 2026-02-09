
# Opis: Definicja listy symboli (tickery) do analizy.


import yfinance as yf
import pandas as pd
import os
from tqdm import tqdm
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
import pytz


DATA_DIR = "all_data\data_stocks_1d"



# Opis: Definicja listy symboli (tickery) do analizy.

SYMBOLS_500 = ['MMM','AOS','ABT','ABBV','ACN','AMD','AES','AFL','A','APD','ABNB','AKAM','ALB','ARE','ALGN','ALLE','LNT','ALL','GOOGL','GOOG','MO','AMZN','AMCR','AEE','AEP','AXP','AIG','AMT','AWK','AMP','AME','AMGN','APH','ADI','AON','APA','APO','AAPL','AMAT','APP','APTV','ACGL','ADM','ARES','ANET','AJG','AIZ','T','ATO','ADSK','ADP','AVB','AVY','AXON','BKR','BALL','BAC','BAX','BDX','BBY','TECH','BIIB','BX','XYZ','BK','BA','BSX','BMY','AVGO','BR','BRO','BLDR','BG','BXP','CHRW','CDNS','CPT','CPB','COF','CAH','CCL','CARR','CVNA','CAT','CBOE','CBRE','CDW','COR','CNC','CNP','CF','CRL','SCHW','CHTR','CVX','CMG','CB','CHD','CI','CINF','CTAS','CSCO','C','CFG','CLX','CME','CMS','KO','CTSH','COIN','CL','CMCSA','FIX','CAG','COP','ED','STZ','CEG','COO','CPRT','GLW','CPAY','CTVA','CSGP','CTRA','CRH','CRWD','CCI','CSX','CMI','CVS','DHR','DRI','DDOG','DVA','DAY','DECK','DE','DELL','DAL','DVN','DXCM','FANG','DLR','DG','DLTR','D','DPZ','DASH','DOV','DOW','DHI','DTE','DUK','DD','ETN','EBAY','ECL','EIX','EW','EA','ELV','EME','EMR','ETR','EOG','EPAM','EQT','EFX','EQR','ERIE','ESS','EL','EG','EVRG','ES','EXC','EXE','EXPE','EXPD','EXR','XOM','FFIV','FDS','FAST','FRT','FDX','FIS','FITB','FSLR','FE','FISV','F','FTNT','FTV','FOXA','FOX','BEN','FCX','GRMN','IT','GE','GEHC','GEV','GEN','GNRC','GD','GIS','GM','GPC','GILD','GPN','GL','GDDY','GS','HAL','HIG','HAS','HCA','DOC','HSIC','HSY','HPE','HLT','HOLX','HD','HON','HRL','HST','HWM','HPQ','HUBB','HUM','HBAN','HII','IBM','IEX','ITW','INCY','IR','PODD','INTC','IBKR','ICE','IFF','IP','ISRG','IVZ','INVH','IQV','IRM','JBHT','JBL','JKHY','J','JNJ','JCI','JPM','KVUE','KDP','KEY','KEYS','KMB','KIM','KMI','KKR','KHC','KR','LHX','LH','LRCX','LW','LVS','LDOS','LEN','LII','LIN','LYV','LMT','L','LOW','LULU','LYB','MTB','MPC','MAR','MMC','MLM','MAS','MA','MTCH','MKC','MCD','MCK','MDT','MRK','META','MET','MGM','MCHP','MU','MSFT','MAA','MRNA','MOH','TAP','MDLZ','MNST','MCO','MS','MOS','MSI','NDAQ','NTAP','NFLX','NEM','NWSA','NWS','NEE','NKE','NI','NDSN','NSC','NTRS','NOC','NCLH','NRG','NUE','NVDA','NXPI','ORLY','OXY','ODFL','OMC','ON','OKE','ORCL','OTIS','PCAR','PKG','PLTR','PANW','PSKY','PH','PAYX','PAYC','PYPL','PNR','PEP','PFE','PCG','PM','PSX','PNW','PNC','POOL','PPG','PPL','PFG','PG','PGR','PLD','PRU','PEG','PTC','PSA','PHM','PWR','QCOM','DGX','Q','RL','RJF','RTX','O','REG','RF','RSG','RMD','RVTY','HOOD','ROK','ROL','ROST','RCL','SPGI','CRM','SNDK','SBAC','SLB','STX','SRE','NOW','SHW','SPG','SWKS','SJM','SW','SNA','SOLV','SO','LUV','SWK','SBUX','STT','STLD','STE','SYK','SMCI','SYF','SNPS','SYY','TMUS','TROW','TTWO','TPR','TRGP','TGT','TEL','TDY','TER','TSLA','TXN','TPL','TXT','TJX','TKO','TTD','TSCO','TT','TRV','TRMB','TFC','TYL','TSN','USB','UBER','UDR','ULTA','UNP','UAL','UPS','UHS','VLO','VTR','VLTO','VRSN','VRSK','VZ','VRTX','VTRS','VICI','V','VST','VMC','WRB','WAB','WMT','DIS','WBD','WM','WAT','WEC','WFC','WELL','WST','WDC','WY','WSM','WMB','WTW','WDAY','WYNN','XEL','XYL','YUM','ZBRA','ZBH','ZTS']



# -----------------------
# CONFIG
# -----------------------
os.makedirs(DATA_DIR, exist_ok=True)

OVERWRITE_DAYS = 2


# NYSE calendar
nyse = mcal.get_calendar("NYSE")

# -----------------------
# Helper Functions
# -----------------------
def latest_safe_trading_date():
    """Return the last fully closed NYSE trading date."""
    now = datetime.now(pytz.timezone("US/Eastern"))
    today = pd.Timestamp(now.date())
    schedule = nyse.schedule(start_date=today - timedelta(days=7), end_date=today)

    closed_sessions = schedule[schedule["market_close"] < now]
    if closed_sessions.empty:
        # fallback to previous session
        return schedule.index[-2].normalize()
    return closed_sessions.index[-1].normalize()


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
safe_date = latest_safe_trading_date()
tqdm.write(f"Safe trading date: {safe_date.date()}")

pbar = tqdm(SYMBOLS_500, desc="Updating")

for symbol in pbar:
    file_path = os.path.join(DATA_DIR, f"{symbol}.csv")

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
    if df_new_last_date <= last_file_date:
        pbar.set_postfix_str(f"{symbol}: already up-to-date through {last_file_date.date()}")
        continue

    # Merge and deduplicate
    df_updated = pd.concat([df_existing_keep, df_new], ignore_index=True)
    df_updated.drop_duplicates(subset=["Date"], keep="last", inplace=True)
    df_updated.sort_values("Date", inplace=True)

    last_downloaded = df_updated["Date"].max().normalize()
    pbar.set_postfix_str(f"{symbol}: {last_downloaded.date()}")

    # Save updated CSV
    df_updated.to_csv(file_path, index=False)
