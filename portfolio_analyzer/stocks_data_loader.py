import pandas as pd
import yfinance as yf

_FREQUENCY_TO_INTERVAL = {
    "daily": "1d",
    "weekly": "1wk",
    "monthly": "1mo",
    "quarterly": "3mo",
}


def yahoo2pandas(
    tickers, from_date="1900-01-01", to_date="2100-12-31", frequency="daily",
    allow_null=False
):
    """Download raw stocks data from yahoo and return a pandas dataframe.
    allow_null parameter allow you to fetch data even by dates where some tickers don't exist.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    interval = _FREQUENCY_TO_INTERVAL.get(frequency, frequency)

    results = {}
    for ticker in tickers:
        history = yf.Ticker(ticker).history(
            start=from_date, end=to_date, interval=interval, auto_adjust=False
        )
        close = history["Close"]
        close.index = close.index.tz_localize(None)
        results[ticker] = close.loc[~close.index.duplicated(keep="first")]

    df = pd.DataFrame.from_dict(results)
    return df.dropna() if not allow_null else df
