from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from portfolio_analyzer.stocks_data_loader import yahoo2pandas


def _fake_history(prices, index):
    return pd.DataFrame({"Close": prices}, index=index)


@patch("portfolio_analyzer.stocks_data_loader.yf.Ticker")
def test_yahoo2pandas_combines_multiple_tickers_into_one_dataframe(mock_ticker_cls):
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="America/New_York")

    def ticker_side_effect(symbol):
        prices = {"AAA": [10.0, 11.0, 12.0], "BBB": [20.0, 21.0, 22.0]}[symbol]
        mock = MagicMock()
        mock.history.return_value = _fake_history(prices, idx)
        return mock

    mock_ticker_cls.side_effect = ticker_side_effect

    df = yahoo2pandas(["AAA", "BBB"], "2024-01-01", "2024-01-03", "daily")

    assert list(df.columns) == ["AAA", "BBB"]
    assert df["AAA"].tolist() == [10.0, 11.0, 12.0]
    # the tz-aware yfinance index must be normalized to tz-naive
    assert df.index.tz is None


@patch("portfolio_analyzer.stocks_data_loader.yf.Ticker")
def test_yahoo2pandas_accepts_a_single_ticker_as_a_plain_string(mock_ticker_cls):
    idx = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
    mock = MagicMock()
    mock.history.return_value = _fake_history([1.0, 2.0], idx)
    mock_ticker_cls.return_value = mock

    df = yahoo2pandas("AAA")

    assert list(df.columns) == ["AAA"]
    mock_ticker_cls.assert_called_once_with("AAA")


@patch("portfolio_analyzer.stocks_data_loader.yf.Ticker")
def test_yahoo2pandas_drops_rows_with_missing_tickers_unless_allow_null(mock_ticker_cls):
    idx_full = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    idx_short = pd.date_range("2024-01-02", periods=2, freq="D", tz="UTC")

    def ticker_side_effect(symbol):
        mock = MagicMock()
        if symbol == "AAA":
            mock.history.return_value = _fake_history([1.0, 2.0, 3.0], idx_full)
        else:
            mock.history.return_value = _fake_history([9.0, 10.0], idx_short)
        return mock

    mock_ticker_cls.side_effect = ticker_side_effect

    strict_df = yahoo2pandas(["AAA", "BBB"], allow_null=False)
    assert len(strict_df) == 2  # only dates where both tickers have data

    lenient_df = yahoo2pandas(["AAA", "BBB"], allow_null=True)
    assert len(lenient_df) == 3
    assert np.isnan(lenient_df["BBB"].iloc[0])
