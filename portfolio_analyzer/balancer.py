import numpy as np
import pandas as pd
from yahoofinancials import YahooFinancials


class Rebalance:
    """Perform the computation to rebalance your portfolio."""

    def __init__(self, portfolio, weights, cash):
        """
        Initialize all the data we need.

        :param portfolio: dict with your current portfolio
        :param weights: dict with optimized weights
        :param cash: float with your free cash
        """
        self.portfolio = portfolio
        self.weights = self._normalize_weights(weights)
        self.cash = cash
        self.quotes = None
        self.balance = None

    def _normalize_weights(self, weights):
        """Define and internal method."""
        norm_factor = np.sum(np.array(list(weights.values())))
        for key in weights.keys():
            weights[key] = weights[key] / norm_factor
        return weights

    def compute(self):
        """Estimate the current exposure."""
        balance = self._get_balance(self.portfolio)
        new_exposure = self._get_new_exposure(self.weights)
        return new_exposure, balance

    def _get_balance(self, portfolio):
        """Define and internal method."""
        tickers = set(portfolio.keys()) | set(self.weights.keys())
        self.quotes = self._get_data(tickers).tail(1)
        balance = self.cash
        for ticker in portfolio.keys():
            balance += self.quotes[ticker].values[-1] * portfolio[ticker]
        self.balance = balance
        return balance

    def _get_data(self, tickers):
        """Define and internal method."""
        historical_stock_prices = YahooFinancials(tickers).get_historical_price_data(
            "2019-01-01", "2030-01-01", "daily"
        )
        results = {}
        for ticker in tickers:
            df = pd.DataFrame(historical_stock_prices[ticker]["prices"])
            df["formatted_date"] = (df["formatted_date"]).astype("datetime64[ns]")
            df = df.set_index("formatted_date")
            results[ticker] = df["close"].loc[
                ~df["close"].index.duplicated(keep="first")
            ]
        return pd.DataFrame.from_dict(results).dropna()

    def _get_new_exposure(self, weights):
        """Define and internal method."""
        new_exposure = {}
        for ticker in weights:
            new_exposure[ticker] = int(
                weights[ticker] * self.balance / self.quotes[ticker] + 0.5
            )
        return new_exposure

    def diff(self, new_exposure):
        """Compute the difference between previous exposure and current exposure."""
        diff_exposure = {}
        for ticker in set(self.portfolio) | set(new_exposure):
            if ticker in set(self.portfolio) & set(new_exposure):
                diff_exposure[ticker] = round(
                    new_exposure[ticker] - self.portfolio[ticker]
                )
            if ticker in set(self.portfolio) - set(new_exposure):
                diff_exposure[ticker] = round(-self.portfolio[ticker])
            if ticker in set(new_exposure) - set(self.portfolio):
                diff_exposure[ticker] = round(new_exposure[ticker])
        return diff_exposure
