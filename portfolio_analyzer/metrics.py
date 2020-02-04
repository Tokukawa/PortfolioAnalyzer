import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels import regression


class MainMetrics:
    """Compute the main metrics for asset."""

    def __init__(self, benchmark):
        self.benchmark = benchmark
        self.benchmark.columns = ["benchmark"]

    def estimate(self, data):
        """Perform the estimation of the metrics for every asset in data."""
        results = {}
        for ticker in data.columns:
            results[ticker] = self.__metrics(data[[ticker]])
        return pd.DataFrame(results)

    def __metrics(self, data):
        main_metrics = {}
        main_metrics["benchmark correlation"] = self.__market_corr(data)
        main_metrics["average return"] = self.__average_return(data)
        main_metrics["alpha"], main_metrics["beta"] = self.__alpha_beta(data)
        main_metrics["sharpe ratio"] = self.__sharpe_ratio(data)
        main_metrics["max draw down"] = self.__max_drawdown(data)
        return main_metrics

    def __market_corr(self, data):
        return (
            pd.concat([data.pct_change(), self.benchmark.pct_change()], axis=1)
            .corr()
            .values[0, 1]
        )

    def __average_return(self, data):
        year_events = self.__event_frequency(data)
        average_return = np.exp(
            np.mean(np.log((1 + data.pct_change()).dropna()))
        ).values[0]
        return average_return ** year_events - 1.0

    def __alpha_beta(self, data):
        pct_change_df = pd.concat(
            [np.log(data).diff(), np.log(self.benchmark).diff()], axis=1
        ).dropna()
        columns = set(pct_change_df.columns)
        asset_name = list(columns.difference(["benchmark"]))[0]
        x = pct_change_df["benchmark"].values
        y = pct_change_df[asset_name].values

        x = sm.add_constant(x)
        model = regression.linear_model.OLS(y, x).fit()
        alpha = model.params[0]
        beta = model.params[1]
        return alpha * self.__event_frequency(data), beta

    @staticmethod
    def __sharpe_ratio(data):
        return_data = data.pct_change().dropna()
        mu = np.mean(return_data).values[0]
        std = np.std(return_data).values[0]
        return mu / std

    @staticmethod
    def __max_drawdown(data):
        prev_high = 0.0
        max_draw = 0.0
        for index, value in data.itertuples():
            prev_high = max(prev_high, value)
            dd = (value - prev_high) / prev_high
            max_draw = min(max_draw, dd)
        return max_draw

    @staticmethod
    def __event_frequency(data):
        data_frequency = (data.index[1] - data.index[0]) / pd.offsets.Day(1)
        return 365 / data_frequency
