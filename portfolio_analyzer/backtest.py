import numpy as np
import pandas as pd

from .utils import portfolio2dic


class NaiveBackTest:
    """Perform naive backtest of the portfolio."""

    def __init__(self, tickers_ratio, data, rebalance=True):
        self.tickers_ratio = tickers_ratio
        self.tickers = tickers_ratio.keys()
        self.weights = np.array(list(tickers_ratio.values()))
        self.data = data
        self.rebalance = rebalance

    def run(self, capital=100.0):
        """Execute the back test."""
        if self.rebalance:
            log_data = np.log(
                np.sum(
                    (1 + self.data[self.tickers].pct_change()).dropna() * self.weights,
                    axis=1,
                )
            )
            historical_series = np.cumsum(log_data)
            portfolio = (np.exp(historical_series) * capital).to_frame()
        else:
            log_data = np.log((1 + self.data[self.tickers].pct_change()).dropna())
            historical_series = np.cumsum(log_data, axis=0)
            portfolio = (
                np.sum(np.exp(historical_series) * self.weights, axis=1) * capital
            ).to_frame()

        portfolio.columns = ["portfolio"]
        return portfolio


class OutOfSampleBackTest:
    """Perform out of sample back test."""

    def __init__(
        self,
        data,
        portfolio_optimization,
        splits,
        method="time-series-cv",
        past_blocks=1,
        benchmark=False,
        rebalance=True,
    ):
        self.data = data
        self.portfolio_optimization = portfolio_optimization
        self.splits = splits
        self.method = method
        self.past_blocks = past_blocks
        self.benchmark = benchmark
        self.rebalance = rebalance

    def run(self):
        """Perform the task needed for the back test."""
        data_splits = np.split(self.data, self.splits)
        out_of_samples_performance = []
        if self.method == "time-series-cv":
            for i in range(self.splits)[1:-1]:
                my_portfolio = portfolio2dic(
                    self.portfolio_optimization(pd.concat(data_splits[:i]))
                )
                backtest = NaiveBackTest(
                    my_portfolio, data_splits[i + 1], self.rebalance
                )
                out_of_samples_performance.append(backtest.run().pct_change().fillna(0))

        elif self.method == "equal-blocks":
            for i in range(self.splits)[:-1]:
                my_portfolio = portfolio2dic(
                    self.portfolio_optimization(data_splits[i])
                )
                backtest = NaiveBackTest(
                    my_portfolio, data_splits[i + 1], self.rebalance
                )
                out_of_samples_performance.append(backtest.run().pct_change().fillna(0))

        elif self.method == "asym-blocks":
            for i in range(self.splits)[self.past_blocks : -1]:
                my_portfolio = portfolio2dic(
                    self.portfolio_optimization(
                        pd.concat(data_splits[(i - self.past_blocks) : i])
                    )
                )
                backtest = NaiveBackTest(
                    my_portfolio, data_splits[i + 1], self.rebalance
                )
                out_of_samples_performance.append(backtest.run().pct_change().fillna(0))
        else:
            print(
                "I don't know the method {}. Allowed methods are: time-series-cv , equal-blocks and asym-blocks.".format(
                    self.method
                )
            )
            out_of_samples_performance.append(pd.DataFrame({"portfolio": []}))

        out_of_samples_performance = pd.concat(out_of_samples_performance)
        historical_series = np.cumsum(np.log(1 + out_of_samples_performance))
        out_of_sample_portfolio = np.exp(historical_series) * 1
        out_of_sample_portfolio.columns = ["portfolio"]
        if isinstance(self.benchmark, pd.DataFrame):
            self.benchmark.columns = ["benchmark"]
            shift_value = self.benchmark.loc[out_of_sample_portfolio.index[0]][0]
            self.benchmark = self.benchmark / shift_value
            out_of_sample_portfolio = pd.concat(
                [out_of_sample_portfolio, self.benchmark], axis=1
            ).dropna()
        return out_of_sample_portfolio
