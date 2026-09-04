import pandas as pd
import pytest

from portfolio_analyzer.backtest import NaiveBackTest, OutOfSampleBackTest


def test_naive_backtest_single_asset_matches_hand_calc():
    # 100 -> 110 -> 99: +10% then -10%.
    data = pd.DataFrame({"AAA": [100.0, 110.0, 99.0]})
    backtest = NaiveBackTest({"AAA": 1.0}, data, rebalance=True)
    portfolio = backtest.run(capital=1000.0)

    expected = [1000.0 * 1.10, 1000.0 * 1.10 * 0.90]
    assert portfolio["portfolio"].tolist() == pytest.approx(expected)


def test_naive_backtest_two_assets_rebalanced_daily():
    data = pd.DataFrame({"AAA": [100.0, 110.0], "BBB": [50.0, 45.0]})
    backtest = NaiveBackTest({"AAA": 0.5, "BBB": 0.5}, data, rebalance=True)
    portfolio = backtest.run(capital=100.0)

    # day-over-day return = 0.5 * 10% + 0.5 * -10% = 0%
    assert portfolio["portfolio"].iloc[0] == pytest.approx(100.0)


def test_naive_backtest_without_rebalance_buys_and_holds():
    data = pd.DataFrame({"AAA": [100.0, 200.0], "BBB": [50.0, 50.0]})
    backtest = NaiveBackTest({"AAA": 0.5, "BBB": 0.5}, data, rebalance=False)
    portfolio = backtest.run(capital=100.0)

    # buy-and-hold: 50 in AAA doubles to 100, 50 in BBB stays 50 -> 150 total
    assert portfolio["portfolio"].iloc[0] == pytest.approx(150.0)


def test_out_of_sample_backtest_equal_blocks_preserves_dataframe_splits():
    # Regression: np.split() on a DataFrame used to return bare ndarrays,
    # stripping the column labels both the optimizer and NaiveBackTest need.
    # Train block (rows 0-2) is arbitrary since naive_5050 ignores it; test
    # block (rows 3-5) grows both tickers 10% on each of its two steps.
    data = pd.DataFrame(
        {
            "AAA": [10.0, 12.0, 9.0, 100.0, 110.0, 121.0],
            "BBB": [20.0, 18.0, 21.0, 50.0, 55.0, 60.5],
        }
    )

    def naive_5050(train_data):
        return pd.DataFrame([[0.5, 0.5]], columns=train_data.columns)

    result = OutOfSampleBackTest(
        data, naive_5050, splits=2, method="equal-blocks", rebalance=True
    ).run()

    assert list(result.columns) == ["portfolio"]
    # A block's own first return has nothing to diff against (pct_change of
    # a single row is NaN -> filled to 0), so only the second step is a real
    # signal: both tickers grow 10% on that step, so does the 50/50 mix.
    assert result["portfolio"].tolist() == pytest.approx([1.0, 1.1])
