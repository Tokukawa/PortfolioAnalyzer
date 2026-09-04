import pandas as pd
import pytest

from portfolio_analyzer.backtest import NaiveBackTest


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
