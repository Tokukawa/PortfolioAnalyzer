from unittest.mock import patch

import pandas as pd
import pytest

from portfolio_analyzer.balancer import Rebalance


def test_normalize_weights_makes_weights_sum_to_one():
    rebalance = Rebalance(portfolio={}, weights={"AAA": 2.0, "BBB": 2.0}, cash=0.0)
    assert rebalance.weights == {"AAA": 0.5, "BBB": 0.5}


def test_diff_computes_buy_sell_and_close_orders():
    rebalance = Rebalance(
        portfolio={"AAA": 10, "CCC": 5}, weights={"AAA": 0.5, "BBB": 0.5}, cash=0.0
    )
    diff = rebalance.diff(new_exposure={"AAA": 15, "BBB": 8})

    assert diff["AAA"] == 5  # bought 5 more
    assert diff["BBB"] == 8  # new position opened
    assert diff["CCC"] == -5  # position fully closed


@patch("portfolio_analyzer.balancer.yahoo2pandas")
def test_compute_uses_latest_quote_to_size_new_exposure(mock_yahoo2pandas):
    mock_yahoo2pandas.return_value = pd.DataFrame(
        {"AAA": [10.0, 20.0], "BBB": [10.0, 5.0]}
    )
    rebalance = Rebalance(portfolio={"AAA": 5}, weights={"AAA": 1.0}, cash=100.0)

    new_exposure, balance = rebalance.compute()

    # balance = cash + AAA quote(20) * 5 shares = 200
    assert balance == pytest.approx(200.0)
    # all weight (1.0) on AAA at price 20 -> 200 / 20 = 10 shares
    assert new_exposure["AAA"] == 10
