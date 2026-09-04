import numpy as np
import pandas as pd
import pytest

from portfolio_analyzer.metrics import MainMetrics


def test_main_metrics_beta_and_correlation_are_one_when_asset_equals_benchmark(
    price_data,
):
    asset = price_data[["AAA"]].rename(columns={"AAA": "benchmark"})
    metrics = MainMetrics(benchmark=asset.copy(), frequency="daily")
    result = metrics.estimate(price_data[["AAA"]])

    assert result.loc["beta", "AAA"] == pytest.approx(1.0, abs=1e-6)
    assert result.loc["benchmark correlation", "AAA"] == pytest.approx(1.0, abs=1e-6)
    assert result.loc["alpha", "AAA"] == pytest.approx(0.0, abs=1e-6)
    assert result.loc["relative variance", "AAA"] == pytest.approx(1.0, abs=1e-6)


def test_main_metrics_returns_expected_row_labels(price_data, benchmark_data):
    metrics = MainMetrics(benchmark=benchmark_data.copy(), frequency="daily")
    result = metrics.estimate(price_data)

    expected_rows = {
        "benchmark correlation",
        "average arithmetic return",
        "average geometric return",
        "alpha",
        "beta",
        "sharpe ratio",
        "max draw down",
        "variance",
        "relative variance",
        "relative return",
        "relative draw down",
        "relative sharpe ratio",
        "theoretical optimal leverage",
    }
    assert set(result.index) == expected_rows
    assert list(result.columns) == ["AAA", "BBB"]


def test_max_drawdown_matches_hand_calc_via_public_api():
    idx = pd.bdate_range("2022-01-03", periods=4)
    data = pd.DataFrame({"X": [100.0, 120.0, 90.0, 95.0]}, index=idx)
    benchmark = pd.DataFrame({"benchmark": [100.0, 101.0, 99.0, 100.5]}, index=idx)

    metrics = MainMetrics(benchmark=benchmark, frequency="daily")
    result = metrics.estimate(data)

    assert result.loc["max draw down", "X"] == pytest.approx((90.0 - 120.0) / 120.0)
