import numpy as np
import pandas as pd
import pytest

from portfolio_analyzer.filtering import l1filter, l2filter, strip_na


def test_strip_na_removes_only_leading_and_trailing_nans():
    s = pd.Series([np.nan, np.nan, 1.0, 2.0, np.nan, 3.0, np.nan])
    result = strip_na(s)
    assert result.index.tolist() == [2, 3, 4, 5]
    assert not np.isnan(result.iloc[0])
    assert not np.isnan(result.iloc[-1])
    assert np.isnan(result.iloc[2])  # internal NaN is preserved, not stripped


def test_l1filter_runs_without_outlier_removal(price_data):
    result, test_results = l1filter(price_data, delta=0.5)
    assert "AAA_filter" in result.columns
    assert "BBB_filter" in result.columns
    assert set(test_results.keys()) == set(price_data.columns)


def test_l1filter_strip_outliers_does_not_raise(price_data):
    # Regression test: strip_outliers used to collide with the module-level
    # remove_outliers() function and crash with "'bool' object is not callable".
    result, _ = l1filter(price_data, delta=0.5, strip_outliers=True)
    assert "AAA_outlier" in result.columns
    assert "BBB_outlier" in result.columns


def test_l2filter_strip_outliers_does_not_raise(price_data):
    result, _ = l2filter(price_data, delta=0.5, strip_outliers=True)
    assert "AAA_outlier" in result.columns
    assert "BBB_outlier" in result.columns
