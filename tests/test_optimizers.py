import numpy as np
import pytest

from portfolio_analyzer.optimizers import (
    approximated_max_kelly,
    calculate_risk_contribution,
    max_kelly,
    minimal_variance,
    risk_parity,
)


def test_minimal_variance_weights_sum_to_one(price_data):
    weights = minimal_variance(price_data)
    assert weights.columns.tolist() == list(price_data.columns)
    assert weights.values.sum() == pytest.approx(1.0)


def test_approximated_max_kelly_weights_sum_to_one(price_data):
    weights = approximated_max_kelly(price_data)
    assert weights.values.sum() == pytest.approx(1.0)


def test_max_kelly_respects_leverage_and_long_only(price_data):
    leverage = 1.0
    weights = max_kelly(price_data, leverage=leverage)
    assert (weights.values >= -1e-8).all()
    assert weights.values.sum() <= leverage + 1e-6


def test_risk_parity_weights_are_long_only_and_normalized(price_data):
    weights = risk_parity(price_data)
    assert weights.values.sum() == pytest.approx(1.0, abs=1e-6)
    assert (weights.values >= -1e-6).all()


def test_risk_parity_gives_every_asset_an_equal_risk_contribution(price_data):
    # This is the actual invariant risk parity optimizes for: each asset
    # contributes the same amount of portfolio risk, not necessarily the
    # same weight (that only coincides when assets have identical vol/corr).
    weights = risk_parity(price_data)
    sigma = price_data.pct_change().dropna().cov().values

    risk_contribution = np.asarray(
        calculate_risk_contribution(weights.values[0], sigma)
    ).flatten()
    assert risk_contribution[0] == pytest.approx(risk_contribution[1], rel=0.05)
