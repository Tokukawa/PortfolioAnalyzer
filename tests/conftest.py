import numpy as np
import pandas as pd
import pytest


def _gbm_prices(seed, n_days, start_price, daily_vol):
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(loc=0.0002, scale=daily_vol, size=n_days)
    return start_price * np.exp(np.cumsum(log_returns))


@pytest.fixture
def price_data():
    """Two-asset synthetic price series, deterministic across runs."""
    idx = pd.bdate_range("2022-01-03", periods=120)
    return pd.DataFrame(
        {
            "AAA": _gbm_prices(seed=1, n_days=120, start_price=100.0, daily_vol=0.01),
            "BBB": _gbm_prices(seed=2, n_days=120, start_price=50.0, daily_vol=0.015),
        },
        index=idx,
    )


@pytest.fixture
def benchmark_data():
    """Single-column benchmark price series aligned with price_data's index."""
    idx = pd.bdate_range("2022-01-03", periods=120)
    return pd.DataFrame(
        {"benchmark": _gbm_prices(seed=3, n_days=120, start_price=1000.0, daily_vol=0.008)},
        index=idx,
    )
