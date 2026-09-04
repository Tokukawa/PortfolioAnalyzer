import pandas as pd

from portfolio_analyzer.utils import factors, portfolio2dic


def test_portfolio2dic_converts_first_row_to_dict():
    data = pd.DataFrame([{"AAA": 0.6, "BBB": 0.4}])
    assert portfolio2dic(data) == {"AAA": 0.6, "BBB": 0.4}


def test_factors_of_prime_number():
    assert factors(13) == [1, 13]


def test_factors_of_composite_number():
    assert factors(12) == [1, 2, 3, 4, 6, 12]


def test_factors_of_perfect_square():
    assert factors(36) == [1, 2, 3, 4, 6, 9, 12, 18, 36]
