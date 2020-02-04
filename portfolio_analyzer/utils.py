from functools import reduce


def portfolio2dic(data):
    """Convert the result of portfolio optimization (pandas data frame) in a dictionary ready to feed a backtest."""
    return data.loc[0].to_dict()


def factors(n):
    """Return the factors of a integer."""
    factors_list = list(
        set(
            reduce(
                list.__add__,
                ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0),
            )
        )
    )
    factors_list.sort()
    return factors_list
