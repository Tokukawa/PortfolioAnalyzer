# Author Pablo Zivic pablozivic@gmail.com original code https://github.com/elsonidoq/py-l1tf
from itertools import chain

import numpy as np
import pandas as pd
from cvxopt import (blas, div, lapack, matrix, mul, normal, sin, solvers,
                    spdiag, spmatrix)
from statsmodels.robust.scale import mad
from statsmodels.tsa.stattools import adfuller

solvers.options["show_progress"] = 0


def get_second_derivative_matrix(n):
    """
    Get the second order derivative matrix. Author Pablo Zivic pablozivic@gmail.com original code https://github.com/elsonidoq/py-l1tf.

    :param n: The size of the time series
    :return: A matrix D such that if x.size == (n,1), D * x is the second derivate of x
    """
    m = n - 2
    D = spmatrix(
        list(chain(*[[1, -2, 1]] * m)),
        list(chain(*[[i] * 3 for i in range(m)])),
        list(chain(*[[i, i + 1, i + 2] for i in range(m)])),
    )
    return D


def _l1tf(corr, delta):
    """
    Minimize    (1/2) * ||x-corr||_2^2 + delta * sum(y)subject to  -y <= D*x <= y Variables x (n), y (n-2). Author Pablo Zivic pablozivic@gmail.com original code https://github.com/elsonidoq/py-l1tf.

    :param x:
    :return:
    """

    n = corr.size[0]
    m = n - 2

    D = get_second_derivative_matrix(n)

    P = D * D.T
    q = -D * corr

    G = spmatrix([], [], [], (2 * m, m))
    G[:m, :m] = spmatrix(1.0, range(m), range(m))
    G[m:, :m] = -spmatrix(1.0, range(m), range(m))

    h = matrix(delta, (2 * m, 1), tc="d")

    res = solvers.qp(P, q, G, h)

    return corr - D.T * res["x"]


def l1tf(corr, delta):
    """
    Filter according l1 model. Author Pablo Zivic pablozivic@gmail.com original code https://github.com/elsonidoq/py-l1tf.

    :param corr: Corrupted signal, should be a numpy array / pandas Series
    :param delta: Strength of regularization
    :return: The filtered series
    """

    m = float(corr.min())
    M = float(corr.max())
    denom = M - m
    # if denom == 0, corr is constant
    t = (corr - m) / (1 if denom == 0 else denom)

    if isinstance(corr, np.ndarray):
        values = matrix(t)
    elif isinstance(corr, pd.Series):
        values = matrix(t.values[:])
    else:
        raise ValueError("Wrong type for corr")

    values = _l1tf(values, delta)
    values = values * (M - m) + m

    if isinstance(corr, np.ndarray):
        values = np.asarray(values).squeeze()
    elif isinstance(corr, pd.Series):
        values = pd.Series(values, index=corr.index, name=corr.name)

    return values


def remove_outliers(t, delta, mad_factor=3):
    """
    Remove outliers. Author Pablo Zivic pablozivic@gmail.com original code https://github.com/elsonidoq/py-l1tf.

    :param t: an instance of pd.Series
    :param delta: parameter for l1tf function
    """
    filtered_t = l1tf(t, delta)

    diff = t.values - np.asarray(filtered_t).squeeze()
    t = t.copy()
    t[np.abs(diff - np.median(diff)) > mad_factor * mad(diff)] = np.nan

    t = t.fillna(method="ffill").fillna(method="bfill")
    return t


def strip_na(s):
    """
    Remove the NaN from the extremes. Author Pablo Zivic pablozivic@gmail.com original code https://github.com/elsonidoq/py-l1tf.

    :param s: an instance of pd.Series
    """
    m = s.min()
    lmask = s.fillna(method="ffill").fillna(m - 1) == m - 1
    rmask = s.fillna(method="bfill").fillna(m - 1) == m - 1
    mask = np.logical_or(lmask, rmask)
    return s[np.logical_not(mask)]


def l1filter(df, delta=1.0e-1, remove_outliers=False, mad_factor=3, print_test=False):
    """
    Apply the l1tf function to the whole portfolio data optionally removing outliers.

    :param df: A pandas Dataframe
    :param delta: The delta parameter of the l1tf function
    :param remove_outliers: Whether outliers should be removed
    :param mad_factor: Strength of the outlier detection technique
    :param print_test: If true print the results of Augmented Dickey-Fuller test
    :return dataframe, test: a dataframe with orginal data en filtered data and the results of stationary test
    """
    l1tf_d = {key + "_filter": None for key in df.keys()}
    test_results = {}
    if remove_outliers:
        wo_outliers_d = {key + "_outlier": None for key in df.keys()}
    ks = df.keys()
    if isinstance(delta, float):
        delta = [delta] * len(df.columns)

    for i, k in enumerate(ks):
        t = strip_na(np.log(df[k]))
        if remove_outliers:
            t = remove_outliers(t, delta[i], mad_factor)
            wo_outliers_d[k + "_outlier"] = t
        s = l1tf(t, delta[i])
        l1tf_d[k + "_filter"] = s

        adfTest = adfuller(t - s, autolag="AIC")
        dfResults = pd.Series(
            adfTest[0:4],
            index=["ADF Test Statistic", "P-Value", "Lags Used", "Observations Used"],
        )
        # Add Critical Values
        for key, value in adfTest[4].items():
            dfResults["Critical Value (%s)" % key] = value

        if print_test:
            print("Augmented Dickey-Fuller Test Results for {}:".format(k))
            print(dfResults)

        test_results[k] = dfResults.to_frame(name=k)
    if remove_outliers:
        results = np.exp(
            pd.concat(
                [pd.DataFrame(l1tf_d), pd.DataFrame(wo_outliers_d), np.log(df)], axis=1
            )
        )
    else:
        results = np.exp(pd.concat([pd.DataFrame(l1tf_d), np.log(df)], axis=1))

    return results, test_results
