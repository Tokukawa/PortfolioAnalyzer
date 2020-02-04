import numpy as np
import pandas as pd
from scipy.optimize import minimize


def minimal_variance(data):
    """Optimize portfolio in order to minimize variance."""
    returns_data = data.pct_change().dropna()
    sigma = returns_data.cov().values
    A = 0.5 * sigma
    A = np.hstack((A, -np.ones((sigma.shape[0], 1))))
    A = np.vstack((A, np.ones((1, sigma.shape[0] + 1))))
    A[-1, -1] = 0.0
    B = np.ones((1, A.shape[0]))[0]
    w = np.dot(np.linalg.inv(A), B)
    return pd.DataFrame([w[:-1]], columns=data.columns)


def approximated_max_kelly(data):
    """Find a approximated solution of the portofolio based on kelly criterion."""
    returns_data = data.pct_change().dropna()
    mu = np.mean(returns_data.values, axis=0)
    sigma = returns_data.cov().values
    A = 0.5 * sigma
    w = np.dot(np.linalg.inv(A), mu)
    w /= w.sum()
    return pd.DataFrame([w], columns=data.columns)


def total_weight_constraint(x):
    """Put the sum of the weights in a lagrange contraint."""
    return np.sum(x) - 1.0


def long_only_constraint(x):
    """Self explanatory."""
    return x


def calculate_portfolio_var(w, sigma):
    """Calculate portfolio variance."""
    w = np.matrix(w)
    return (w * sigma * w.T)[0, 0]


def calculate_risk_contribution(w, sigma):
    """Caluclate risk contribution for every component of the portfolio."""
    w = np.matrix(w)
    theta = np.sqrt(calculate_portfolio_var(w, sigma))
    MRC = sigma * w.T
    return np.multiply(MRC, w.T) / theta


def risk_budget_objective(w, pars):
    """Objective function for the risk parity optimizer."""
    sigma = pars[0]
    risk_allocation = pars[1]
    sig_p = np.sqrt(calculate_portfolio_var(w, sigma))
    risk_target = np.asmatrix(np.multiply(sig_p, risk_allocation))
    asset_RC = calculate_risk_contribution(w, sigma)
    return sum(np.square(asset_RC - risk_target.T))[0, 0]


def risk_parity(data):
    """Optimize portfolio according to risk parity criterion."""
    returns_data = data.pct_change().dropna()
    sigma = returns_data.cov().values
    risk_allocation = np.ones((1, sigma.shape[0])) / sigma.shape[0]
    w0 = np.random.rand(1, sigma.shape[0])
    cons = (
        {"type": "eq", "fun": total_weight_constraint},
        {"type": "ineq", "fun": long_only_constraint},
    )
    res = minimize(
        risk_budget_objective,
        w0,
        args=[sigma, risk_allocation],
        method="SLSQP",
        constraints=cons,
        options={"disp": True},
        tol=1e-12,
    )
    w_rb = np.asmatrix(res.x)
    return pd.DataFrame(w_rb, columns=data.columns)
