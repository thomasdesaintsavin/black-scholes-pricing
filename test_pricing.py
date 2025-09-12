import math
from black_scholes import (
    black_scholes_call, black_scholes_put,
    delta_call, delta_put, gamma, vega,
    theta_call, theta_put, rho_call, rho_put,
    monte_carlo_call, implied_vol_call
)

S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

def test_put_call_parity():
    """Parité call-put : C - P = S - K e^{-rT}"""
    C = black_scholes_call(S, K, T, r, sigma)
    P = black_scholes_put(S, K, T, r, sigma)
    rhs = S - K * math.exp(-r*T)
    assert abs((C - P) - rhs) < 1e-6

def test_mc_close_to_bs():
    """Monte Carlo doit être proche de BS (<2% d'écart)"""
    bs = black_scholes_call(S, K, T, r, sigma)
    mc = monte_carlo_call(S, K, T, r, sigma, n_simulations=200_000, seed=123)
    assert abs(mc - bs) / bs < 0.02

def test_implied_vol_recovery():
    """Vol implicite doit retrouver sigma vrai"""
    target = black_scholes_call(S, K, T, r, sigma)
    iv = implied_vol_call(target, S, K, T, r, sigma_init=0.3)
    assert abs(iv - sigma) < 1e-4

def test_greeks_basic_ranges():
    """Checks basiques sur les Greeks"""
    assert 0.0 < delta_call(S, K, T, r, sigma) < 1.0
    assert -1.0 < delta_put(S, K, T, r, sigma) < 0.0
    assert gamma(S, K, T, r, sigma) > 0.0
    assert vega(S, K, T, r, sigma) > 0.0
