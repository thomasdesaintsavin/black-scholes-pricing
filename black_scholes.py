import math
from scipy.stats import norm

# --- Black-Scholes formulas ---

def black_scholes_call(S, K, T, r, sigma):
    """
    Prix d'un call européen avec Black-Scholes.
    """
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    """
    Prix d'un put européen avec Black-Scholes.
    """
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# --- Greeks ---

def delta_call(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return norm.cdf(d1)

def delta_put(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return norm.cdf(d1) - 1

def gamma(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return norm.pdf(d1) / (S * sigma * math.sqrt(T))

def vega(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return S * norm.pdf(d1) * math.sqrt(T)

def theta_call(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    first = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
    second = -r * K * math.exp(-r * T) * norm.cdf(d2)
    return first + second

def theta_put(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    first = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
    second = r * K * math.exp(-r * T) * norm.cdf(-d2)
    return first + second

def rho_call(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * T * math.exp(-r * T) * norm.cdf(d2)

def rho_put(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return -K * T * math.exp(-r * T) * norm.cdf(-d2)



import numpy as np

def monte_carlo_call(S, K, T, r, sigma, n_simulations=200_000, seed=42):
    """
    Prix d'un call européen par simulation Monte Carlo.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_simulations)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0.0)
    return math.exp(-r * T) * payoff.mean()

def implied_vol_call(price_obs, S, K, T, r, sigma_init=0.2, tol=1e-8, max_iter=100):
    """
    Trouve la volatilité implicite (call) par Newton-Raphson.
    price_obs = prix observé de l'option
    """
    sigma = max(1e-6, sigma_init)
    for _ in range(max_iter):
        # f(sigma) = BS_call(sigma) - prix_observé
        f = black_scholes_call(S, K, T, r, sigma) - price_obs
        # dérivée ≈ vega
        v = vega(S, K, T, r, sigma)
        if abs(v) < 1e-12:
            break
        step = f / v
        sigma -= step
        if abs(step) < tol:
            return max(sigma, 0.0)
    return max(sigma, 0.0)



# --- Exemple d’utilisation ---
if __name__ == "__main__":
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    print("Call =", round(black_scholes_call(S, K, T, r, sigma), 2))
    print("Put  =", round(black_scholes_put(S, K, T, r, sigma), 2))
    print("Delta call =", round(delta_call(S, K, T, r, sigma), 4))
    print("Gamma =", round(gamma(S, K, T, r, sigma), 4))
    mc = monte_carlo_call(S, K, T, r, sigma, n_simulations=500_000)
    bs = black_scholes_call(S, K, T, r, sigma)
    print("Monte Carlo =", round(mc, 4), "| Black-Scholes =", round(bs, 4))
    target = black_scholes_call(S, K, T, r, sigma)  # on génère un prix "observé"
    iv = implied_vol_call(target, S, K, T, r, sigma_init=0.3)
    print("Vol implicite ≈", round(iv, 4), "(sigma vrai =", sigma, ")")
