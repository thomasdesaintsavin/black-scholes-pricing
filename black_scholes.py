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

# --- Exemple d’utilisation ---
if __name__ == "__main__":
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    print("Call =", round(black_scholes_call(S, K, T, r, sigma), 2))
    print("Put  =", round(black_scholes_put(S, K, T, r, sigma), 2))
    print("Delta call =", round(delta_call(S, K, T, r, sigma), 4))
    print("Gamma =", round(gamma(S, K, T, r, sigma), 4))
