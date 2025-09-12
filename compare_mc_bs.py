import numpy as np
import matplotlib.pyplot as plt
from black_scholes import black_scholes_call, monte_carlo_call

S, K, T, r = 100, 100, 1.0, 0.05
sigmas = np.linspace(0.05, 0.6, 15)

bs_prices = [black_scholes_call(S, K, T, r, s) for s in sigmas]
mc_prices = [monte_carlo_call(S, K, T, r, s, n_simulations=200_000) for s in sigmas]

plt.figure(figsize=(8,5))
plt.plot(sigmas, bs_prices, marker="o", label="Black-Scholes (fermé)")
plt.plot(sigmas, mc_prices, marker="x", linestyle="--", label="Monte Carlo")
plt.xlabel("Volatilité (σ)")
plt.ylabel("Prix du Call")
plt.title("Call européen : comparaison Black-Scholes vs Monte Carlo")
plt.grid(True); plt.legend()
plt.tight_layout()
plt.show()
