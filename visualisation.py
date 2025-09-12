import numpy as np
import matplotlib.pyplot as plt
from black_scholes import black_scholes_call

# Paramètres fixes
S, K, T, r = 100, 100, 1, 0.05

# Volatilité de 1% à 100%
sigmas = np.linspace(0.01, 1.0, 100)
call_prices = [black_scholes_call(S, K, T, r, sigma) for sigma in sigmas]

# Tracé
plt.figure(figsize=(8,5))
plt.plot(sigmas, call_prices, label="Prix du Call")
plt.xlabel("Volatilité (σ)")
plt.ylabel("Prix du Call")
plt.title("Prix du Call en fonction de la volatilité (Black-Scholes)")
plt.legend()
plt.grid(True)
plt.show()
