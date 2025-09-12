# black-scholes-pricing
Implémentation du modèle de Black-Scholes en Python avec calcul des Greeks et visualisations

# Black-Scholes Pricing Project

Implémentation en Python du modèle de Black-Scholes avec plusieurs fonctionnalités :

- Prix call & put (formule fermée)
- Greeks (Delta, Gamma, Vega, Theta, Rho)
- Simulation Monte Carlo (comparaison avec la formule fermée)
- Volatilité implicite (Newton-Raphson)
- Visualisations (prix en fonction de la volatilité, comparaison Black-Scholes vs Monte Carlo)

# Installation

Clonez le dépôt et installez les dépendances :

bash
git clone https://github.com/TonPseudo/black-scholes-pricing.git
cd black-scholes-pricing
pip install -r requirements.txt

# Formule Black-Scholes

C = S0 * Φ(d1) - K * exp(-rT) * Φ(d2)

d1 = [ln(S0/K) + (r + 0.5σ²)T] / (σ√T)
d2 = d1 - σ√T

# Utilisation

Exécutez le script principal pour tester le modèle :
python3 black_scholes.py
Call = 10.45
Put  = 5.57
Monte Carlo = 10.46 | Black-Scholes = 10.45
Vol implicite ≈ 0.2000

# Visualsations

visualisation.py : prix du call en fonction de la volatilité
compare_mc_bs.py : comparaison Monte Carlo vs Black-Scholes