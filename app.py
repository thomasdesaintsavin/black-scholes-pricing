import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from black_scholes import (
    black_scholes_call, black_scholes_put,
    delta_call, delta_put, gamma, vega, theta_call, theta_put, rho_call, rho_put
)

st.set_page_config(page_title="Black-Scholes Pricing", layout="centered")

st.title("Black-Scholes Pricing – Interactive App")

with st.sidebar:
    st.header("Paramètres")
    S = st.number_input("Spot (S)", min_value=0.01, value=100.0, step=1.0)
    K = st.number_input("Strike (K)", min_value=0.01, value=100.0, step=1.0)
    T = st.number_input("Maturité (T, en années)", min_value=0.0001, value=1.0, step=0.25, format="%.4f")
    r = st.number_input("Taux sans risque (r)", value=0.05, step=0.01, format="%.4f")
    sigma = st.number_input("Volatilité (σ)", min_value=0.0001, value=0.20, step=0.01, format="%.4f")

col1, col2 = st.columns(2)
with col1:
    call = black_scholes_call(S, K, T, r, sigma)
    st.metric("Prix Call (BS)", f"{call:.4f}")
with col2:
    put = black_scholes_put(S, K, T, r, sigma)
    st.metric("Prix Put (BS)", f"{put:.4f}")

st.subheader("Greeks")
g1, g2, g3 = st.columns(3)
with g1:
    st.write(f"Delta Call = {delta_call(S, K, T, r, sigma):.4f}")
    st.write(f"Delta Put  = {delta_put(S, K, T, r, sigma):.4f}")
    st.write(f"Gamma      = {gamma(S, K, T, r, sigma):.6f}")
with g2:
    st.write(f"Vega       = {vega(S, K, T, r, sigma):.4f}")
    st.write(f"Theta Call = {theta_call(S, K, T, r, sigma):.4f}")
    st.write(f"Theta Put  = {theta_put(S, K, T, r, sigma):.4f}")
with g3:
    st.write(f"Rho Call   = {rho_call(S, K, T, r, sigma):.4f}")
    st.write(f"Rho Put    = {rho_put(S, K, T, r, sigma):.4f}")

st.subheader("Courbes rapides")

# 1) Prix du Call vs Volatilité
sigmas = np.linspace(0.01, 1.0, 100)
call_prices = [black_scholes_call(S, K, T, r, s) for s in sigmas]
fig1 = plt.figure()
plt.plot(sigmas, call_prices, label="Call")
plt.xlabel("Volatilité (σ)")
plt.ylabel("Prix Call")
plt.title("Prix du Call en fonction de σ (S, K, T, r fixés)")
plt.grid(True); plt.legend()
st.pyplot(fig1)

# 2) Delta Call vs Spot
spots = np.linspace(max(1e-6, 0.2*K), 2.0*K, 100)
deltas = [delta_call(s, K, T, r, sigma) for s in spots]
fig2 = plt.figure()
plt.plot(spots, deltas, label="Delta Call")
plt.xlabel("Spot (S)")
plt.ylabel("Delta")
plt.title("Delta du Call en fonction de S (K, T, r, σ fixés)")
plt.grid(True); plt.legend()
st.pyplot(fig2)
