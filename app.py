import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

from scipy.optimize import brentq
from black_scholes import (
    black_scholes_call, black_scholes_put,
    delta_call, delta_put, gamma, vega, theta_call, theta_put, rho_call, rho_put
)

st.set_page_config(page_title="Black-Scholes Pricing", layout="wide")
st.title("Black-Scholes Pricing – Interactive App")

# -----------------------------
# Onglets
# -----------------------------
tab_bs, tab_smile = st.tabs(["Black-Scholes & Greeks", "Smile (Yahoo)"])

# =========================================================
# Onglet 1 — Black-Scholes (tes éléments existants)
# =========================================================
with tab_bs:
    with st.sidebar:
        st.header("Paramètres BS")
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
    # Prix du Call vs volatilité
    sigmas = np.linspace(0.01, 1.0, 100)
    call_prices = [black_scholes_call(S, K, T, r, s) for s in sigmas]
    fig1 = plt.figure()
    plt.plot(sigmas, call_prices, label="Call")
    plt.xlabel("Volatilité (σ)"); plt.ylabel("Prix Call")
    plt.title("Prix du Call en fonction de σ (S, K, T, r fixés)")
    plt.grid(True); plt.legend()
    st.pyplot(fig1)

    # Delta Call vs Spot
    spots = np.linspace(max(1e-6, 0.2*K), 2.0*K, 100)
    deltas = [delta_call(s, K, T, r, sigma) for s in spots]
    fig2 = plt.figure()
    plt.plot(spots, deltas, label="Delta Call")
    plt.xlabel("Spot (S)"); plt.ylabel("Delta")
    plt.title("Delta du Call en fonction de S (K, T, r, σ fixés)")
    plt.grid(True); plt.legend()
    st.pyplot(fig2)

# =========================================================
# Onglet 2 — Smile Yahoo Finance
# =========================================================
@st.cache_data(ttl=300)
def get_ticker_info(ticker: str):
    tk = yf.Ticker(ticker)
    # spot
    spot = None
    try:
        spot = float(tk.fast_info["last_price"])
    except Exception:
        hist = tk.history(period="1d")
        if not hist.empty:
            spot = float(hist["Close"].iloc[-1])
    # expiries
    expiries = tk.options or []
    return spot, expiries

def implied_vol_call_brent(price_obs, S, K, T, r, sigma_low=1e-6, sigma_high=5):
    def f(sig):
        return black_scholes_call(S, K, T, r, sig) - price_obs
    try:
        return brentq(f, sigma_low, sigma_high, maxiter=500)
    except Exception:
        return np.nan

def year_fraction(pd_ts):
    # ACT/365 simple
    now = pd.Timestamp.utcnow().tz_localize(None)  # tz-naive
    return max(1e-8, (pd_ts.tz_localize(None) - now).total_seconds() / (365.0*24*3600))


with tab_smile:
    st.subheader("Smile de volatilité (données Yahoo Finance)")
    # Entrées utilisateur
    tick = st.text_input("Ticker", value="AAPL").upper().strip()
    if tick:
        spot, expiries = get_ticker_info(tick)
        if spot is None or len(expiries) == 0:
            st.warning("Impossible de récupérer le spot ou les échéances pour ce ticker.")
        else:
            exp = st.selectbox("Échéance", options=expiries, index=min(0, len(expiries)-1))
            side = st.radio("Type d'option", ["call", "put"], horizontal=True, index=0)
            min_vol = st.number_input("Filtre min midprice ($)", value=0.10, step=0.05)
            min_volu = st.number_input("Filtre min volume", value=10, step=10)

            if st.button("Calculer le smile"):
                tk = yf.Ticker(tick)
                chain = tk.option_chain(exp)
                df_raw = chain.calls if side == "call" else chain.puts
                df = df_raw.copy()
                # midprice
                df["mid"] = df[["bid","ask"]].mean(axis=1, skipna=True)
                df["mid"] = df["mid"].where(df["mid"] > 0, df.get("lastPrice"))
                # filtres de liquidité
                if "volume" in df.columns:
                    df = df[(df["mid"] >= min_vol) & (df["volume"] >= min_volu)]
                else:
                    df = df[df["mid"] >= min_vol]
                df = df.dropna(subset=["mid","strike"]).sort_values("strike")

                T_exp = year_fraction(pd.to_datetime(exp).tz_localize(None))

                # pour les puts, on convertit en prix "call équivalent" via la parité
                ivs = []
                for _, row in df.iterrows():
                    K = float(row["strike"]); px = float(row["mid"])
                    if not (math.isfinite(K) and math.isfinite(px) and px > 0):
                        ivs.append(np.nan); continue
                    if side == "call":
                        iv = implied_vol_call_brent(px, spot, K, T_exp, r=0.0)
                    else:
                        call_equiv = px + spot - K*math.exp(-0.0*T_exp)
                        iv = implied_vol_call_brent(call_equiv, spot, K, T_exp, r=0.0)
                    if 0 <= (iv if iv==iv else -1) < 5:  # iv==iv pour filtrer NaN
                        ivs.append(iv)
                    else:
                        ivs.append(np.nan)

                df["implied_vol"] = ivs
                df = df.dropna(subset=["implied_vol"])

                st.caption(f"Spot {tick} ≈ {spot:.4f} | Échéance {exp} | Options retenues : {len(df)}")
                st.dataframe(df[["contractSymbol","strike","mid","implied_vol"]].reset_index(drop=True))

                # tracé
                fig = plt.figure(figsize=(8,5))
                plt.plot(df["strike"], df["implied_vol"], marker="o")
                plt.xlabel("Strike"); plt.ylabel("Volatilité implicite (σ)")
                plt.title(f"{tick} – {side.capitalize()} – Smile ({exp})")
                plt.grid(True)
                st.pyplot(fig)
