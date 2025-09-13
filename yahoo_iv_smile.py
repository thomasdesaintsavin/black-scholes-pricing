# yahoo_iv_smile.py
# Usage:
#   python3 yahoo_iv_smile.py AAPL 2025-12-19 call
#   python3 yahoo_iv_smile.py MSFT 2025-01-17 put

import sys
import math
from datetime import datetime, timezone
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from black_scholes import implied_vol_call, black_scholes_call, black_scholes_put

def year_fraction(expiry_dt, now_dt=None):
    if now_dt is None:
        now_dt = datetime.now(timezone.utc)
    # convention ACT/365 simple
    return max(1e-8, (expiry_dt - now_dt).total_seconds() / (365.0 * 24 * 3600))

def get_spot(ticker):
    tk = yf.Ticker(ticker)
    # Essaye fast_info, sinon dernier close/historique
    try:
        price = tk.fast_info["last_price"]
        if price and price > 0:
            return float(price)
    except Exception:
        pass
    hist = tk.history(period="1d")
    if not hist.empty:
        return float(hist["Close"].iloc[-1])
    raise RuntimeError("Impossible de récupérer le spot.")

def fetch_chain(ticker, expiry_str):
    tk = yf.Ticker(ticker)
    if expiry_str not in tk.options:
        raise ValueError(f"Échéance {expiry_str} introuvable. Disponibles: {tk.options}")
    chain = tk.option_chain(expiry_str)
    calls = chain.calls.copy()
    puts = chain.puts.copy()
    return calls, puts

def clean_midprice(df):
    # mid = (bid+ask)/2 si possible, sinon lastPrice
    mid = df[["bid", "ask"]].mean(axis=1, skipna=True)
    mid = mid.where(mid > 0, df.get("lastPrice", pd.Series(index=df.index)))
    return mid

def compute_iv_series(options_df, side, S, Kcol="strike", pricecol="mid", T=0.25, r=0.00):
    ivs = []
    for _, row in options_df.iterrows():
        K = float(row[Kcol])
        px = float(row[pricecol])
        if not (math.isfinite(px) and px > 0 and math.isfinite(K) and K > 0):
            ivs.append(float("nan"))
            continue
        try:
            if side == "call":
                iv = implied_vol_call(price_obs=px, S=S, K=K, T=T, r=r, sigma_init=0.2)
            else:
                # astuce: put via parité -> prix_call = P + S - K e^{-rT}
                call_equiv = px + S - K * math.exp(-r * T)
                iv = implied_vol_call(price_obs=call_equiv, S=S, K=K, T=T, r=r, sigma_init=0.2)
            ivs.append(iv if 0 <= iv < 5 else float("nan"))  # filtre outliers grossiers
        except Exception:
            ivs.append(float("nan"))
    return pd.Series(ivs, index=options_df.index, name="implied_vol")

def main():
    ticker = "AAPL"
    tk = yf.Ticker(ticker)

    # Récupération des échéances disponibles
    expiries = tk.options
    print("Échéances disponibles :", expiries)

    # Choix de la première échéance disponible (tu peux changer l’index [0] en [1], [2], etc.)
    expiry = expiries[0]
    print(f"On utilise l'échéance : {expiry}")

    # Récupération de la chaîne d’options
    chain = tk.option_chain(expiry)
    calls = chain.calls

    # Approximation : on suppose que l’IV est dans 'impliedVolatility'
    strikes = calls['strike']
    iv = calls['impliedVolatility']

    # Tracé du smile
    plt.figure(figsize=(10,6))
    plt.plot(strikes, iv, 'o-', label="Smile Call")
    plt.xlabel("Strike")
    plt.ylabel("Volatilité implicite (σ)")
    plt.title(f"{ticker} – Call – Smile de volatilité ({expiry})")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
