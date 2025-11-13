# app.py
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt
import yaml
import io
import base64
from datetime import datetime

st.set_page_config(page_title="Ultimate Indian Quant Analyzer", layout="wide")
st.title("Ultimate Indian Quant Analyzer")
st.markdown("**Stat Arb + Supertrend + RSI(2) + VWAP + Volume Profile**")

# === CONFIG ===
default_config = {
    'capital': 500000,
    'risk_pct': 0.02,
    'entry_z': 2.0,
    'rsi': {'period': 2, 'oversold': 10, 'overbought': 90},
    'supertrend': {'period': 7, 'multiplier': 2.0},
    'volume_profile': {'bins': 50, 'value_area_pct': 0.7},
    'lots': {'nifty': 25, 'banknifty': 15},
    'futures': {'nifty_price': 24850, 'banknifty_price': 51200,
                'nifty_symbol': 'NIFTY25N27FUT', 'banknifty_symbol': 'BANKNIFTY25N27FUT'}
}

config = st.sidebar.expander("Settings", expanded=True)
with config:
    capital = st.number_input("Capital (₹)", value=500000)
    risk_pct = st.slider("Risk %", 0.5, 5.0, 2.0) / 100
    entry_z = st.slider("Entry z-score", 1.5, 3.0, 2.0)
    st.markdown("**Futures**")
    n_price = st.number_input("NIFTY FUT Price", value=24850)
    b_price = st.number_input("BANKNIFTY FUT Price", value=51200)

# === FILE UPLOAD ===
uploaded_file = st.file_uploader("Upload CSV (Date, Time, NIFTY_*, BANKNIFTY_*)", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        required = [f'{s}_{t}' for s in ['NIFTY', 'BANKNIFTY'] for t in ['Open','High','Low','Close','Volume']]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing[:5]}...")
            st.stop()

        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df = df.set_index('DateTime').drop(['Date', 'Time'], axis=1).dropna()

        # === INDICATORS ===
        # RSI
        delta = df['NIFTY_Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(2).mean()
        loss = -delta.where(delta < 0, 0).rolling(2).mean()
        rs = gain / loss
        df['RSI_2'] = 100 - (100 / (1 + rs))

        # VWAP
        df['TPV'] = df['NIFTY_Close'] * df['NIFTY_Volume']
        df['Date'] = df.index.date
        df['CumTPV'] = df.groupby('Date')['TPV'].cumsum()
        df['CumVol'] = df.groupby('Date')['NIFTY_Volume'].cumsum()
        df['VWAP'] = df['CumTPV'] / df['CumVol']

        # Supertrend
        high, low, close = df['NIFTY_High'], df['NIFTY_Low'], df['NIFTY_Close']
        tr = pd.DataFrame(index=df.index)
        tr['tr0'] = abs(high - low); tr['tr1'] = abs(high - close.shift()); tr['tr2'] = abs(low - close.shift())
        atr = tr.max(axis=1).rolling(7).mean()
        hl2 = (high + low) / 2
        upper = hl2 + 2.0 * atr; lower = hl2 - 2.0 * atr
        st = pd.Series(index=df.index); trend = pd.Series(index=df.index, dtype=int)
        for i in range(7, len(df)):
            curr, prev = i, i-1
            if close.iloc[curr] > upper.iloc[prev]:
                st.iloc[curr] = max(lower.iloc[curr], st.iloc[prev] if not pd.isna(st.iloc[prev]) else lower.iloc[curr])
                trend.iloc[curr] = 1
            else:
                st.iloc[curr] = min(upper.iloc[curr], st.iloc[prev] if not pd.isna(st.iloc[prev]) else upper.iloc[curr])
                trend.iloc[curr] = -1
        df['Supertrend'] = st; df['Trend'] = trend

        # Volume Profile
        prices = df['NIFTY_Close']; volumes = df['NIFTY_Volume']
        hist, edges = np.histogram(prices, bins=50, weights=volumes)
        poc_idx = np.argmax(hist); poc = (edges[poc_idx] + edges[poc_idx+1]) / 2
        total_vol = volumes.sum(); cum_vol = 0; vah, val = poc, poc
        for idx in np.argsort(hist)[::-1]:
            if cum_vol >= total_vol * 0.7: break
            left, right = edges[idx], edges[idx+1]
            vol = hist[idx]; cum_vol += vol
            vah = max(vah, right); val = min(val, left)

        # Stat Arb
        model = sm.OLS(df['NIFTY_Close'], sm.add_constant(df['BANKNIFTY_Close'])).fit()
        beta = model.params[1]
        spread = df['NIFTY_Close'] - beta * df['BANKNIFTY_Close']
        spread_diff = spread.diff().dropna()
        X = sm.add_constant(spread[:-1])
        ou = sm.OLS(spread_diff, X).fit()
        kappa = -ou.params[1]
        mu = -ou.params[0] / ou.params[1] if abs(ou.params[1]) > 1e-6 else spread.mean()
        sigma = spread_diff.std() / np.sqrt(max(kappa, 1e-6)) if kappa > 0 else spread.std()
        z = (spread.iloc[-1] - mu) / sigma

        # Signal Logic
        stat_signal = 1 if z < -entry_z else -1 if z > entry_z else 0
        rsi = df['RSI_2'].iloc[-1]; price = df['NIFTY_Close'].iloc[-1]; vwap = df['VWAP'].iloc[-1]; trend = df['Trend'].iloc[-1]

        final_signal = 0; reason = "No z-signal"
        if stat_signal == 1 and rsi < 10 and price < vwap and trend == 1:
            final_signal = 1; reason = "ALL PASS"
        elif stat_signal == -1 and rsi > 90 and price > vwap and trend == -1:
            final_signal = -1; reason = "ALL PASS"
        else:
            reasons = []
            if stat_signal == 0: reasons.append("z-score weak")
            if stat_signal == 1 and rsi >= 10: reasons.append("RSI not oversold")
            if stat_signal == -1 and rsi <= 90: reasons.append("RSI not overbought")
            if stat_signal == 1 and price >= vwap: reasons.append("Price ≥ VWAP")
            if stat_signal == -1 and price <= vwap: reasons.append("Price ≤ VWAP")
            if stat_signal == 1 and trend != 1: reasons.append("Supertrend down")
            if stat_signal == -1 and trend != -1: reasons.append("Supertrend up")
            reason = " | ".join(reasons)

        # Trade
        if final_signal != 0:
            risk = capital * risk_pct
            contracts = max(1, int(risk / (abs(z) * 100 * 25 * 0.5)))
            n_qty = contracts * 25; b_qty = int(contracts * 15 * beta)
            side = "BUY" if final_signal == 1 else "SELL"
            opp = "SELL" if final_signal == 1 else "BUY"
            trade = f"{side} {n_qty} NIFTY25N27FUT @ ~₹{n_price}\n{opp} {b_qty} BANKNIFTY25N27FUT @ ~₹{b_price}\nRisk: ₹{int(risk):,} | Target: VAH {'↑' if final_signal==1 else '↓'} {vah:.0f}"
        else:
            trade = "HOLD / EXIT"

        # === DISPLAY ===
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("z-score", f"{z:+.2f}")
            st.metric("RSI(2)", f"{rsi:.1f}")
            st.metric("VWAP", f"₹{vwap:.0f}")
            st.metric("Trend", "UP" if trend==1 else "DOWN")
            st.metric("POC", f"₹{poc:.0f}")
            st.success(f"**SIGNAL: {reason}**")
            st.code(trade)

        with col2:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
            ax1.plot(spread, label='Spread'); ax1.axhline(mu, color='green', ls='--')
            ax1.axhspan(mu - 2*sigma, mu + 2*sigma, alpha=0.2, color='gray')
            ax1.set_title(f"Spread | z = {z:+.2f}"); ax1.legend(); ax1.grid()

            ax2.plot(df['NIFTY_Close'], label='NIFTY', alpha=0.7)
            ax2.plot(df['VWAP'], label='VWAP', color='orange')
            ax2.plot(df['Supertrend'], label='Supertrend', color='green' if trend==1 else 'red')
            ax2.set_title("Price + VWAP + Supertrend"); ax2.legend(); ax2.grid()

            hist, edges = np.histogram(prices, bins=50, weights=volumes)
            ax3.barh((edges[:-1] + edges[1:])/2, hist, height=(edges[1]-edges[0]), alpha=0.7)
            ax3.axhline(poc, color='red', ls='--'); ax3.axhspan(val, vah, alpha=0.3, color='green')
            ax3.set_title("Volume Profile"); ax3.grid(axis='y')
            plt.tight_layout()
            st.pyplot(fig)

        # === EXPORT ===
        result = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "z_score": z, "RSI_2": rsi, "VWAP": vwap, "Trend": "UP" if trend==1 else "DOWN",
            "Signal": "LONG SPREAD" if final_signal==1 else "SHORT SPREAD" if final_signal==-1 else "HOLD",
            "Trade": trade.replace("\n", " | ")
        }
        result_df = pd.DataFrame([result])
        csv = result_df.to_csv(index=False).encode()
        st.download_button("Export Signal", csv, "quant_signal.csv", "text/csv")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload your CSV to start analysis.")
    st.markdown("**Required columns:** `Date`, `Time`, `NIFTY_Open/High/Low/Close/Volume`, `BANKNIFTY_Open/High/Low/Close/Volume`")
