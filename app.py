# app.py (2-FILE VERSION)
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime
import io

st.set_page_config(page_title="Dual-File Quant Analyzer", layout="wide")
st.title("Dual-File Indian Quant Analyzer")
st.markdown("**Upload `NIFTY_data.csv` + `BANKNIFTY_data.csv` → Get Stat Arb Signal**")

# === CONFIG ===
config = st.sidebar.expander("Settings", expanded=True)
with config:
    capital = st.number_input("Capital (₹)", value=500000)
    risk_pct = st.slider("Risk %", 0.5, 5.0, 2.0) / 100
    entry_z = st.slider("Entry z-score", 1.5, 3.0, 2.0)
    n_price = st.number_input("NIFTY FUT Price", value=24850)
    b_price = st.number_input("BANKNIFTY FUT Price", value=51200)

# === FILE UPLOADS ===
nifty_file = st.file_uploader("NIFTY_data.csv", type="csv")
bank_file = st.file_uploader("BANKNIFTY_data.csv", type="csv")

if nifty_file and bank_file:
    try:
        # Load and clean
        nifty = pd.read_csv(nifty_file)
        bank = pd.read_csv(bank_file)

        # Validate columns
        for name, df, prefix in [("NIFTY", nifty, "NIFTY"), ("BANKNIFTY", bank, "BANKNIFTY")]:
            req = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing = [c for c in req if c not in df.columns]
            if missing:
                st.error(f"{name} missing: {missing}")
                st.stop()

        # Merge on DateTime
        nifty['DateTime'] = pd.to_datetime(nifty['Date'] + ' ' + nifty['Time'])
        bank['DateTime'] = pd.to_datetime(bank['Date'] + ' ' + bank['Time'])
        df = pd.merge(nifty, bank, on='DateTime', suffixes=('_NIFTY', '_BANK'))
        df = df.set_index('DateTime').dropna()

        # Rename for clarity
        df.rename(columns={
            'Open_NIFTY': 'NIFTY_Open', 'High_NIFTY': 'NIFTY_High', 'Low_NIFTY': 'NIFTY_Low',
            'Close_NIFTY': 'NIFTY_Close', 'Volume_NIFTY': 'NIFTY_Volume',
            'Open_BANK': 'BANKNIFTY_Open', 'High_BANK': 'BANKNIFTY_High',
            'Low_BANK': 'BANKNIFTY_Low', 'Close_BANK': 'BANKNIFTY_Close',
            'Volume_BANK': 'BANKNIFTY_Volume'
        }, inplace=True)

        # === SAME INDICATOR LOGIC (from fixed version) ===
        # RSI(2)
        delta = df['NIFTY_Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(2).mean()
        loss = -delta.where(delta < 0, 0).rolling(2).mean()
        rs = gain / loss.replace(0, np.nan)
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
        st = pd.Series(index=df.index, dtype=float)
        trend = pd.Series(index=df.index, dtype=int)
        for i in range(7, len(df)):
            curr, prev = i, i-1
            if pd.isna(st.iloc[prev]):
                st.iloc[curr] = lower.iloc[curr]; trend.iloc[curr] = 1
            elif close.iloc[curr] > upper.iloc[prev]:
                st.iloc[curr] = max(lower.iloc[curr], st.iloc[prev])
                trend.iloc[curr] = 1
            else:
                st.iloc[curr] = min(upper.iloc[curr], st.iloc[prev])
                trend.iloc[curr] = -1
        df['Supertrend'] = st; df['Trend'] = trend

        # Volume Profile
        prices = df['NIFTY_Close'].values; volumes = df['NIFTY_Volume'].values
        hist, edges = np.histogram(prices, bins=50, weights=volumes)
        poc_idx = np.argmax(hist); poc = (edges[poc_idx] + edges[poc_idx + 1]) / 2
        total_vol = volumes.sum(); cum_vol = 0; vah, val = poc, poc
        for idx in np.argsort(hist)[::-1]:
            if cum_vol >= total_vol * 0.7: break
            left, right = edges[idx], edges[idx + 1]
            vol = hist[idx]; cum_vol += vol
            vah = max(vah, right); val = min(val, left)

        # Stat Arb
        y = df['NIFTY_Close'].values; x = df['BANKNIFTY_Close'].values
        slope, _, _, _, _ = stats.linregress(x, y)
        beta = slope
        spread = y - beta * x
        mu = np.mean(spread); sigma = np.std(spread)
        z = (spread[-1] - mu) / sigma

        # Signal
        stat_signal = 1 if z < -entry_z else -1 if z > entry_z else 0
        rsi = df['RSI_2'].iloc[-1]; price = df['NIFTY_Close'].iloc[-1]; vwap = df['VWAP'].iloc[-1]; trend = df['Trend'].iloc[-1]

        final_signal = 0; reason = "No z-signal"
        if stat_signal == 1 and not pd.isna(rsi) and rsi < 10 and price < vwap and trend == 1:
            final_signal = 1; reason = "ALL PASS"
        elif stat_signal == -1 and not pd.isna(rsi) and rsi > 90 and price > vwap and trend == -1:
            final_signal = -1; reason = "ALL PASS"

        # Trade
        if final_signal != 0:
            risk = capital * risk_pct
            contracts = max(1, int(risk / (abs(z) * 100 * 25 * 0.5)))
            n_qty = contracts * 25; b_qty = int(contracts * 15 * beta)
            side = "BUY" if final_signal == 1 else "SELL"
            opp = "SELL" if final_signal == 1 else "BUY"
            trade = f"{side} {n_qty} NIFTY25N27FUT @ ~₹{n_price:,}\n{opp} {b_qty} BANKNIFTY25N27FUT @ ~₹{b_price:,}\nRisk: ₹{int(risk):,}"
        else:
            trade = "HOLD / EXIT"

        # === DISPLAY ===
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("z-score", f"{z:+.2f}")
            st.metric("RSI(2)", f"{rsi:.1f}")
            st.metric("VWAP", f"₹{vwap:.0f:,}")
            st.metric("Trend", "UP" if trend==1 else "DOWN")
            st.metric("POC", f"₹{poc:.0f:,}")
            st.success(f"**SIGNAL: {reason}**")
            st.code(trade)

        with col2:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
            ax1.plot(spread, label='Spread'); ax1.axhline(mu, color='green', ls='--')
            ax1.set_title(f"Spread | z = {z:+.2f}"); ax1.legend(); ax1.grid()
            ax2.plot(df['NIFTY_Close'], label='NIFTY'); ax2.plot(df['VWAP'], label='VWAP', color='orange')
            ax2.plot(df['Supertrend'], label='Supertrend', color='green' if trend==1 else 'red')
            ax2.set_title("NIFTY + Indicators"); ax2.legend(); ax2.grid()
            plt.tight_layout()
            st.pyplot(fig)

        # Export
        result_df = pd.DataFrame([{
            "Time": datetime.now().strftime("%H:%M"),
            "z_score": round(z, 2), "Signal": reason, "Trade": trade.replace("\n", " | ")
        }])
        st.download_button("Export", result_df.to_csv(index=False).encode(), "signal.csv", "text/csv")

    except Exception as e:
        st...
